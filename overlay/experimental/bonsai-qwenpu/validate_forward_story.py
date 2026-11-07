"""Compare the original stories260K guest with Bonsai's native CUDA decoder."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from export_forward_fixture import GGUFReader, GRAPH_NAME, OUTPUT, TOKENIZER, checked_tokenizer, ref, spec
from export_forward_story import (
    FORMAT, GUEST_SHA256, ONE_GUEST, REVISIONS, RUN, SNAPSHOTS, SOURCE_SHA256,
    THREE_GUEST, TOKENIZER_SHA256, checked_file, file_info, manifest_path,
)
from pack_checkpoint import hash_file
from probe_resources import ProbePreflightError
from validate import FORWARD_CUBIN, cap_processes, forward_probe_environment, sample_probe
from validate_forward import read_json, write_json
from chipformer.rv import elf

PROMPT = "Run<tool_call></tool_call>"
MAX_STEPS = 250_000
TESTS = ("first-vmat", "first-story-token", "three-story-tokens")
TEST_GUESTS = dict(zip(TESTS, ("one", "one", "three")))


def reference(guest, text, vocabulary):
    # The CPU interpreter starts at the bundled guest's ELF entry point.
    machine = ref.Machine.from_program(elf.program(guest))
    raw = []
    selected = [vocabulary + spec.tok("GO")]
    first_vmat = None
    last_vmat = None
    vmat_count = 0
    while not machine.halted and len(selected) < MAX_STEPS:
        ins = ref.decode(machine.rom[(machine.pc - spec.ROM_BASE) // 4])
        vmat = machine.n != spec.N_MAX and ins.op == "VMAT"
        start = len(raw) + 1
        record = machine.record()
        raw.extend(record)
        for token in record:
            selected.append(vocabulary + token)
            family, payload = spec.family(token)
            if family == "BYTE":
                selected.append(text.printable[payload])
            if token == spec.tok("HALT"):
                selected.append(text.NL)
        if vmat:
            vmat_count += 1
            last_vmat = {"first_record_token": start, "last_record_token": len(raw),
                         "selected_tokens_through": len(selected)}
            if first_vmat is None:
                first_vmat = last_vmat
    if (not machine.halted or machine.fault or first_vmat is None
            or machine.exit_code != 0 or len(selected) > MAX_STEPS):
        raise RuntimeError(f"Reference guest failed: {guest.name}, fault={machine.fault}")
    return (np.asarray(raw, dtype="<i4"), np.asarray(selected, dtype="<i4"), first_vmat,
            last_vmat, vmat_count, bytes(machine.out))


def run_probe(probe, runtime, checkpoint, out, key, prefix, vocabulary, selected, raw,
              expected_bytes, released, process, logical, timeout):
    cmd = [str(probe), str(runtime), str(checkpoint), str(out), "story", str(len(selected)), key,
           *map(str, prefix)]
    print(f"{out.name}: CUDA run for {len(selected):,} circuit selections.", flush=True)
    code, resources, wall = sample_probe(
        cmd, runtime, out, out.name, process, logical, timeout, env=forward_probe_environment(),
    )
    events = read_json_lines(out / "stdout.jsonl")
    prefix_events = [e for e in events if e["phase"] == "prefix"]
    steps = [e for e in events if e["phase"] == "step"]
    result = next((e for e in events if e["phase"] == "result"), {})
    layers = next((e["chip_layers"] for e in events if e["phase"] == "architecture"), 0)
    actual = np.asarray([e["id"] for e in steps], dtype="<i4")
    native_raw = actual[actual >= vocabulary] - vocabulary
    # GO belongs to the text bridge rather than to an RV instruction record.
    native_raw = native_raw[1:] if len(native_raw) and native_raw[0] == spec.tok("GO") else native_raw
    actual.tofile(out / "selected.i32")
    native_raw.tofile(out / "trace.i32")
    common = min(len(selected), len(actual))
    differences = np.flatnonzero(selected[:common] != actual[:common])
    mismatch = int(differences[0]) if len(differences) else (common if len(selected) != len(actual) else None)
    elapsed = result.get("elapsed_seconds")
    echoed = bytes(result.get("echoed_bytes", []))
    checks = {
        "exact_bridge_trace": mismatch is None,
        "exact_instruction_trace": np.array_equal(native_raw, raw),
        "first_prediction_from_tool_close": [e["prediction"] for e in prefix_events] == [-1, -1, int(selected[0])],
        "free_running_selection": len(prefix_events) == len(prefix) and all(
            prev["prediction"] == step["id"] for prev, step in zip(prefix_events[-1:] + steps, steps)),
        "cuda_layers_executed": layers == 23 and result.get("cuda_layer_calls") == layers * (len(prefix) + len(steps)),
        "cache_and_position_preserved": result.get("cache_equal_during_chip") is True
        and result.get("position_correct") is True,
        "ownership_boundary": result.get("released") is released,
        "expected_echo": bool(result) and echoed == expected_bytes,
    }
    run = {"exit_code": code, "checks": checks, "passed": code == 0 and all(checks.values()),
           "expected_record_tokens": len(raw), "actual_record_tokens": len(native_raw),
           "expected_selected_tokens": len(selected), "actual_selected_tokens": len(actual),
           "first_difference": None if mismatch is None else {
               "position": mismatch + 1,
               "expected": int(selected[mismatch]) if mismatch < len(selected) else None,
               "actual": int(actual[mismatch]) if mismatch < len(actual) else None,
           },
           "echo": echoed.decode("utf-8", "replace"), "wall_seconds": wall,
           "record_tokens_per_second": len(native_raw) / elapsed if elapsed else None,
           "selected_tokens_per_second": len(actual) / elapsed if elapsed else None,
           "native": result, "resources": resources}
    print(f"{out.name}: {len(native_raw):,}/{len(raw):,} record tokens, "
          f"{run['record_tokens_per_second']} records/s, pass={run['passed']}.", flush=True)
    return run


def read_json_lines(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify_checkpoint(checkpoint, manifest):
    if manifest["format"] != FORMAT or manifest["revisions"] != REVISIONS:
        raise ValueError("Unexpected story checkpoint format or source revisions.")
    if manifest["stock_gguf"]["sha256"] != SOURCE_SHA256:
        raise ValueError("The manifest does not identify the pinned Bonsai model.")
    actual = checked_file(checkpoint, manifest["checkpoint"]["sha256"])
    if actual != manifest["checkpoint"]:
        raise ValueError("Checkpoint size differs from the manifest.")
    reader = GGUFReader(checkpoint)
    names = {GRAPH_NAME, *SNAPSHOTS.values()}
    payloads = {
        t.name: {"bytes": t.n_bytes, "sha256": hashlib.sha256(t.data).hexdigest()}
        for t in reader.tensors if t.name in names
    }
    if payloads.keys() != names or payloads != manifest["payloads"]:
        raise ValueError("The embedded graph or program snapshots differ from the manifest.")
    return reader


def references(args, out, manifest, text, vocabulary):
    guests = {"one": args.one_guest, "three": args.three_guest}
    needed = {TEST_GUESTS[label] for label in args.tests}
    if args.previous_one_reference:
        needed.add("one")
    refs, metadata = {}, {}
    for key, guest in guests.items():
        if key not in needed:
            continue
        guest_info = checked_file(guest, GUEST_SHA256[key])
        if guest_info != manifest["guests"][key]:
            raise ValueError(f"The {key} guest differs from the manifest.")
        raw, selected, first, last, vmat_count, emitted = refs[key] = reference(guest, text, vocabulary)
        raw_path = out / f"{key}-reference.i32"
        selected_path = out / f"{key}-reference-selected.i32"
        raw.tofile(raw_path)
        selected.tofile(selected_path)
        if not emitted.endswith(b"\n") or not emitted[:-1]:
            raise ValueError(f"The {key} guest did not emit a story piece.")
        metadata[key] = {"elf": guest.name, "elf_sha256": guest_info["sha256"],
                         "reference_sha256": hash_file(raw_path),
                         "selected_reference_sha256": hash_file(selected_path),
                         "record_tokens": len(raw), "selected_tokens": len(selected),
                         "first_vmat": first, "last_vmat": last, "vmat_count": vmat_count,
                         "output": emitted.decode("utf-8")}
    if "one" in refs and "three" in refs and not refs["three"][-1].startswith(refs["one"][-1][:-1]):
        raise ValueError("The three-token reference does not begin with the one-token story.")
    if args.previous_one_reference and args.previous_one_reference.read_bytes() != refs["one"][0].tobytes():
        raise ValueError("The one-token reference differs from the optional archived trace.")
    return refs, metadata


def validate(args):
    args.report_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S-"), dir=args.report_root)).resolve()
    report_path = out / "validation.json"
    print(f"Report directory: {out}", flush=True)
    data = {
        "checkpoint": args.checkpoint.name,
        "prompt": PROMPT,
        "requested_tests": [label for label in TESTS if label in args.tests],
        "tests": {label: {"status": "SKIP", "reason": "Not reached" if label in args.tests else "Not requested"}
                  for label in TESTS},
        "timeouts_seconds": dict(zip(TESTS, (args.first_vmat_timeout, args.one_timeout, args.three_timeout))),
    }
    current = None
    try:
        checkpoint_manifest = args.manifest or manifest_path(args.checkpoint)
        manifest = read_json(checkpoint_manifest)
        reader = verify_checkpoint(args.checkpoint, manifest)
        data["artifact"] = manifest
        data["manifest_sha256"] = hash_file(checkpoint_manifest)
        data["checkpoint_sha256_before"] = manifest["checkpoint"]["sha256"]
        if checked_file(args.tokenizer, TOKENIZER_SHA256) != manifest["tokenizer"]:
            raise ValueError("Tokenizer differs from the manifest.")
        text, vocabulary = checked_tokenizer(reader, args.tokenizer)
        del reader
        prefix = data["prefix"] = text.encode(PROMPT)
        refs, data["guests"] = references(args, out, manifest, text, vocabulary)
        if args.previous_one_reference:
            data["archived_one_reference_sha256"] = hash_file(args.previous_one_reference)

        data["probe"] = {"name": args.probe.name, **file_info(args.probe)}
        for name in ("llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cuda.dll", FORWARD_CUBIN):
            if not (args.runtime / name).is_file():
                raise FileNotFoundError(f"Required runtime file missing: {args.runtime / name}")
        data["runtime"] = {path.name: file_info(path) for path in sorted(args.runtime.iterdir())
                           if path.is_file() and path.suffix.lower() in (".dll", ".cubin")}
        process, _, logical = cap_processes()
        for name in ("TEMP", "TMP", "CUDA_CACHE_PATH"):
            os.environ[name] = str(out)
        os.environ.pop("GGML_SCHED_DEBUG", None)

        rate = None
        for label in data["requested_tests"]:
            current = label
            key = TEST_GUESTS[label]
            raw, selected, first, _, _, emitted = refs[key]
            released = label != "first-vmat"
            if not released:
                raw = raw[:first["last_record_token"]]
                selected = selected[:first["selected_tokens_through"]]
            timeout = data["timeouts_seconds"][label]
            estimate = len(selected) / rate if rate else None
            if estimate is not None and estimate > timeout:
                data["tests"][label] = {"status": "SKIP", "reason": "Projected probe time exceeds timeout",
                                        "estimated_seconds": estimate}
                continue
            try:
                result = run_probe(args.probe, args.runtime, args.checkpoint, out / label, key,
                                   prefix, vocabulary, selected, raw, emitted + b"\n" if released else b"",
                                   released, process, logical, timeout)
            except ProbePreflightError as error:
                data["tests"][label] = {"status": "SKIP", "reason": str(error)}
                break
            except subprocess.TimeoutExpired:
                data["tests"][label] = {"status": "FAIL", "reason": f"Probe timed out after {timeout} seconds"}
                break
            data["tests"][label] = {"status": "PASS" if result["passed"] else "FAIL", **result}
            write_json(report_path, data)
            if not result["passed"]:
                break
            rate = result["selected_tokens_per_second"]
        current = None
        data["checkpoint_sha256_after"] = hash_file(args.checkpoint)
        if data["checkpoint_sha256_after"] != data["checkpoint_sha256_before"]:
            raise RuntimeError("Checkpoint bytes changed during validation.")
    except Exception as error:
        data["error"] = str(error)
        if current:
            data["tests"][current] = {"status": "FAIL", "reason": str(error)}
        print(f"Validation failed: {error}", flush=True)

    outcomes = [data["tests"][label]["status"] for label in data["requested_tests"]]
    if "error" in data or "FAIL" in outcomes:
        data["status"] = "FAIL"
    elif "SKIP" in outcomes:
        data["status"] = "SKIP"
    else:
        data["status"] = "PASS"
    data["full_suite_passed"] = data["status"] == "PASS" and all(
        case["status"] == "PASS" for case in data["tests"].values()
    )
    write_json(report_path, data)
    for label, result in data["tests"].items():
        reason = f" ({result['reason']})" if "reason" in result else ""
        print(f"{label}: {result['status']}{reason}")
    print(f"{data['status']} - {report_path}")
    return {"PASS": 0, "FAIL": 1, "SKIP": 2}[data["status"]]


def positive_seconds(value):
    seconds = int(value)
    if seconds < 1:
        raise argparse.ArgumentTypeError("Timeout must be a positive number of seconds.")
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Override the checkpoint's sibling .manifest.json")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--one-guest", type=Path, default=ONE_GUEST)
    parser.add_argument("--three-guest", type=Path, default=THREE_GUEST)
    parser.add_argument("--previous-one-reference", type=Path, help="Optional archived raw .i32 trace")
    parser.add_argument("--runtime", type=Path, default=OUTPUT / "candidate")
    parser.add_argument("--probe", type=Path, default=OUTPUT / "probe_forward.exe")
    parser.add_argument("--report-root", type=Path, default=RUN)
    parser.add_argument("--tests", nargs="+", choices=TESTS, default=TESTS)
    parser.add_argument("--first-vmat-timeout", type=positive_seconds, default=150)
    parser.add_argument("--one-timeout", type=positive_seconds, default=360)
    parser.add_argument("--three-timeout", type=positive_seconds, default=960)
    args = parser.parse_args()
    for name in ("checkpoint", "runtime", "probe"):
        setattr(args, name, getattr(args, name).resolve())
    return validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
