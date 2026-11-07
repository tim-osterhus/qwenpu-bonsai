"""Validate the one-token stories260K guest in Bonsai's embedded QwenPU graph."""
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess

import numpy as np

from validate import OUT, cap_processes, sample_probe, text_from_bytes
from export_fixture import core, export_state, ref, spec
from chipformer.rv import elf

GUEST = OUT.parent / "qwenpu-trace/stories_vx.elf"
RUN = OUT / "stories260k"
LIMIT = 150_000
PRACTICAL_SECONDS = 300
CPU_SAMPLE_NOTE = "The first worker CPU sample includes time spent preparing the reference in Python."


def reference(run_dir, embedded):
    program = elf.program(GUEST)
    machine = ref.Machine.from_program(program)
    if not embedded:
        circuit = core.build(vx=True)
        core.snapshot(circuit, machine)
        levels = [[node for node in circuit.nodes if node.level == level]
                  for level in sorted({node.level for node in circuit.nodes})]
        export_state(run_dir / "story.qps", levels)
    trace = []
    first_vmat = None
    while not machine.halted and len(trace) < LIMIT:
        current = ref.decode(machine.rom[(machine.pc - spec.ROM_BASE) // 4])
        offset = len(trace)
        refresh = machine.n == spec.N_MAX
        trace.extend(machine.record())
        if first_vmat is None and not refresh and current.op == "VMAT":
            first_vmat = (offset + 1, len(trace))
    if not first_vmat or first_vmat[1] > 5000 or not machine.halted or machine.fault:
        raise RuntimeError("The reference guest did not reach VMAT and halt within the probe bounds.")
    output = bytes(machine.out)
    if not output.endswith(b"\n") or not output[:-1]:
        raise RuntimeError("The one-token reference guest did not print a story piece.")
    expected = np.asarray(trace, dtype="<i4")
    expected.tofile(run_dir / "reference.i32")
    return expected, first_vmat, output


def run(run_dir, model, checkpoint, label, expected, output, process, logical, timeout):
    directory = run_dir / label
    runtime = OUT / "candidate"
    cmd = [str(OUT / "probe.exe"), str(runtime), str(checkpoint or model), str(directory)]
    if checkpoint:
        cmd.append("checkpoint-story")
    else:
        cmd.extend(("story", str(OUT / "vector.qpc"), str(run_dir / "story.qps")))
    cmd.append(str(expected.size))
    print(f"{label}: launching {expected.size:,} trace steps.", flush=True)
    returncode, resources, wall = sample_probe(cmd, runtime, directory, label, process, logical, timeout)
    path = directory / "trace.i32"
    actual = np.fromfile(path, dtype="<i4") if path.exists() else np.empty(0, dtype="<i4")
    shared = min(expected.size, actual.size)
    mismatch = np.flatnonzero(actual[:shared] != expected[:shared])
    if mismatch.size:
        first = int(mismatch[0])
        diff = {"position": first + 1, "expected": spec.name(int(expected[first])),
                "actual": int(actual[first])}
        if 0 <= actual[first] < spec.VOCAB:
            diff["actual"] = spec.name(int(actual[first]))
    elif actual.size != expected.size:
        diff = {"position": shared + 1, "reason": "trace lengths differ"}
    else:
        diff = None
    events = [json.loads(line) for line in (directory / "stdout.jsonl").read_text().splitlines() if line]
    result = next((event for event in events if event["phase"] == "result"), None)
    data = {"returncode": returncode, "expected_trace_tokens": int(expected.size),
            "actual_trace_tokens": int(actual.size), "first_difference": diff,
            "resources": resources, "wall_seconds": wall, "result": result}
    if result:
        text = text_from_bytes(result["output_bytes"])
        data.update(output=text, trace_tokens_per_second=actual.size / result["circuit_seconds"],
                    passed=returncode == 0 and diff is None and text == output.decode("utf-8")
                    and result["cache_equal"] and result["position_before"] == result["position_after"])
    else:
        data["passed"] = False
    print(json.dumps({key: data[key] for key in (
        "passed", "actual_trace_tokens", "first_difference", "trace_tokens_per_second", "output")
        if key in data}, ensure_ascii=False), flush=True)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model", type=Path, help="Stock Bonsai GGUF with external circuit payloads")
    model.add_argument("--checkpoint", type=Path, help="Combined Bonsai and QwenPU GGUF")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    model_path = args.model.resolve() if args.model else None
    run_dir = OUT / "checkpoint/stories260k" if checkpoint else RUN
    process, cores, logical = cap_processes()
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("TEMP", "TMP", "CUDA_CACHE_PATH"):
        os.environ[name] = str(OUT)
    expected, (first, last), output = reference(run_dir, checkpoint is not None)
    report = {"first_vmat": {"first_trace_position": first, "last_trace_position": last},
              "reference_complete_trace_tokens": int(expected.size),
              "reference_output": output.decode("utf-8"), "cpu_affinity_limit_pct": 100 * len(cores) / logical,
              "cpu_sampling_note": CPU_SAMPLE_NOTE, "checkpoint": str(checkpoint) if checkpoint else None}
    report["through_first_vmat"] = prefix = run(
        run_dir, model_path, checkpoint, "first-vmat", expected[:last], b"", process, logical, 150)
    report_path = run_dir / "validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not prefix["passed"]:
        return 1
    estimate = expected.size / prefix["trace_tokens_per_second"]
    report["estimated_full_circuit_seconds"] = estimate
    if estimate <= PRACTICAL_SECONDS:
        try:
            report["first_story_token"] = run(
                run_dir, model_path, checkpoint, "first-story-token", expected, output, process, logical, 360)
        except subprocess.TimeoutExpired as error:
            report["first_story_token"] = {"passed": False, "error": str(error)}
    else:
        report["first_story_token"] = {"skipped": "estimated circuit time exceeds 300 seconds"}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 1 if report["first_story_token"].get("passed") is False else 0


if __name__ == "__main__":
    sys.exit(main())

