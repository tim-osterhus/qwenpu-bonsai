"""Run the native Bonsai forward probe and compare its traces after execution."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np

from export_forward_fixture import GRAPH_NAME, GGUFReader, OUTPUT
from pack_checkpoint import hash_file
from validate import FORWARD_CUBIN, cap_processes, forward_probe_environment, sample_probe

CHECKPOINT = "Bonsai-27B-Q1_0-QwenPU-forward.gguf"
CHUNK = 16 * 1024**2


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def verify_mutation(original, copy, offset, before, after):
    """Hash both files and verify that all bytes outside the one field match."""
    if original.stat().st_size != copy.stat().st_size:
        raise RuntimeError("Mutated file size changed.")
    original_hash, copy_hash = hashlib.sha256(), hashlib.sha256()
    with original.open("rb") as left, copy.open("rb") as right:
        remaining = offset
        while remaining:
            old = left.read(min(remaining, CHUNK))
            new = right.read(len(old))
            if old != new or not old:
                raise RuntimeError("File differs before the coefficient.")
            original_hash.update(old)
            copy_hash.update(new)
            remaining -= len(old)
        if left.read(len(before)) != before or right.read(len(after)) != after:
            raise RuntimeError("Unexpected coefficient bytes.")
        original_hash.update(before)
        copy_hash.update(after)
        while old := left.read(CHUNK):
            new = right.read(len(old))
            if old != new:
                raise RuntimeError("File differs after the coefficient.")
            original_hash.update(old)
            copy_hash.update(new)
    return original_hash.hexdigest(), copy_hash.hexdigest()


def run_probe(args, report, process, logical, checkpoint, case, label, stock=False):
    runtime = args.stock_runtime if stock else args.runtime
    mode = "stock" if stock else "forward"
    out = report / label
    argv = [str(args.probe), str(runtime), str(checkpoint), str(out), mode, "128", case["digit"]]
    argv.extend(str(token) for token in case["prefix"] + case.get("suffix", []))
    code, resources, elapsed = sample_probe(
        argv, runtime, out, label, process, logical, 120, env=forward_probe_environment(),
    )
    events = [json.loads(line) for line in (out / "stdout.jsonl").read_text(encoding="utf-8").splitlines() if line]
    result = next((event for event in events if event["phase"] == "result"), {})
    architecture = next((event for event in events if event["phase"] == "architecture"), {})
    prefix = [event for event in events if event["phase"] == "prefix"]
    steps = [event for event in events if event["phase"] == "step"]
    print(f"{label}: exit {code}, {len(steps)} circuit-owned selections, {elapsed:.2f} s.", flush=True)
    return {"exit_code": code, "resources": resources, "wall_seconds": elapsed, "architecture": architecture,
            "result": result, "steps": steps, "tokens": case["prefix"] + [step["id"] for step in steps],
            "ownership": [step["prediction"] >= 0 for step in prefix + steps],
            "echo": bytes(result.get("echoed_bytes", [])).decode("utf-8", "replace")}


def logits_difference(report, label, filename):
    stock = np.fromfile(report / "stock" / filename, dtype="<f4")
    candidate = np.fromfile(report / label / filename, dtype="<f4")
    if not stock.size or stock.shape != candidate.shape or not np.isfinite(stock).all() or not np.isfinite(candidate).all():
        raise RuntimeError("Invalid ordinary-text logit comparison.")
    return {"bitwise_equal": stock.tobytes() == candidate.tobytes(),
            "max_abs_difference": float(np.max(np.abs(stock - candidate))),
            "greedy_id_equal": bool(stock.argmax() == candidate.argmax())}


def passed_forward(run):
    result = run["result"]
    layers = run["architecture"].get("chip_layers", 0)
    return (
        run["exit_code"] == 0 and layers > 0 and result.get("released") is True
        and result.get("cache_equal_during_chip") is True and result.get("position_correct") is True
        and result.get("cuda_layer_calls", 0) == layers * len(run["tokens"])
    )


def cuda_schedule(path):
    """Check recorded placement and the chip-to-Bonsai normalization edge."""
    log = path.read_text(encoding="utf-8", errors="replace")
    embeddings = re.findall(r"^.*GET_ROWS.*model\.input_embed.*$", log, flags=re.MULTILINE)
    return {
        "cuda_circuit_split": "QWENPU-CUDA" in log,
        "cuda_input_embedding": bool(embeddings)
        and all(re.search(r"model\.input_embed .*?\[CUDA\d", line) for line in embeddings),
        "cuda_chip_norm_edge": bool(re.search(
            r"\(\s*SQR\).*\[CUDA\d.*qwenpu\.kept.*\n.*\(\s*SUM_ROWS\).*\n.*qwenpu\.kept_mean_squ",
            log,
        )),
    }


def main(args):
    for name in ("output", "probe", "runtime", "stock_checkpoint", "stock_runtime"):
        setattr(args, name, getattr(args, name).resolve())
    report = args.output / "validation" / time.strftime("%Y%m%d-%H%M%S")
    report.mkdir(parents=True)
    process, _, logical = cap_processes()
    for variable in ("TEMP", "TMP", "CUDA_CACHE_PATH"):
        os.environ[variable] = str(args.output)
    os.environ["GGML_SCHED_DEBUG"] = "2"
    cases = read_json(args.output / "inputs.json")
    checkpoint = args.output / CHECKPOINT
    report_path = report / "validation.json"
    data = {"checkpoint": str(checkpoint), "cubin_sha256": hash_file(args.runtime / FORWARD_CUBIN), "runs": {}}

    def measure(label, case_index, artifact, stock=False):
        run = run_probe(args, report, process, logical, artifact, cases[case_index], label, stock)
        data["runs"][label] = run
        write_json(report_path, data)
        return run

    original = measure("original", 0, checkpoint)
    if not passed_forward(original):
        raise RuntimeError(f"Original CUDA forward did not complete. See {report}.")
    suffix = [step["id"] for step in original["steps"] if step["id"] < original["architecture"]["text_vocab"]]
    stock = run_probe(args, report, process, logical, args.stock_checkpoint, {**cases[0], "suffix": suffix},
                      "stock", stock=True)
    data["runs"]["stock"] = stock
    if stock["exit_code"]:
        write_json(report_path, data)
        raise RuntimeError(f"Stock control failed. See {report}.")
    alternate = measure("alternate_input", 1, checkpoint)
    if not passed_forward(alternate):
        raise RuntimeError(f"Alternate CUDA forward did not complete. See {report}.")

    mutation = read_json(args.output / "mutation.json")
    graph = next(tensor for tensor in GGUFReader(checkpoint).tensors if tensor.name == GRAPH_NAME)
    field_offset = mutation["graph_offset"]
    before = bytes.fromhex(mutation["original_hex"])
    after = bytes.fromhex(mutation["replacement_hex"])
    if not 0 <= field_offset <= graph.n_bytes - 4 or len(before) != 4 or len(after) != 4 or before == after:
        raise RuntimeError("Invalid single-coefficient mutation.")
    offset = graph.data_offset + field_offset
    copy = report / "mutation-copy.gguf"
    shutil.copyfile(checkpoint, copy)
    with copy.open("r+b") as stream:
        stream.seek(offset)
        if stream.read(4) != before:
            raise RuntimeError("Original coefficient bytes differ.")
        stream.seek(offset)
        stream.write(after)
    original_hash, mutated_hash = verify_mutation(checkpoint, copy, offset, before, after)
    data["mutation"] = {**mutation, "checkpoint_offset": offset, "field_bytes": 4,
                        "all_other_bytes_equal": True, "original_sha256": original_hash,
                        "mutant_sha256": mutated_hash}
    try:
        mutant = measure("mutated", 0, copy)
    finally:
        with copy.open("r+b") as stream:
            stream.seek(offset)
            stream.write(before)
        restored_hash = hash_file(copy)
        data["mutation"]["restored_sha256"] = restored_hash
        write_json(report_path, data)
    if restored_hash != original_hash:
        raise RuntimeError("Restored file hash differs from the original.")
    restored = measure("restored", 0, copy)

    # The reference is read by this verifier only, after the native runs.
    reference_path = args.output / "reference.json"
    reference = read_json(reference_path)
    data["reference_sha256"] = hash_file(reference_path)
    expected_mutant = list(reference[0]["tokens"])
    divergence = mutation["expected_first_divergence_stream_index"]
    expected_mutant[divergence] = mutation["mutated_selected_id"]
    expected_mutant[divergence + 1] = mutation["mutated_echo_id"]
    data["ordinary_logits"] = {label: logits_difference(report, label, "idle.f32") for label in
                              ("original", "mutated", "restored")}
    data["resumed_logits"] = logits_difference(report, "original", "resumed.f32")
    data["cuda_schedule"] = {label: cuda_schedule(report / label / "stderr.log")
                             for label in ("original", "alternate_input", "mutated", "restored")}
    data["checks"] = {
        "original_exact": original["tokens"] == reference[0]["tokens"] and original["echo"] == "8\n",
        "alternate_input_exact": alternate["tokens"] == reference[1]["tokens"] and alternate["echo"] == "6\n",
        "mutant_changed_selected_output": mutant["tokens"] == expected_mutant and mutant["echo"] == "0\n",
        "restored_exact": restored["tokens"] == reference[0]["tokens"] and restored["echo"] == "8\n",
        "token_ownership_matches": all(run["ownership"] == expected["ownership"] for run, expected in (
            (original, reference[0]), (alternate, reference[1]), (mutant, reference[0]), (restored, reference[0])
        )),
        "mutation_leaves_idle_logits_identical": (
            (report / "original/idle.f32").read_bytes() == (report / "mutated/idle.f32").read_bytes()
        ),
        "restored_model_logits_identical": (
            (report / "original/resumed.f32").read_bytes() == (report / "restored/resumed.f32").read_bytes()
        ),
        "stock_idle_greedy_token_equal": data["ordinary_logits"]["original"]["greedy_id_equal"],
        "stock_resumed_greedy_token_equal": data["resumed_logits"]["greedy_id_equal"],
        "cuda_and_text_state_checks": all(passed_forward(run) for run in (original, alternate, mutant, restored)),
        "cuda_schedule_has_circuit_norm_and_embeddings": all(
            all(checks.values()) for checks in data["cuda_schedule"].values()
        ),
        "reference_hash_matches_frozen_oracle": data["reference_sha256"]
        == read_json(args.output / "checkpoint-verification.json")["reference_sha256"],
    }
    data["passed"] = all(data["checks"].values())
    write_json(report_path, data)
    print(json.dumps(data["checks"], indent=2))
    print(report_path)
    return 0 if data["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--probe", type=Path, default=OUTPUT / "probe_forward.exe")
    parser.add_argument("--runtime", type=Path, default=OUTPUT / "candidate")
    parser.add_argument("--stock-checkpoint", type=Path, required=True)
    parser.add_argument("--stock-runtime", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
