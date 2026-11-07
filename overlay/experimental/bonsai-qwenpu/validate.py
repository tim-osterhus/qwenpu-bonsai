"""Launch the two native Bonsai probes only when explicitly invoked."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil

from probe_resources import ProbePreflightError, ResourceSampler, cap_processes, gpu_memory

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dist/local-bench/bonsai-in-graph"
RUN = OUT / "two-program"
FORWARD_CUBIN = "qwenpu-forward-cuda.cubin"


def forward_probe_environment():
    # Use the cubin beside llama.dll and the backends in the supplied runtime directory.
    env = os.environ.copy()
    for name in ("QWENPU_FORWARD_CUBIN", "GGML_BACKEND_PATH"):
        env.pop(name, None)
    return env


def sample_probe(cmd, runtime, directory, label, process, logical, timeout, *, env=None):
    directory.mkdir(parents=True, exist_ok=True)
    psutil.cpu_percent()
    time.sleep(2)
    used, free = gpu_memory()
    memory = psutil.virtual_memory()
    baseline = {
        "system_cpu_pct": psutil.cpu_percent(),
        "gpu_mib": used,
        "system_ram_used_bytes": memory.used,
        "available_ram_bytes": memory.available,
    }
    if baseline["system_cpu_pct"] > 80 or memory.available < 8 * 1024**3 or free < 5500:
        raise ProbePreflightError(f"{label} preflight requires CPU below 80%, 8 GiB free RAM and 5500 MiB free VRAM.")
    monitor = ResourceSampler(process, logical, baseline)
    monitor.stage = label
    started = time.perf_counter()
    try:
        with (directory / "stdout.jsonl").open("w", encoding="utf-8") as stdout, \
             (directory / "stderr.log").open("w", encoding="utf-8") as stderr:
            status = subprocess.run(cmd, cwd=runtime, stdout=stdout, stderr=stderr,
                                    timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW, env=env)
    finally:
        resources = monitor.summary()
    return status.returncode, resources, time.perf_counter() - started


def run_probe(run_dir, label, runtime, model, process, logical, embedded=False):
    directory = run_dir / label
    mode = "checkpoint" if embedded else label
    cmd = [str(OUT / "probe.exe"), str(runtime), str(model), str(directory), mode]
    if label == "candidate" and not embedded:
        cmd.append(str(OUT / "vector.qpc"))
    returncode, resources, seconds = sample_probe(cmd, runtime, directory, label, process, logical, 150)
    if returncode:
        raise RuntimeError(f"{label} exited with {returncode}. See {directory / 'stderr.log'}.")
    events = [json.loads(line) for line in (directory / "stdout.jsonl").read_text().splitlines() if line]
    programs = [event for event in events if event["phase"] == "program"]
    if events[-1]["phase"] != "result" or [p["program"] for p in programs] != [1, 2]:
        raise RuntimeError(f"{label} did not produce its report.")
    return {"programs": programs, "resources": resources,
            "wall_seconds": seconds}


def logits_comparison(run_dir, filename):
    baseline_bytes = (run_dir / "baseline" / filename).read_bytes()
    candidate_bytes = (run_dir / "candidate" / filename).read_bytes()
    a = np.frombuffer(baseline_bytes, dtype="<f4")
    b = np.frombuffer(candidate_bytes, dtype="<f4")
    if not a.size or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError(f"Invalid logit arrays for {filename}")
    return {"bitwise_equal": baseline_bytes == candidate_bytes,
            "vocabulary_size": a.size, "max_abs_diff": float(np.max(np.abs(a - b)))}


def text_from_bytes(values):
    return bytes(value & 255 for value in values).decode("utf-8", "replace")


def verify_program(run_dir, stock, actual, expected, expected_answer):
    number = actual["program"]
    idle = logits_comparison(run_dir, f"idle-{number}.f32")
    continuation = logits_comparison(run_dir, f"continuation-{number}.f32")
    output = text_from_bytes(actual["output_bytes"])
    answer = text_from_bytes(actual["answer_bytes"]).strip()
    return {
        "program": number,
        "trace_tokens": len(actual["trace"]),
        "output": output,
        "answer": answer,
        "idle_logits": idle,
        "continuation_logits": continuation,
        "checks": {
            "trace_exact": actual["trace"] == expected["trace"],
            "vector_output": output == expected["output"],
            "cache_frozen": actual["cache_equal"] and actual["cache_bytes"] > 0,
            "position_frozen": stock["position_before"] == actual["position_before"]
            == actual["position_after"] and actual["position_before"] >= 0,
            "position_resumed": actual["position_final"] == stock["position_final"]
            and actual["position_final"] > actual["position_after"],
            "idle_logits_bitwise": idle["bitwise_equal"],
            "continuation_logits_bitwise": continuation["bitwise_equal"],
            "answer_correct": answer == expected_answer,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Combined Bonsai and QwenPU GGUF for the candidate")
    args = parser.parse_args()
    process, cores, logical = cap_processes()
    for name in ("TEMP", "TMP", "CUDA_CACHE_PATH"):
        os.environ[name] = str(OUT)
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    run_dir = OUT / "checkpoint/two-program" if checkpoint else RUN
    print("Running stock Bonsai control.", flush=True)
    baseline = run_probe(run_dir, "baseline", args.baseline.resolve(), args.model.resolve(), process, logical)
    print("Running Bonsai with its private QwenPU graph.", flush=True)
    candidate = run_probe(run_dir, "candidate", OUT / "candidate", checkpoint or args.model.resolve(),
                          process, logical, embedded=checkpoint is not None)
    expected = json.loads((OUT / "expected.json").read_text())
    answers = ("8", "86")
    programs = [verify_program(run_dir, stock, actual, reference, answer)
                for stock, actual, reference, answer in zip(
                    baseline["programs"], candidate["programs"], expected["programs"], answers,
                    strict=True)]
    first, second = candidate["programs"]
    dirty = expected["contaminated_second"]
    checks = {
        "both_programs_passed": all(all(p["checks"].values()) for p in programs),
        "context_carried_between_programs": second["position_before"] > first["position_final"],
        "second_program_matches_clean_state": programs[1]["checks"]["trace_exact"]
        and programs[1]["checks"]["vector_output"] and second["trace"] != dirty["trace"]
        and programs[1]["output"] != dirty["output"],
    }
    report = {"passed": all(checks.values()), "checks": checks, "programs": programs,
              "contaminated_second_output": dirty["output"],
              "cpu_affinity_limit_pct": 100 * len(cores) / logical,
              "baseline": baseline, "candidate": candidate}
    report["checkpoint"] = str(checkpoint) if checkpoint else None
    (run_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"checks": checks, "programs": programs}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

