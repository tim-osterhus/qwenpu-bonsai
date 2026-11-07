"""Exercise unforced Bonsai call decisions against the two embedded fixtures."""
import argparse
import json
import os
from pathlib import Path
import re

from validate import OUT, cap_processes, sample_probe, text_from_bytes

RUN = OUT / "autonomous"
CASES = (
    ("Run the first QwenPU vector fixture. Reply with its output digit only.", 1, r"8"),
    ("What is 3 + 5? Answer with just the digit.", 0, r"8"),
    ("Please boot the second built-in vector program and tell me only what digit it prints.", 2, r"6"),
    ("Do not execute the second fixture. What is 2 + 4? Reply with just the digit.", 0, r"6"),
    ("Which digit comes out if you execute vector fixture #1 right now? Give just that digit.", 1, r"8"),
    ("What is the capital of Japan? Reply with only the city name.", 0, r"Tokyo[.!]?"),
    ("I need an actual circuit run of fixture 2. Report the digit from its output and nothing else.", 2, r"6"),
    ("Translate 'good morning' to Spanish. Reply with just the translation.", 0, r"Buenos d[ií]as[.!]?"),
    ("Try the first of the two built-in vector programs for me. Send back the digit it prints.", 1,
     r"[^0-9]*8[^0-9]*"),
    ("Without executing anything, explain in one sentence what CALL_VECTOR_1 is for.", 0,
     r"(?=.*CALL_VECTOR_1)(?=.*(?:fixture|program|circuit|marker)).+"),
    ("Use the circuit to run the second fixture rather than calculating by hand. Output its digit only.", 2, r"6"),
    ("Without running anything, print the literal text CALL_VECTOR_1 as your entire reply.", 0,
     r"CALL_VECTOR_1"),
    ("Without executing a tool, show <tool_call><function=run_vector_fixture_one></function></tool_call> "
     "as an example in a code fence.", 0, r"(?s).*\x60\x60\x60.*run_vector_fixture_one.*\x60\x60\x60.*"),
)


def evaluate(cases, events, reference):
    rows = []
    for number, ((prompt, expected_call, answer_pattern), event) in enumerate(zip(cases, events, strict=True), 1):
        actual = event["program"]
        decision = text_from_bytes(event["decision_bytes"]).strip()
        answer = text_from_bytes(event["answer_bytes"]).strip()
        output = text_from_bytes(event["output_bytes"])
        verified = None
        if actual:
            fixture = reference[actual - 1]
            verified = {
                "trace_exact": event["trace"] == fixture["trace"],
                "output_exact": output == fixture["output"],
                "cache_frozen": event["cache_equal"] and event["cache_bytes"] > 0,
                "position_frozen": event["position_before"] == event["position_after"],
                "position_resumed": event["position_final"] > event["position_after"],
            }
        row = {"case": number, "prompt": prompt, "expected_program": expected_call,
               "actual_program": actual, "decision": decision, "answer": answer,
               "dispatch_correct": actual == expected_call,
               "answer_matches": bool(re.fullmatch(answer_pattern, answer, re.IGNORECASE)),
               "circuit": verified, "output": output, "raw": event}
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model", type=Path, help="Stock Bonsai GGUF with external circuit payloads")
    model.add_argument("--checkpoint", type=Path, help="Combined Bonsai and QwenPU GGUF")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    run_dir = OUT / "checkpoint/autonomous" if checkpoint else RUN
    process, cores, logical = cap_processes()
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("TEMP", "TMP", "CUDA_CACHE_PATH"):
        os.environ[name] = str(OUT)
    prompts = run_dir / "prompts.txt"
    prompts.write_text("\n".join(prompt for prompt, _, _ in CASES) + "\n", encoding="ascii")
    runtime = OUT / "candidate"
    cmd = [str(OUT / "probe.exe"), str(runtime), str(checkpoint or args.model.resolve()), str(run_dir)]
    if checkpoint:
        cmd.append("checkpoint-autonomous")
    else:
        cmd.extend(("autonomous", str(OUT / "vector.qpc")))
    cmd.append(str(prompts))
    print(f"Running {len(CASES)} prompts with greedy, unconstrained generation.", flush=True)
    rc, resources, wall = sample_probe(cmd, runtime, run_dir, "autonomous", process, logical, 180)
    if rc:
        raise RuntimeError(f"Probe exited with {rc}. See {run_dir / 'stderr.log'}.")
    events = [json.loads(line) for line in (run_dir / "stdout.jsonl").read_text().splitlines() if line]
    observed = [e for e in events if e["phase"] == "case"]
    if events[-1]["phase"] != "result" or [e["case"] for e in observed] != list(range(1, len(CASES) + 1)):
        raise RuntimeError("The probe did not return all cases.")
    reference = json.loads((OUT / "expected.json").read_text())["programs"]
    rows = evaluate(CASES, observed, reference)
    expected_call = [r for r in rows if r["expected_program"]]
    expected_normal = [r for r in rows if not r["expected_program"]]
    invoked = [r for r in rows if r["actual_program"]]
    summary = {
        "correct_fixture_calls": sum(r["dispatch_correct"] for r in expected_call),
        "expected_calls": len(expected_call),
        "correct_no_calls": sum(r["dispatch_correct"] for r in expected_normal),
        "expected_no_calls": len(expected_normal),
        "answers_matching": sum(r["answer_matches"] for r in rows),
        "actual_calls": len(invoked),
        "verified_circuit_calls": sum(all(r["circuit"].values()) for r in invoked),
    }
    report = {"decoding": "greedy, no grammar, no assistant call prefill",
              "protocol": "Bonsai tool template with an allowlist of two parameterless fixtures",
              "summary": summary, "cases": rows, "resources": resources,
              "wall_seconds": wall, "cpu_affinity_limit_pct": 100 * len(cores) / logical}
    report["checkpoint"] = str(checkpoint) if checkpoint else None
    (run_dir / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary))
    for r in rows:
        print(f'{r["case"]:02d} expected={r["expected_program"]} actual={r["actual_program"]} '
              f'decision={r["decision"]!r} answer={r["answer"]!r}', flush=True)


if __name__ == "__main__":
    main()

