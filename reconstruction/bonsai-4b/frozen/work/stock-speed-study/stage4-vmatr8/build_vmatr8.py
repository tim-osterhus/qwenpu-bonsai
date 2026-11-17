"""Add a signed-byte VMATR multiply to the exact VEW16 checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage4-dot64-later"))
from build_vew16 import BF16, F16, F32, M, S, dot, gguf, merge, narrow_vew, np, render


BASE_SHA = "8f335ef91555304dcc62f44fbece0185c48499c7c18626b5d1c19b87f597178d"
CALLER, PRODUCT = "VX_R_MUL", "VX_R_PRODUCT"


def circuit_transform(c):
    narrow_vew(c)
    if c.routes[CALLER] != {"M_MUL_INIT": None}:
        raise ValueError("Unexpected VMATR signed-byte multiply route")
    effects = c.effects[CALLER]
    right = c.vx.fields["tile_products"][32:64]
    if any(effects.get(S.RS2 + bit) != S.buffer(S.WORK + lane) for bit, lane in enumerate(right)):
        raise ValueError("Wrong VMATR right operand input")
    # Initialize the original loop at iteration 24 using the zero-extended byte.
    # Existing signed corrections use RS2, encoded as the signed byte << 24.
    c.move(CALLER, M.M_LOW, c.vx.resize(c.vx.state(right[:8])))
    c.fix(CALLER, M.M_HIGH, 0)
    c.fix(CALLER, M.M_COUNT, 24, 5)
    for bit in range(24):
        effects[S.RS2 + bit] = S.constant(-1)
    c.move(CALLER, S.RS2 + 24, c.vx.state(right[:8]))
    c.routes[CALLER].clear()
    c.route(CALLER, "M_MUL_PICK")
    # The accumulator add is the only consumer and uses product bits 0–31.
    if any(c.effects[PRODUCT].get(S.B + bit) != S.buffer(S.WORK + M.M_LOW + bit) for bit in range(32)):
        raise ValueError("Unexpected VMATR product consumer")
    c.move(PRODUCT, S.B, (*S.bits(S.WORK + M.M_LOW + 24, 8), *S.bits(S.WORK + M.M_HIGH, 24)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    control = json.loads(args.baseline.with_suffix(".manifest.json").read_text())
    if control["sha256"] != BASE_SHA:
        raise ValueError("Unexpected VEW16 baseline manifest")
    _, (ids, scores, gate, up, down, rows), _, _ = dot.circuit(circuit_transform)
    if ids != control["tokens"]:
        raise ValueError("Action token IDs changed")
    report = dict(source_sha256=BASE_SHA, ffn_rows_used=rows, ffn_rows_allocated=21248, tokens=ids,
                  dot_actions=control["dot_actions"], width_aware_vew_callers=control["width_aware_vew_callers"],
                  width_aware_vmatr_caller=CALLER)
    if args.output is None:
        print(json.dumps({key: report[key] for key in ("ffn_rows_used", "ffn_rows_allocated")}))
        return
    target = args.output.resolve()
    manifest = target.with_suffix(".manifest.json")
    if target.suffix != ".gguf" or target.exists() or manifest.exists():
        raise ValueError("Use a fresh .gguf output")
    if dot.sha(args.baseline) != BASE_SHA:
        raise ValueError("Unexpected VEW16 baseline bytes")
    if shutil.disk_usage(target.parent).free < args.baseline.stat().st_size + 2 * 2**30:
        raise OSError("Insufficient space for checkpoint and evidence")
    source = gguf.GGUFReader(args.baseline)
    src_tensors = {tensor.name: tensor for tensor in source.tensors}
    ffn = {"blk.64.ffn_gate.weight": (gate, F32), "blk.64.ffn_up.weight": (up, BF16),
           "blk.64.ffn_down.weight": (down, F16)}
    replacement = {}
    for name, (entries, kind) in ffn.items():
        tensor = src_tensors[name]
        if tensor.tensor_type != kind:
            raise ValueError(f"Changed storage geometry: {name}")
        replacement[name] = render(entries, tuple(map(int, tensor.shape[::-1])), kind), kind
    head = src_tensors[merge.COMBINED + merge.HEAD]
    if head.tensor_type != merge.Q4 or tuple(head.shape) != (dot.WIDTH, merge.GUARD + 1):
        raise ValueError("Unexpected trimmed action head geometry")
    control_scores = dict(scores)
    source_lane = S.ACTION + ids[CALLER]
    control_scores[ids["M_MUL_INIT"], source_lane] = control_scores.pop((ids["M_MUL_PICK"], source_lane))

    def encode(weights):
        data = np.zeros((merge.GUARD + 1, dot.WIDTH), dtype="<f4")
        for index, value in weights.items():
            data[index] = value
        return merge.encode_head(SimpleNamespace(data=data, tensor_type=F32))

    if not np.array_equal(encode(control_scores), head.data):
        raise ValueError("Unexpected action-head changes beyond the VMATR multiply route")
    replacement[head.name] = encode(scores), head.tensor_type
    try:
        tensors = ((t.name, *replacement.get(t.name, (t.data, t.tensor_type))) for t in source.tensors)
        merge.write_gguf(target, "qwen35", tensors, source)
        report.update(merge.verify(source, target, replacement), sha256=dot.sha(target), bytes=target.stat().st_size,
                      exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if report["unchanged_tensors"] != 865 or report["bytes"] != args.baseline.stat().st_size:
            raise ValueError("Changes exceeded the four circuit tensors or altered checkpoint size")
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except Exception:
        target.unlink(missing_ok=True)
        raise
    print(json.dumps({key: report[key] for key in ("ffn_rows_used", "sha256", "bytes")}))


if __name__ == "__main__":
    main()
