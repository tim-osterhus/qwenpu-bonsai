"""Replace the VEW multiplier entry and normalize its exact signed product."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys

import gguf
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "stage4-dot64"), str(HERE.parent / "stage4-dot64-merge")]
import build_dot64 as dot
import merge_dot64 as merge

S = dot.s
M = dot.m
INIT, NORMALIZE = "VEW_MUL16_INIT", "VEW_MUL16_NORMALIZE"
REUSED = {"DOT64_UNUSED_2": INIT, "DOT64_UNUSED_3": NORMALIZE}
CALLERS = ("VX_W_P_MUL", "VX_W_G_MUL", "VX_W_RF_ODD", "VX_W_RF_EVEN", "VX_W_RS_ODD", "VX_W_RS_EVEN")
DOWN_SHA = "142ecc7d0991186165da626d5b7ec7365dd1d71b515584022f9dcd965da99859"
F32, BF16, F16 = (gguf.GGMLQuantizationType[name] for name in ("F32", "BF16", "F16"))


def narrow_vew(c):
    if any(c.routes[phase] != {"SET_ILLEGAL": None} for phase in REUSED):
        raise ValueError("Reused phase is already reachable")
    if any(phase in routes for routes in c.routes.values() for phase in REUSED):
        raise ValueError("Reused phase already has a predecessor")
    c.phases = dict.fromkeys(REUSED.get(phase, phase) for phase in c.phases)
    for phase in REUSED:
        del c.routes[phase]

    # Moving the signed RHS left 16 preserves the original sign correction.
    # Start the unsigned 32-bit shift/add loop at iteration 16 with the original
    # RHS bits in its low word. The resulting full product is shifted left 16.
    low_zero = S.constant(-1)
    for phase in CALLERS:
        if c.routes[phase] != {"M_MUL_INIT": None}:
            raise ValueError(f"Unexpected multiplier entry: {phase}")
        effects = c.effects[phase]
        rhs = [effects[S.RS2 + bit] for bit in range(32)]
        if phase == "VX_W_P_MUL":
            if any(value != S.buffer(S.WORK + S.ZERO) for value in rhs[15:]):
                raise ValueError(f"Expected unsigned 15-bit RHS: {phase}")
        elif any(value != rhs[15] for value in rhs[16:]):
            raise ValueError(f"Expected signed 16-bit RHS: {phase}")
        for bit in range(16):
            effects[S.RS2 + bit] = low_zero
            effects[S.RS2 + 16 + bit] = rhs[bit]
        del c.routes[phase]["M_MUL_INIT"]
        c.route(phase, INIT)
    c.move(INIT, M.M_LOW, (*S.bits(S.WORK + S.RS2 + 16, 16), *((S.WORK + S.ZERO,) * 16)))
    c.fix(INIT, M.M_HIGH, 0)
    c.fix(INIT, M.M_COUNT, 16, 5)
    c.route(INIT, "M_MUL_PICK")

    # The shifted 48-bit signed product fits in 64 bits. Both existing signed
    # correction paths join here. An arithmetic right shift restores its words.
    if c.routes["VX_OP_MUL_OUT"] != {"VX_OP_MUL_RETURN": None}:
        raise ValueError("Unexpected product continuation")
    c.route("VX_OP_MUL_OUT", (NORMALIZE, c.op_flag["VEW"]))
    c.move(NORMALIZE, M.M_LOW, (*S.bits(S.WORK + M.M_LOW + 16, 16), *S.bits(S.WORK + M.M_HIGH, 16)))
    c.move(NORMALIZE, M.M_HIGH, (*S.bits(S.WORK + M.M_HIGH + 16, 16), *((S.WORK + M.M_HIGH + 31,) * 16)))
    c.route(NORMALIZE, "VX_OP_MUL_RETURN")


def render(entries, shape, kind):
    data = np.zeros(shape, dtype="<f4")
    for index, value in entries:
        data[index] = value
    if kind == F16:
        result = data.astype("<f2")
        if not np.array_equal(result.astype("<f4").view("<u4"), data.view("<u4")):
            raise ValueError("FFN down coefficient does not round-trip through FP16")
        return result
    if kind == BF16:
        raw = data.view("<u4")
        if np.any(raw & 65535):
            raise ValueError("Inexact BF16 FFN up coefficient")
        return (raw >> 16).astype("<u2").view(np.uint8)
    if kind != F32:
        raise ValueError(f"Unexpected dense tensor type: {kind}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    manifest = json.loads(args.baseline.with_suffix(".manifest.json").read_text())
    if manifest["sha256"] != DOWN_SHA:
        raise ValueError("Unexpected down-only baseline manifest")
    _, compiled, _, _ = dot.circuit(narrow_vew)
    ids, scores, gate, up, down, rows = compiled
    if ids != {REUSED.get(name, name): value for name, value in manifest["tokens"].items()}:
        raise ValueError("Action token IDs changed")
    if rows > 21248:
        raise ValueError("VEW candidate exceeds the existing FFN width")
    report = dict(source_sha256=DOWN_SHA, ffn_rows_used=rows, ffn_rows_allocated=21248,
                  tokens=ids, dot_actions=manifest["dot_actions"], width_aware_vew_callers=CALLERS)
    if args.output is None:
        print(json.dumps({key: report[key] for key in ("ffn_rows_used", "ffn_rows_allocated")}))
        return
    target = args.output.resolve()
    evidence = target.with_suffix(".manifest.json")
    if target.suffix != ".gguf" or target.exists() or evidence.exists():
        raise ValueError("Use a fresh output path")
    if dot.sha(args.baseline) != DOWN_SHA:
        raise ValueError("Unexpected down-only checkpoint bytes")
    if shutil.disk_usage(target.parent).free < args.baseline.stat().st_size + 2 * 2**30:
        raise OSError("Insufficient space for trial checkpoint and evidence")
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
        raise ValueError("Expected packed trimmed action head")
    data = np.zeros((merge.GUARD + 1, dot.WIDTH), dtype="<f4")
    for index, value in scores.items():
        data[index] = value
    replacement[head.name] = merge.encode_head(SimpleNamespace(data=data, tensor_type=merge.F32)), head.tensor_type
    norm = src_tensors["blk.64.nextn.shared_head_norm.weight"]
    if norm.tensor_type != F32 or norm.data.shape != (dot.WIDTH,):
        raise ValueError("Unexpected draft head norm geometry")
    norm_data = norm.data.copy()
    added_lanes = sorted({lane for _, lane in scores if norm_data[lane] == 0})
    if len(added_lanes) != 1 or added_lanes[0] != max(lane for _, lane in scores):
        raise ValueError(f"Unexpected new dispatch norm lanes: {added_lanes}")
    norm_data[added_lanes] = 1
    replacement[norm.name] = norm_data, F32
    report["enabled_dispatch_norm_lanes"] = added_lanes
    try:
        tensors = ((t.name, *replacement.get(t.name, (t.data, t.tensor_type))) for t in source.tensors)
        merge.write_gguf(target, "qwen35", tensors, source)
        report.update(merge.verify(source, target, replacement), sha256=dot.sha(target),
                      unchanged_checkpoint_size=target.stat().st_size == args.baseline.stat().st_size,
                      bytes=target.stat().st_size, exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if report["unchanged_tensors"] != 864 or not report["unchanged_checkpoint_size"]:
            raise ValueError("Changes exceeded the five expected tensors")
        evidence.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except Exception:
        # A failed owned output is unusable. The pinned baseline is separate.
        target.unlink(missing_ok=True)
        raise
    print(json.dumps({key: report[key] for key in ("ffn_rows_used", "sha256", "bytes")}))


if __name__ == "__main__":
    main()
