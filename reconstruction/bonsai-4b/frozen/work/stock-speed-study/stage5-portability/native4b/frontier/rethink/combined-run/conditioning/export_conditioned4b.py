"""Patch the VEXP raw readback operand gates in the pinned 4B checkpoint."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
EXPORT = HERE.parent / "export"
sys.path.insert(0, str(EXPORT))
import export_combined4b as base

placement = base.placement
s = placement.s
SOURCE = EXPORT / "Ternary-Bonsai-2-4B-PQ2_0-QwenPU.gguf"
SOURCE_SHA256 = "31c46924342d983a60ca6654507720e29a41aa4017b607b98820b06c53bf3a29"
GATE = base.PREFIX + "ffn_gate.weight"
SETTER = "VX_E_VM_READ"
GAIN = 8192


def coefficient_plan(report):
    compile_source = placement.production_source
    setter_inventory = {}

    def checked_source():
        result = compile_source()
        c, ids, *_ = result[1]
        callers = result[3]
        operands = {*s.bits(s.A), *s.bits(s.B)}
        readback = set(s.bits(s.READBACK, 57))
        direct = {}
        for phase in callers:
            raw = {
                dest: expr for dest, expr in c.effects[phase].items()
                if dest in operands and any(col in readback for _, terms in expr for col, _ in terms)
            }
            if raw:
                direct[phase] = raw
        expected = {s.B + i: s.buffer(s.READBACK + i, -1) for i in range(32)}
        if direct != {SETTER: expected} or ids[SETTER] != 477:
            raise ValueError("Direct readback operand setters changed")
        if c.routes[SETTER] != {"ADD_INIT": None}:
            raise ValueError("VEXP adder route changed")
        setter_inventory.update(all_callers=len(callers), direct_readback_callers=[SETTER],
                                action_id=ids[SETTER], source_slots=list(s.bits(s.READBACK, 32)))
        return result

    with patch.object(placement, "production_source", checked_source):
        rows, codes, selector, place, _ = base.compile_rows(report)
    group, member = divmod(codes[setter_inventory["action_id"]], selector.columns)
    groups = [place["input"][placement.grid.GROUP + g] for g in range(selector.groups)]
    up = ((place["input"][placement.grid.MEMBER + member], 32),)
    raw = place["fixed_raw"] | place["raw"]
    mapped = {raw[s.READBACK + i]: raw[s.B + i] for i in range(32)}
    matched = defaultdict(set)
    edits = []
    for idx, (gate, actual_up, down) in enumerate(rows):
        coeffs = dict(gate)
        selected_group = groups[group]
        if actual_up != up or abs(coeffs.get(selected_group, 0)) != 8:
            continue
        readback = coeffs.keys() & mapped.keys()
        if not readback:
            continue
        if len(readback) != 1 or coeffs.keys() != set(groups) | readback:
            raise ValueError(f"Row {idx} has a non-unary readback gate")
        lane, = readback
        bias = coeffs[selected_group] // 2
        if bias in matched[lane] or down != ((mapped[lane], -bias / 4),):
            raise ValueError(f"Row {idx} has an unexpected VEXP down projection")
        matched[lane].add(bias)
        old_off = -2 * placement.grid.penalty_for(((lane, 512),), bias, 4)
        new_off = -2 * placement.grid.penalty_for(((lane, GAIN),), bias, 4)
        expected = {g: old_off for g in groups if g != selected_group} | {lane: 512}
        if any(coeffs[col] != value for col, value in expected.items()):
            raise ValueError(f"Row {idx} is shared or has unexpected weights")
        edits.append((idx, lane, 512, GAIN))
        edits.extend((idx, g, old_off, new_off) for g in groups if g != selected_group)
    if dict(matched) != {lane: {-4, 4} for lane in mapped}:
        raise ValueError("Expected both signed SiLU branches for all 32 VEXP bits")
    setter_inventory.update(paired_rows=len(matched) * 2, bit_output_slots=list(mapped.values()),
                            selector_group=group, selector_member=member, scheduled_actions_changed=0,
                            old_source_gain=512, new_source_gain=GAIN)
    return sorted(edits), setter_inventory


def copy_and_patch(source, output, edits, gate_start, gate_end):
    source.seek(0)
    source_sha, output_sha, old_gate_sha, new_gate_sha = (sha256() for _ in range(4))
    pending = iter(edits)
    change = next(pending, None)
    position = 0
    with output.open("xb") as dst:
        while chunk := source.read(16 * 1024**2):
            end = position + len(chunk)
            source_sha.update(chunk)
            overlap = slice(max(gate_start - position, 0), min(gate_end, end) - position)
            overlap_gate = position < gate_end and gate_start < end
            if overlap_gate:
                old_gate_sha.update(chunk[overlap])
            if change and change[0] < end:
                chunk = bytearray(chunk)
                while change and change[0] < end:
                    offset, old, new = change
                    at = offset - position
                    if at < 0 or at + 4 > len(chunk) or struct.unpack_from("<f", chunk, at)[0] != old:
                        raise ValueError(f"Unexpected archived F32 gate coefficient at GGUF offset {offset}")
                    struct.pack_into("<f", chunk, at, new)
                    change = next(pending, None)
            if overlap_gate:
                new_gate_sha.update(chunk[overlap])
            dst.write(chunk)
            output_sha.update(chunk)
            position = end
    if change or source_sha.hexdigest() != SOURCE_SHA256:
        raise ValueError("Archived combined checkpoint differs from its pinned SHA-256 or gate extent")
    return output_sha.hexdigest(), old_gate_sha.hexdigest(), new_gate_sha.hexdigest(), position


def export(target):
    receipt = target.with_suffix(".manifest.json")
    exemplar_path = SOURCE.with_suffix(".manifest.json")
    exemplar = json.loads(exemplar_path.read_bytes())
    if target.exists() or receipt.exists():
        raise FileExistsError(f"Expected a fresh output and manifest for {target.name}")
    if shutil.disk_usage(HERE).free < exemplar["bytes"] + 1024**2:
        raise OSError("Insufficient room for the conditioned combined model")
    report, _ = base.frozen_inputs()
    edits, scope = coefficient_plan(report)
    with SOURCE.open("rb") as source, tempfile.TemporaryDirectory(dir=HERE, prefix="gate-export-") as temp:
        header = base.pad.Header(source).parse()
        tensor = header["tensors"][GATE]
        if (header["header_sha256"] != exemplar["header_sha256"]
                or tensor["dims"] != [base.WIDTH, base.FFN] or tensor["type_id"] != 0):
            raise ValueError("Unexpected archived GGUF header or F32 gate geometry")
        start = header["tensor_data_offset"] + tensor["offset"]
        end = start + base.pad.tensor_bytes(tensor)
        offsets = [(start + (row * base.WIDTH + col) * 4, old, new) for row, col, old, new in edits]
        if not all(start <= offset < end for offset, _, _ in offsets):
            raise ValueError("FFN gate edit escaped the archived gate tensor")
        staged = Path(temp) / target.name
        print(f"Writing {len(edits)} F32 edits into a fresh GGUF stream", flush=True)
        output_sha, old_gate_sha, gate_sha, size = copy_and_patch(source, staged, offsets, start, end)
        if size != exemplar["bytes"] or old_gate_sha != exemplar["circuit_tensors"][GATE]["sha256"]:
            raise ValueError("Archived gate byte extent or digest differs from the combined export manifest")
        result = {
            "output": str(target),
            "sha256": output_sha,
            "bytes": size,
            "source": str(SOURCE),
            "source_sha256": SOURCE_SHA256,
            "source_manifest_sha256": sha256(exemplar_path.read_bytes()).hexdigest(),
            "placement_sha256": base.MAP_SHA,
            "launcher_abi_header_sha256": sha256((HERE.parent / "launcher/native4b_abi.hpp").read_bytes()).hexdigest(),
            "header_sha256": header["header_sha256"],
            "only_changed_tensor": GATE,
            "unchanged_tensor_count": len(header["tensors"]) - 1,
            "language_tensor_extents_byte_equal": exemplar["language_tensors_byte_equal"],
            "gate_tensor": tensor | {"sha256": gate_sha, "source_sha256": old_gate_sha},
            "changed_gate_coefficients": len(edits),
            "gate_edits_row_col_old_new": edits,
            "conditioning": scope,
            "export_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "reproduction_argv": [sys.executable, "-B", *sys.argv],
        }
        staged_receipt = Path(temp) / receipt.name
        staged_receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
        os.link(staged, target)
        try:
            os.link(staged_receipt, receipt)
        except OSError:
            target.unlink()
            raise
    print(json.dumps({key: result[key] for key in ("output", "sha256", "bytes", "changed_gate_coefficients")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="Ternary-Bonsai-2-4B-PQ2_0-QwenPU-conditioned.gguf")
    options = parser.parse_args()
    if Path(options.output).name != options.output or not options.output.endswith(".gguf"):
        parser.error("--output must be a local GGUF filename")
    export(HERE / options.output)
