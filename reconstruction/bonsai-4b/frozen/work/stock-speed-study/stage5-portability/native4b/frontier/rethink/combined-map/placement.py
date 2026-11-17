"""Place the combined production sparse circuit, without exporting a checkpoint."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
import struct
import sys

import numpy as np
import check_flow
import transport

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nibble-census"))
import nibble_census as nibble

public, spill, full, compact, grid, live, ret, s = (
    nibble.public, nibble.spill, nibble.full, nibble.compact, nibble.spill.grid,
    nibble.live, nibble.ret, nibble.s,
)
from export_tensors import q4_head

WIDTH, HEAD, GROUPS = 2560, full.HEAD, full.GROUPS
GUARD, SCORE_SIGNS = 513, 9
DOT = full.dot
ROOT = full.STUDY.parents[1]


def production_source():
    old, (ids, _, gates, ups, downs, count), ports, gates_dot, _ = full.build()
    gates, ups, downs = (live.sparse_rows(gates, count), live.sparse_rows(ups, count),
                         live.sparse_rows(downs, count, transpose=True))
    original = (old, ids, *([None] * 12), gates, ups, downs)
    cut = count - len(compact.structured.dot_plane.compile_ff(False, True).g) - 4 * len(gates_dot)
    _, _, scratch, _, _, private = live.inventory(old, ids, ports, gates_dot, downs)
    _, _, _, phases, _, _ = live.summarize_rows(old, ids, gates, ups, downs, private, scratch)
    callers, setter_checks, _ = ret.prove_calls(old, ids, gates, ups, downs, phases, private)
    assert len(callers) == 54 and setter_checks
    c, pending = nibble.compact_returns(nibble.transplant(old, callers), ids, callers)
    new_ids, g, u, d, dot_start = spill.ordinary_compile(c, original, cut)
    assert ids == new_ids and len(ids) == GUARD
    d = [{lane: weight for lane, weight in row.items() if lane < s.DISPATCH} for row in d]
    report, details, deps = nibble.analyze(c, ids, g, u, d, dot_start, ports, gates_dot, callers, True, pending)
    assert report["fixed_emission_colors"] == 842 and report["active_actions"] == 506
    return report, details, deps, callers, ports, gates_dot, pending


def side_roles(c):
    # Keep the low plane's raw slot as the replacement signed nibble's
    # ingress. Take the four anchor columns from the released second plane.
    side = tuple(c.dot_payload_ingress)
    kept = tuple(i for group in range(GROUPS) for i in range(group * 64, group * 64 + 16))
    release = tuple(i for i in range(256) if i not in kept)
    anchor_keys = tuple(range(DOT.SIDE_PAYLOAD + 257, DOT.SIDE_PAYLOAD + 261))
    signed = {DOT.SIDE_PAYLOAD + i: side[i] for i in (*kept, 256)}
    anchors = dict(zip(anchor_keys, (side[i] for i in release[:4]), strict=True))
    return signed, anchors, [side[i] for i in release]


def allocate(details, signed, anchors, width):
    c, _, _, _, _, selector, _, _, _, _, _, _, _, colors, ingress, raw, pool, scratch, _, floor = details
    fixed_raw = raw - pool
    fixed_ingress = ingress - pool
    side = signed | anchors
    host_only = (ingress - raw) | set(side.values())
    assert not (ingress | raw) & set(c.dot_payload_ingress)
    controls = [*range(compact.CLASS, compact.CLASS + len(selector.class_masks)),
                *range(grid.GROUP, grid.GROUP + selector.groups), *range(grid.MEMBER, grid.MEMBER + 32),
                grid.REFERENCE]
    ncolors = len(set(colors.values()))
    roles = floor + 2 * ncolors - 257 + len(side)
    assert len(controls) == selector.input_roles
    if roles > width:
        raise ValueError(f"{roles} residual roles exceed width {width} by {roles-width}")

    # Input and score tiles can share a 128-lane tile when their coordinates
    # are disjoint. Host-only raw addresses can overlap muted projected roles.
    for score_start in reversed(range(0, width, 32)):
        score = set(range(score_start, score_start + 32))
        if score & (fixed_raw | host_only):
            continue
        for tile_start in reversed(range(0, width, 128)):
            control_slots = sorted(set(range(tile_start, tile_start + 128)) - fixed_raw - host_only - score)
            if len(control_slots) >= len(controls):
                break
        else:
            continue
        break
    else:
        raise ValueError("Unable to place the aligned PQ2 input and Q4 score tiles")
    action = dict(zip(controls, control_slots[:len(controls)], strict=True))
    free = set(range(width)) - fixed_raw - score - set(action.values())
    raw_colors = sorted(free - host_only)[:ncolors]
    assert len(raw_colors) == ncolors
    free = iter(sorted(free - set(raw_colors)))
    fixed_work = {lane: next(free) for lane in sorted(fixed_ingress)}
    work_colors = [next(free) for _ in range(ncolors)]
    scratch_map = {lane: next(free) for lane in sorted(scratch)}
    projected_side = {lane: next(free) for lane in side}
    # Raw anchors consume only released transport columns. Preserve the
    # distinction even if a future packing order moves a side output.
    assert set(anchors.values()).isdisjoint(projected_side.values())
    raw_map = {lane: raw_colors[color] for lane, color in colors.items()}
    output = fixed_raw | set(raw_colors) | score
    projected = set(fixed_work.values()) | set(work_colors) | set(scratch_map.values()) | set(projected_side.values())
    used = output | projected | set(action.values())
    assert len(used) == roles and not output & host_only and max(used | host_only) < width
    return dict(width=width, raw=raw_map, fixed_raw={lane: lane for lane in sorted(fixed_raw)},
                work=fixed_work | {lane: work_colors[color] for lane, color in colors.items()},
                side=projected_side, scratch=scratch_map, input=action, score_start=score_start,
                score_tile=sorted(score), input_tile_start=tile_start, head_feedback=sorted(output),
                unused=sorted(set(range(width)) - used), active_roles_with_score_padding=roles)


def signed_attention(details):
    """Remap one signed channel per W nibble and preserve all other paths."""
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, matrices, _ = details
    q, k, original_v, original_o = matrices
    v, o = dict(original_v), dict(original_o)
    signed_outputs = set()
    for group in range(GROUPS):
        for slot in range(64, 128):
            original_col = DOT.SIDE_PAYLOAD + 64 * group + slot - 64
            old_v = v.pop((HEAD * group + slot, original_col))
            assert old_v == 1
            if slot < 80:
                v[HEAD * group + slot, original_col] = 1
            for sub, sign in ((2, 1), (3, -1)):
                head = 4 * group + sub
                key = DOT.PLANE, head * HEAD + slot
                expected = sign * (1, 2, 4, -8)[(slot - 64) // 16] / 512
                assert o.pop(key) == expected
                signed_outputs.add(key)
                if slot < 80:
                    o[key] = sign / 64
    assert {key: value for key, value in original_o.items() if key not in signed_outputs} == {
        key: value for key, value in o.items() if key not in signed_outputs}
    assert all(v[key] == value for key, value in original_v.items() if key[1] < DOT.SIDE_PAYLOAD
               or key[1] == DOT.SIDE_PAYLOAD + 256)
    assert len(original_v) - len(v) == 192 and len(original_o) - len(o) == 384
    return q, k, v, o


def project(details, place, side_raw, matrices):
    _, _, _, _, groups, selector, *_ = details
    width = place["width"]
    work, raw = place["work"], place["fixed_raw"] | place["raw"]
    staged = {s.WORK + lane: col for lane, col in work.items()} | place["side"]
    eh = {(col, col): 1 for col in place["input"].values()}
    for lane, col in work.items():
        key = (col, width + raw.get(lane, lane))
        eh[key] = 1
    for lane, col in place["side"].items():
        eh[col, width + side_raw[lane]] = 1
    if len({i for i, _ in eh}) != len(eh) or len({j for _, j in eh}) != len(eh):
        raise ValueError("Static EH has colliding raw columns or projected rows")
    assert len(eh) == (len(set(work.values())) + len(place["side"]) + selector.input_roles)
    assert all(0 <= i < width and 0 <= j < 2 * width for i, j in eh)

    def qkv(matrix):
        by_row = defaultdict(dict)
        mapped = {}
        for (row, col), value in matrix.items():
            if s.ACTION <= col < s.DISPATCH:
                by_row[row][col - s.ACTION] = value
            else:
                key = row, staged[col]
                assert key not in mapped
                mapped[key] = value
        assert len(mapped) + sum(map(len, by_row.values())) == len(matrix)
        for row, values in by_row.items():
            for group, members in enumerate(groups):
                weights = {values.get(action, 0) for action in members}
                assert len(weights) == 1
                if weight := weights.pop():
                    key = row, place["input"][compact.CLASS + group]
                    assert key not in mapped
                    mapped[key] = 2 * weight
        return mapped

    q, k, v, o = matrices
    o_phys = {(place["scratch"].get(dest, raw.get(dest, dest)), col): value for (dest, col), value in o.items()}
    assert len(o_phys) == len(o)
    assert {r for r, _ in o_phys} <= set(place["scratch"].values()) | set(place["fixed_raw"].values())
    projected = dict(EH=eh, Q=qkv(q), K=qkv(k), V=qkv(v), O=o_phys)
    for name, limits in {"EH": (width, 2 * width), "Q": (2 * full.QUERY * HEAD, width),
                         "K": (GROUPS * HEAD, width), "V": (GROUPS * HEAD, width),
                         "O": (width, full.QUERY * HEAD)}.items():
        assert all(0 <= i < limits[0] and 0 <= j < limits[1] and float(np.float32(val)) == val
                   for (i, j), val in projected[name].items())
    return projected


def pack_action_tables(details, place):
    _, ids, active, codes, _, selector, *_ = details
    width = place["width"]
    tile_index = place["input_tile_start"] // 128
    zero_tile = b"\x00\x00" + bytes([0x55]) * 32
    embedding = sha256()
    score = {}
    for token in range(GUARD + 1):
        tiles = [zero_tile] * (width // 128)
        if token in codes:
            code = codes[token]
            assert ids[active[code]] == token
            tile = bytearray(struct.pack("<e", 0.5) + bytes([0x55]) * 32)
            slots = {place["input"][lane] % 128 for lane, value in selector.embedding(code) if value == 0.5}
            assert len(slots) == 4
            for slot in slots:
                tile[2 + slot // 4] ^= 3 << (2 * (slot % 4))
            decoded = compact.structured.decode_tiles(tile + tile)[:128]
            assert all(value == (0.5 if i in slots else 0) for i, value in enumerate(decoded))
            tiles[tile_index] = tile
            score.update({(token, place["score_start"] + bit): s.sign(code, bit) for bit in range(SCORE_SIGNS)})
        embedding.update(b"".join(tiles))
    packed, live_tiles = q4_head(score, (GUARD + 1, width))
    assert live_tiles == len(active)
    score_tile = place["score_start"] // 32
    values = np.zeros((GUARD + 1, 32), dtype=np.int16)
    for token in range(GUARD + 1):
        for col in range(width // 32):
            cells = packed[token, col]
            if col != score_tile or token not in codes:
                assert cells[:2].tobytes() == b"\x00\x00" and np.all(cells[2:] == 0x88)
                continue
            assert cells[:2].tobytes() == b"\x00\x34"
            values[token] = [((int(cells[2 + i % 16]) >> (4 if i >= 16 else 0)) & 15) - 8 for i in range(32)]
            assert all(weight == 4 * score.get((token, place["score_start"] + i), 0)
                       for i, weight in enumerate(values[token]))
    min_margin = 36
    for token in codes:
        logits_times_four = values @ (values[token] // 4)
        assert int(logits_times_four.argmax()) == token and logits_times_four[token] == SCORE_SIGNS * 4
        assert logits_times_four[GUARD] == 0
        winner = logits_times_four[token]
        logits_times_four[token] = -1
        min_margin = min(min_margin, int(winner - logits_times_four.max()))
    return dict(action_pq2_rows_sha256=embedding.hexdigest(),
                action_q4_rows_sha256=sha256(packed.tobytes()).hexdigest(),
                head_row_to_token="identity", guard=GUARD, guard_exact_zero=True,
                inactive_action_rows_zero=sorted(set(range(GUARD)) - codes.keys()),
                unit_sign_code_minimum_margin=min_margin / 4,
                active_rows=len(codes), one_pq2_tile_per_active_row=True)


def negative_controls(details, place, converted, signed, anchors):
    c = details[0]
    wide = {DOT.SIDE_PAYLOAD + i: col for i, col in enumerate(c.dot_payload_ingress)}
    try:
        allocate(details, wide, {}, WIDTH)
    except ValueError as exc:
        assert str(exc) == "2745 residual roles exceed width 2560 by 185"
    else:
        raise AssertionError("Full W bitplanes unexpectedly fit")
    collided = dict(anchors)
    collided[next(iter(collided))] = next(iter(signed.values()))
    try:
        project(details, place, signed | collided, converted)
    except ValueError as exc:
        assert str(exc) == "Static EH has colliding raw columns or projected rows"
    else:
        raise AssertionError("Duplicated anchor ingress was accepted")
    return dict(full_wide_wrom_roles=2745, wide_wrom_native_fit=False,
                aliased_retained_wrom_column_rejected=True)


def tensor_contract(place, projections):
    eps = transport.EPS
    reference = json.loads((full.PORTABILITY / "headers.json").read_text(encoding="utf-8"))
    header = reference["remote"]["Ternary-Bonsai-2-4B-PQ2_0.gguf"]
    meta = header["metadata"]
    assert (meta["qwen35.embedding_length"], meta["qwen35.attention.head_count"],
            meta["qwen35.attention.head_count_kv"], meta["qwen35.attention.key_length"],
            meta["qwen35.attention.value_length"], meta["qwen35.rope.dimension_count"],
            meta["qwen35.attention.layer_norm_rms_epsilon"]) == (WIDTH, full.QUERY, GROUPS, HEAD, HEAD, s.ROTARY, eps)
    scaled = float(np.float32(1 / sqrt(WIDTH)))
    hnorm = sorted(col - WIDTH for _, col in projections["EH"] if col >= WIDTH)
    qk_norm = list(range(s.ROTARY, s.ROTARY + 33*s.REPEATS + 1))
    full_width = "all 2560 coordinates"
    normals = {
        "nextn.enorm.weight": dict(coefficient=scaled, coordinates=full_width),
        "nextn.hnorm.weight": dict(coefficient=scaled, coordinates=hnorm),
        "attn_norm.weight": dict(coefficient=scaled, coordinates=full_width),
        "post_attention_norm.weight": dict(coefficient=scaled, coordinates=full_width),
        "attn_q_norm.weight": dict(coefficient=8, coordinates=qk_norm),
        "attn_k_norm.weight": dict(coefficient=8, coordinates=qk_norm),
        "nextn.shared_head_norm.weight": dict(coefficient=1, coordinates=place["head_feedback"]),
    }
    return dict(bonsai4b_header_sha256=header["header_sha256"],
                attention_geometry=dict(heads=full.QUERY, kv_groups=GROUPS, head_width=HEAD, rope_dims=s.ROTARY),
                rms_epsilon=eps, norms_f32=normals, eh_qkvo_type="F32", cache_kv_type="F16",
                ffn_type={"gate": "F32", "up": "BF16", "down": "F16"},
                packed_input_type="PQ2_0", trimmed_score_type="Q4_0", trimmed_ids_type="I32",
                trimmed_ids=list(range(GUARD + 1)),
                required_mtp_tensors=["nextn.embed_tokens.weight", "nextn.shared_head_norm.weight",
                                      "nextn.draft_head.weight", "nextn.draft_head_ids.weight"],
                required_absent_mtp_tensor="nextn.shared_head_head.weight",
                required_unset_runtime_variable="LLAMA_MTP_DRAFT_HEAD_OFF",
                projection_bytes_above_old_f16_eh_k_v_ledger=2 * (2*WIDTH*WIDTH + 2*GROUPS*HEAD*WIDTH))


def file_hashes():
    paths = [Path(__file__), Path(transport.__file__), Path(check_flow.__file__),
             Path(nibble.__file__), Path(public.__file__), Path(spill.__file__), Path(ret.__file__),
             Path(full.__file__), Path(full.vx.__file__), Path(s.__file__), Path(DOT.__file__),
             Path(full.vmatr.__file__), Path(live.__file__), Path(compact.__file__), Path(grid.__file__),
             Path(compact.structured.__file__), public.OBSERVER, public.PROBE,
             full.PORTABILITY / "headers.json", full.STUDY / "b12128-runtime-pin.json",
             HERE.parent / "nibble-census/nibble-census.json",
             HERE.parent / "signed-rom-stock/evidence-verified/receipt.json"]
    return {path.relative_to(ROOT).as_posix(): sha256(path.read_bytes()).hexdigest() for path in paths}


def main(output):
    liveness, details, deps, callers, ports, gates_dot, pending = production_source()
    c = details[0]
    signed, anchors, released = side_roles(c)
    place = allocate(details, signed, anchors, WIDTH)
    print(f"Physical map uses {place['active_roles_with_score_padding']} / {WIDTH} residual roles",
          file=sys.stderr, flush=True)
    converted = signed_attention(details)
    projected = project(details, place, signed | anchors, converted)
    controls = negative_controls(details, place, converted, signed, anchors)
    pack = pack_action_tables(details, place)
    fields, read_tokens, tile, range_guard = public.contract(details[1])
    abi = transport.launcher_abi(details, place, signed, anchors, fields, read_tokens, tile, range_guard, s, DOT)
    transport_proof = transport.seed_and_transport_checks(details, place, abi, projected, s, full, ports, gates_dot)
    transport_proof["pinned_file_seed_packing"] = transport.check_pinned_wrom_elf(ROOT / "guests/stories_vx_three.elf", abi)
    flow, phase_rows = check_flow.proof(details, deps, place, abi, callers, fields, read_tokens, tile, range_guard,
                                        (spill, public, live, ret, s))
    grouped = spill.regroup_rows(details, place)
    assert grouped["rows"] == 36352 and grouped["storage"]["shared_h512_rows"] == 36352
    baseline = json.loads((HERE.parent / "nibble-census/nibble-census.json").read_text(encoding="utf-8"))
    control = baseline["variants"]["radix16_compact_return"]["guarded_public"]
    prior_packed = nibble.verify_placement(details, spill.physical_witness(details))
    prior_recorded = control["full_wrom_packed_control"]
    assert all(prior_packed[k] == prior_recorded[k] for k in ("roles", "width", "pq2_tile_sha256", "q4_tile_sha256",
                                                           "sparse_projection_sha256", "sparse_projection_coefficients"))
    controls["reproduced_3072_wide_control"] = {key: prior_packed[key]
                                               for key in ("width", "roles", "sparse_projection_sha256")}
    print(f"Placed {grouped['rows']} sparse FFN rows", file=sys.stderr, flush=True)
    report = dict(width=WIDTH, projected_seed_pins=len(place["side"]), fixed_emission_colors=liveness["fixed_emission_colors"],
                  role_total=place["active_roles_with_score_padding"], free_lanes=place["unused"],
                  fixed_floor=place["active_roles_with_score_padding"] - 2*liveness["fixed_emission_colors"],
                  released_wrom_ingress=released, reused_anchor_ingress=list(anchors.values()),
                  slots=place, immutable_seed_raw_signed=signed, immutable_seed_raw_anchors=anchors,
                  packed_actions=pack, grouped_ffn=grouped, abi=abi, transport=transport_proof, flow=flow,
                  arithmetic=nibble.arithmetic_proof(), controls=controls,
                  tensor_contract=tensor_contract(place, projected), source_sha256=file_hashes(),
                  projection_sha256={kind: nibble.digest_matrix(matrix) for kind, matrix in projected.items()},
                  projection_nonzero={kind: len(matrix) for kind, matrix in projected.items()})
    if output:
        report["phase_masks_sha256"] = check_flow.write_phase_masks(output.with_name("phase-masks.json"), phase_rows)
        projection_path = output.with_name("projections.json")
        projections_text = json.dumps({kind: sorted((i, j, value) for (i, j), value in matrix.items())
                                       for kind, matrix in projected.items()}, separators=(",", ":")) + "\n"
        projection_path.write_text(projections_text, encoding="utf-8", newline="\n")
        report["projection_artifact_sha256"] = sha256(projections_text.encode("utf-8")).hexdigest()
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    result = main(options.output)
    if not options.output:
        print(json.dumps({key: value for key, value in result.items() if key != "slots"}, indent=2))
