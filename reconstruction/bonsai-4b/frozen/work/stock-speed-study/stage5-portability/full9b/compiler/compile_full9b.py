"""Compile the full 513-action VMATR8/VEW16 circuit for the 4096-wide 9B block."""

from collections import Counter, defaultdict
from pathlib import Path
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PORTABILITY = HERE.parents[1]
STUDY = PORTABILITY.parent
sys.path.insert(0, str(STUDY / "stage4-vmatr8"))
import build_vmatr8 as vmatr

dot, s, vx = vmatr.dot, vmatr.S, vmatr.dot.vx
WIDTH, FF, QUERY, GROUPS, HEAD = 4096, 20480, 16, 4, 256
SECOND = "DOT64_UNUSED_4"
FIRST = tuple(group * 16 + lane for group in range(4) for lane in range(8))
LATE = tuple(lane + 8 for lane in FIRST)
ORDINARY = (s.FETCH, "READ_RS1", "READ_RS2", "READ_A7", "READ_A0", "READ_LW", "STORE_LOOKUP", "REG_LOOKUP")


def transport(c, audit):
    vmatr.circuit_transform(c)
    if c.routes[SECOND] != {"SET_ILLEGAL": None} or any(SECOND in rs for rs in c.routes.values()):
        raise ValueError("The second VA token is already in use")
    c.routes[SECOND].clear()
    fields = c.vx.fields
    gates, marker = fields["tile_activations"], fields["tile_weights"][:8]
    for bit, phase in enumerate(dot.VA):
        if c.routes[phase] != {(dot.CN if bit == 7 else dot.CP): None}:
            raise ValueError("Unexpected first VA routing")
        c.routes[phase].clear()
        c.route(phase, SECOND)
        for lane in LATE:
            c.effects[phase][gates[lane]] = ()
        for index in range(33):
            # VA chunks are 64-byte aligned. Echo byte offset 8 (+2 words),
            # preserving the typed address and the read marker.
            c.put(phase, s.READ_ADDR + index, s.buffer(s.WORK + s.READ_ADDR + index, -1 if index == 3 else 1))
        c.put(phase, s.READ_MARK, s.buffer(s.WORK + s.READ_MARK))
    for first, late in zip(FIRST, LATE, strict=True):
        c.put(SECOND, gates[late], s.buffer(dot.VA_OUT + first))
    c.branch(SECOND, marker[7], dot.CN, dot.CP)

    # All scalar-visible feedback and observed VX values are kept even if
    # none of the circuit's own expressions reads an observed coordinate.
    public = set(range(s.OLD_HANDLE, s.BUS_WIDTH + 2))
    public.update((*s.bits(s.BUS_ADDR), *s.bits(s.BUS_WORD), *s.bits(s.EXIT_CODE), s.PUTCHAR_FLAG))
    used = public | set(gates) | set(marker)
    for key in ("vmx", "vsum", "varg", "vbest", "vbest_valid", "wrom_word", "tile_products", "tile_generate"):
        if key in ("tile_products", "tile_generate"):
            used.update(fields[key][:(29 if key == "tile_products" else 18)])
        else:
            used.update(fields[key])
    for effects in c.effects.values():
        used.update(lane - s.WORK for expr in effects.values() for _, terms in expr for lane, _ in terms)
    used.update(abs(flag) for routes in c.routes.values() for flag in routes.values() if flag is not None)
    before = c.persistent
    c.persistent = tuple(lane for lane in before if lane in used)
    dead = set(before) - set(c.persistent)
    for phase, effects in c.effects.items():
        for lane in effects.keys() & dead:
            del effects[lane]
    audit["dead_persistent_lanes"] = sorted(dead)


def build():
    audit = {}
    c, compiled, ports, gates = dot.circuit(lambda c: transport(c, audit))
    ids, scores, gate, up, down, rows = compiled
    baseline = json.loads((STUDY / "stage4-vmatr8/vmatr8.manifest.json").read_text())
    if ids != baseline["tokens"] or len(ids) != 513 or rows > FF:
        raise ValueError(f"Action IDs or FFN capacity changed: {len(ids)} actions, {rows} rows")
    audit.update(ffn_rows_used=rows, ffn_rows_allocated=FF, actions=len(ids), persistent_lanes=len(c.persistent),
                 va1_ids=[ids[p] for p in dot.VA], va2_id=ids[SECOND])
    return c, compiled, ports, gates, audit


def attention(c, ids, ports, gates):
    """Sparse coefficients in the staged logical coordinates."""
    q, k, v, o = {}, {}, {}, {}
    marker = c.vx.fields["tile_weights"][:8]
    normal = set(ORDINARY) | c.vx.reads
    sentinel = s.ROTARY + 33 * s.REPEATS
    for group in range(GROUPS):
        base = HEAD * group
        for bit in range(33):
            for rep in range(s.REPEATS):
                k[base + s.ROTARY + bit * s.REPEATS + rep, s.WORK + s.WRITE_ADDR + bit] = 1
        k[base + sentinel, s.ACTION + ids[s.SEED_MISS]] = 1
        for bit in range(32):
            v[base + bit, s.WORK + s.WRITE_WORD + bit] = 1
            v[base + bit, s.ACTION + ids[s.SEED_MISS]] = -1 / 16
        for lane in range(64):
            v[base + 64 + lane, dot.SIDE_PAYLOAD + group * 64 + lane] = 1
        for sub in range(4):
            head = 4 * group + sub
            start = 2 * HEAD * head
            if sub < 2:
                word = group * 4 + sub
                for bit in range(33):
                    flip = 2 <= bit < 6 and bool(word >> (bit - 2) & 1)
                    for rep in range(s.REPEATS):
                        q[start + s.ROTARY + bit * s.REPEATS + rep, s.WORK + s.READ_ADDR + bit] = -1 if flip else 1
                q[start + sentinel, s.WORK + s.READ_MARK] = 11.125
                for slot in range(32):
                    q[start + HEAD + slot, s.WORK + marker[slot % 8]] = 65536
                    for phase, token in ids.items():
                        if phase != SECOND:
                            # Strong ordinary head-zero gates dominate the retained
                            # marker that is consulted exclusively by the VA2 token.
                            active = phase == dot.VA[slot % 8] or (head == 0 and phase in normal)
                            q[start + HEAD + slot, s.ACTION + token] = 65536 if active else -65536
                    o[dot.VA_OUT + 16 * group + 4 * sub + slot // 8, head * HEAD + slot] = 1
                continue
            for bit in range(33):
                if bit < 2:
                    source = s.WORK + s.ZERO
                elif bit < 18:
                    source = s.WORK + c.vx.fields["wrom_word"][bit - 2]
                else:
                    source = s.WORK + (s.ONE if dot.SIDE_BASE >> bit & 1 else s.ZERO)
                for rep in range(s.REPEATS):
                    q[start + s.ROTARY + bit * s.REPEATS + rep, source] = 1
            allowed = (dot.CP, dot.FP) if sub == 2 else (dot.CN, dot.FN)
            for slot in range(64, 128):
                row = start + HEAD + slot
                q[row, s.WORK + gates[group * 16 + (slot - 64) % 16]] = 16777216
                for phase, token in ids.items():
                    if phase not in allowed:
                        q[row, s.ACTION + token] = -100000000
                o[dot.PLANE, head * HEAD + slot] = (1 if sub == 2 else -1) * (1, 2, 4, -8)[(slot - 64) // 16] / 512

    for bit in range(32, 57):
        for phase, token in ids.items():
            q[HEAD + bit, s.ACTION + token] = 128 if phase in normal else -128
        v[bit, s.ACTION + ids[s.SEED_MISS]] = -1 / 16
    for bit in range(57):
        o[s.READBACK + bit, bit] = 1
    for bit in range(24):
        v[32 + bit, s.WORK + s.HANDLE_IN + bit] = 1
    for phase in (s.SEED_WORD, s.STORE, s.REGISTER):
        v[56, s.ACTION + ids[phase]] = 1 / 16

    # Group zero's positive W head supplies reference/correction on either sign.
    rh = 2
    start = 2 * HEAD * rh
    for slot in range(128, 136):
        v[slot, dot.SIDE_PAYLOAD + 256] = 1
        for phase, token in ids.items():
            q[start + HEAD + slot, s.ACTION + token] = (65536 if phase in (dot.CN, dot.CP, dot.FN, dot.FP)
                                                      and (slot != 135 or phase in (dot.FN, dot.FP)) else -65536)
    o[dot.REF, HEAD * rh + 128] = o[dot.REF2, HEAD * rh + 128] = 1
    for bit, lane in enumerate(ports["Q"]):
        slot = 129 + bit
        row = start + HEAD + slot
        for token in ids.values():
            q.pop((row, s.ACTION + token), None)
        q[row, s.WORK + lane] = 16777216
        for phase, token in ids.items():
            if phase not in (dot.FN, dot.FP):
                q[row, s.ACTION + token] = -100000000
        o[dot.PLANE, HEAD * rh + slot] = -(32 << bit) / 512
    o[dot.PLANE, HEAD * rh + 135] = 1
    return q, k, v, o


def coalesce_dispatch(c, ids, scores):
    """Color phase expressions only when every eligible score agrees."""
    eligible, phase_of, expression_of = {}, {}, {}
    cursor = s.DISPATCH
    for phase, routes in c.routes.items():
        eligible[phase] = {ids[target] for target in routes}
        seen = set()
        for flag in routes.values():
            if flag is None:
                continue
            expr = c.effects[phase].get(abs(flag), s.buffer(s.WORK + abs(flag)))
            if expr not in seen:
                seen.add(expr)
                phase_of[cursor] = phase
                expression_of[cursor] = expr
                cursor += 1
    wants = defaultdict(dict)
    for (target, col), value in scores.items():
        if col >= s.DISPATCH:
            wants[col][target] = value
    if phase_of.keys() != wants.keys():
        raise ValueError("Dispatch allocation no longer matches pinned scalar compiler")
    conflicts = {col: set() for col in phase_of}
    for left in phase_of:
        for right in range(left + 1, cursor):
            lp, rp = phase_of[left], phase_of[right]
            if lp == rp or any(wants[left].get(t, 0) != wants[right].get(t, 0) for t in eligible[lp] & eligible[rp]):
                conflicts[left].add(right)
                conflicts[right].add(left)
    colors = {}
    adjacent = defaultdict(set)
    while len(colors) < len(conflicts):
        col = max((k for k in conflicts if k not in colors),
                  key=lambda k: (len(adjacent[k]), len(conflicts[k]), -k))
        color = next(i for i in range(len(conflicts)) if i not in adjacent[col])
        colors[col] = color
        for conflict in conflicts[col]:
            adjacent[conflict].add(color)
    merged = defaultdict(dict)
    for col, color in colors.items():
        for target, value in wants[col].items():
            if target in merged[color] and merged[color][target] != value:
                raise ValueError("Conflicting coalesced dispatch coefficient")
            merged[color][target] = value
    max_ineligible = 0.0
    cols_by_phase = defaultdict(list)
    for column, phase in phase_of.items():
        cols_by_phase[phase].append(column)
        for target in eligible[phase]:
            if merged[colors[column]].get(target, 0) != wants[column].get(target, 0):
                raise ValueError("Coalescing changed an eligible dispatch score")
    for phase, columns in cols_by_phase.items():
        if len(set(colors[col] for col in columns)) != len(columns):
            raise ValueError("A phase simultaneously writes one physical dispatch lane twice")
        for target in set(ids.values()) - eligible[phase]:
            bound = sum(abs(merged[colors[col]].get(target, 0)) * sum(abs(f) for f, _ in expression_of[col])
                        for col in columns)
            max_ineligible = max(max_ineligible, bound)
    if max_ineligible >= 1:
        raise ValueError("Ineligible dispatch coefficient bound exceeds budget")
    return colors, merged, max_ineligible


def remap(c, compiled, ports, gates, report):
    ids, scores, gate, up, down, rows = compiled
    q, k, v, o = attention(c, ids, ports, gates)
    colors, merged, ineligible = coalesce_dispatch(c, ids, scores)
    ext = (*s.bits(s.WRITE_ADDR, 33), *s.bits(s.WRITE_WORD), *s.bits(s.HANDLE_IN, s.HANDLE_BITS),
           *s.bits(s.ENTRY), *s.bits(s.READ_ADDR, 33), s.READ_MARK)
    ingress = set(ext) | set(c.persistent) | set(c.dot_ephemeral)
    raw = ({i for (i, _), value in down if value and i < s.DISPATCH}
           | set(s.bits(s.READBACK, 57)) | set(c.persistent) | set(c.dot_ephemeral)
           | set(s.bits(s.READ_ADDR, 33)) | {s.READ_MARK}
           | set(s.bits(s.WRITE_ADDR, 33)) | set(s.bits(s.WRITE_WORD)))
    seed_raw = set(c.dot_payload_ingress)
    if raw & seed_raw or max(raw) >= s.WORK or len(seed_raw) != 257:
        raise ValueError("Seed or projected working coordinates collide with raw outputs")
    align = lambda n: (n + 31) // 32 * 32
    start_action = align(max(seed_raw | raw) + 1)
    action = {s.ACTION + i: start_action + i for i in range(len(ids))}
    start_dispatch = align(max(action.values()) + 1)
    dispatch = {col: start_dispatch + color for col, color in colors.items()}
    side_start = start_dispatch + len(merged)
    side = {dot.SIDE_PAYLOAD + i: side_start + i for i in range(len(seed_raw))}
    reserved = raw | set(action.values()) | set(dispatch.values()) | set(side.values())
    scratch = {r for (r, _), coefficient in o.items() if coefficient and r not in raw}
    logical_work = sorted(s.WORK + i for i in ingress)
    free = [i for i in range(WIDTH) if i not in reserved]
    if len(logical_work) + len(scratch) > len(free):
        raise ValueError(f"4096 lanes cannot fit {len(logical_work) + len(scratch)} working roles in {len(free)}")
    work = dict(zip(logical_work, free))
    temp = dict(zip(sorted(scratch), free[len(logical_work):]))
    projection = work | action | side
    if len(projection) != len(work) + len(action) + len(side):
        raise ValueError("Staged projection coordinates collide")
    source = projection | temp | {i: i for i in raw}
    if len(source) != len(projection) + len(temp) + len(raw):
        raise ValueError("Staged nonlinear inputs collide")
    if set(work.values()) & (set(temp.values()) | reserved):
        raise ValueError("Physical working coordinates collide")

    def cols(entries, mapping):
        return {(i, mapping[j]): value for (i, j), value in entries.items() if value}

    out = {i: i for i in raw} | dispatch
    down_source = dict(down)
    down_remapped = {(out[i], j): value for (i, j), value in down_source.items()}
    if len(down_remapped) != len(down_source):
        raise ValueError("A fused FFN row writes two coalesced output lanes")
    norm = {(i,): 1 / np.sqrt(WIDTH) for i in range(WIDTH)}
    head_out = raw | set(action.values()) | set(dispatch.values())
    if (head_out & seed_raw or head_out & (set(work.values()) | set(temp.values()) | set(side.values())) or
            set(dispatch.values()) & set(side.values()) or max(reserved | set(work.values()) | set(temp.values())) >= WIDTH):
        raise ValueError("A projected or seed-only coordinate can leak into feedback")
    input_projection = {(v, v): 1 for v in action.values()}
    input_projection.update({(work[s.WORK + i], WIDTH + i): 1 for i in ingress})
    input_projection.update({(side[dot.SIDE_PAYLOAD + i], WIDTH + raw_i): 1
                             for i, raw_i in enumerate(c.dot_payload_ingress)})
    score_head = {(row, action[col]): value for (row, col), value in scores.items() if col < s.DISPATCH}
    score_head.update({(row, start_dispatch + color): coefficient
                       for color, values in merged.items() for row, coefficient in values.items()})
    attn_norm = {(i,): 8 for i in range(s.ROTARY, s.ROTARY + 33 * s.REPEATS + 1)}
    tensors = {
        "nextn.draft_head.weight": score_head,
        "nextn.enorm.weight": norm,
        "nextn.hnorm.weight": {(i,): 1 / np.sqrt(WIDTH) for i in ingress | seed_raw},
        "nextn.eh_proj.weight": input_projection,
        "nextn.shared_head_norm.weight": {(i,): 1 for i in head_out},
        "attn_norm.weight": norm,
        "attn_q.weight": cols(q, projection),
        "attn_k.weight": cols(k, projection),
        "attn_v.weight": cols(v, projection),
        "attn_output.weight": {(source[r], j): value for (r, j), value in o.items()},
        "attn_q_norm.weight": attn_norm,
        "attn_k_norm.weight": attn_norm,
        "post_attention_norm.weight": norm,
        "ffn_gate.weight": cols(dict(gate), source),
        "ffn_up.weight": cols(dict(up), source),
        "ffn_down.weight": down_remapped,
    }
    report.update(width=WIDTH, query_heads=QUERY, kv_heads=GROUPS, head_width=HEAD,
                  raw_output_lanes=len(raw), physical_work_lanes=len(work), scratch_lanes=len(scratch),
                  live_physical_lanes=len(reserved) + len(work) + len(temp),
                  unused_physical_lanes=len(free) - len(work) - len(temp),
                  staged_dispatch_columns=len(colors), physical_dispatch_columns=len(merged),
                  ineligible_dispatch_coefficient_bound=ineligible, raw_seed_wrom=[2048, 2304],
                  projected_seed_wrom=[side_start, side_start + 256],
                  action_coordinates=[start_action, max(action.values())],
                  dispatch_coordinates=[start_dispatch, start_dispatch + len(merged) - 1],
                  all_public_raw_feedback_identity=True, seed_projection_and_dispatch_disjoint=True,
                  raw_feedback_lanes=sorted(raw), projected_work_by_raw_lane={i - s.WORK: v for i, v in work.items()},
                  attention_scratch_by_staged_lane=temp, dispatch_by_staged_lane=dispatch)
    return tensors, start_action, report


if __name__ == "__main__":
    c, co, ports, gates, report = build()
    ids, scores, gate, up, down, rows = co
    report.update(down_destinations=len({d for (d, r), v in down if v}),
                  ffn_gate_sources=len({j for (r, j), v in gate if v}),
                  scores_per_lane_class=dict(Counter("action" if col < s.DISPATCH else "dispatch"
                                                    for _, col in scores)))
    tensors, action_start, report = remap(c, co, ports, gates, report)
    (HERE / "compile-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "dead_persistent_lanes"}, indent=2))
