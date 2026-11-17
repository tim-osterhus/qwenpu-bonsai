"""Audit symbolic phase packing using the sparse 9B compiler. No model I/O."""

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "full9b" / "compiler"))
import compile_full9b as full


WIDTH = 2560
WITNESS_ALIGNMENT = 512


def require(condition, description):
    if not condition:
        raise ValueError(description)


def mask(values):
    result = 0
    for value in values:
        result |= 1 << value
    return result


def union_bits(values):
    result = 0
    for value in values:
        result |= value
    return result


def members(bits):
    while bits:
        lowest = bits & -bits
        yield lowest.bit_length() - 1
        bits ^= lowest


def sparse_rows(entries, count, transpose=False):
    rows = [{} for _ in range(count)]
    for (first, second), value in dict(entries).items():
        row, col = (second, first) if transpose else (first, second)
        if value:
            rows[row][col] = value
    return rows


def inventory(circuit, ids, ports, gates, down):
    s = full.s
    entry = set(s.bits(s.ENTRY))
    handle = set(s.bits(s.HANDLE_IN, s.HANDLE_BITS))
    ingress = (entry | handle | set(s.bits(s.WRITE_ADDR, 33)) | set(s.bits(s.WRITE_WORD))
               | set(s.bits(s.READ_ADDR, 33)) | {s.READ_MARK} | set(circuit.persistent)
               | set(circuit.dot_ephemeral))
    q, k, v, o = full.attention(circuit, ids, ports, gates)
    attention_inputs = {
        name: {col - s.WORK for (_, col), value in matrix.items() if value and s.WORK <= col < s.ACTION}
        for name, matrix in (("q", q), ("k", k), ("v", v))
    }
    readback = set(s.bits(s.READBACK, 57))
    raw = ({lane for row in down for lane in row if lane < s.DISPATCH} | readback
           | set(circuit.persistent) | set(circuit.dot_ephemeral)
           | set(s.bits(s.READ_ADDR, 33)) | {s.READ_MARK}
           | set(s.bits(s.WRITE_ADDR, 33)) | set(s.bits(s.WRITE_WORD)))
    attention_outputs = {lane for (lane, _), value in o.items() if value}
    scratch = attention_outputs - raw
    require(attention_outputs & raw == readback, "New attention output hits a raw lane")
    require(ingress - raw == entry | handle, "Changed host-only ingress lanes")

    # Keep the current host field indices, including compatibility observer fields
    # used only at retirements or on the legacy tile action.
    observer = (full.HERE / "full9b.observer.txt").read_text().splitlines()
    require(observer[0] == "QPVXOBSERVER2", "Changed observer format")
    observed = {int(observer[5].split()[0])}
    for line in (1, 2, 3, 4, 6, 7, 8):
        count, *lanes = map(int, observer[line].split())
        require(count == len(lanes), "Invalid observer lane count")
        observed.update(lanes)
    observed.update(range(s.OLD_HANDLE, s.BUS_WIDTH + 2))
    observed.update((*s.bits(s.BUS_ADDR), *s.bits(s.BUS_WORD), *s.bits(s.EXIT_CODE), s.PUTCHAR_FLAG))
    dot_ports = {lane for value in ports.values() for lane in ((value,) if isinstance(value, int) else value)}
    fixed = set().union(observed, dot_ports, readback, *attention_inputs.values()) & (raw | ingress)
    candidates = (raw & ingress) - fixed
    require(candidates <= set(circuit.persistent), "Candidate does not follow the persistent feedback protocol")
    return ingress, raw, scratch, attention_inputs, fixed, candidates


def summarize_rows(circuit, ids, gate, up, down, candidates, scratch):
    s, dot = full.s, full.dot
    names = tuple(ids)
    actions = set(range(s.ACTION, s.ACTION + len(ids)))
    destinations = [{} for _ in ids]
    default_reads = [0 for _ in ids]
    base_reads = [0 for _ in ids]
    row_phases = []
    analog = defaultdict(set)
    ephemeral_writers = defaultdict(set)
    declared_uses = set()
    for row, (g, u, d) in enumerate(zip(gate, up, down, strict=True)):
        action_sources = u.keys() & actions
        if action_sources:
            require(u.keys() <= actions and set(u.values()) == {16}, f"Mixed up sources in row {row}")
            phases = sorted(col - s.ACTION for col in action_sources)
        else:
            require(len(u) == 1 and next(iter(u)) - s.WORK in circuit.dot_ephemeral,
                    f"Unclassified unconditional up in row {row}")
            source = next(iter(u)) - s.WORK
            phases = sorted(col - s.ACTION for col in g.keys() & actions)
            require(tuple(names[p] for p in phases) in ((dot.FN, dot.FP), (dot.SUM,)),
                    f"Unclassified dot phase in row {row}")
            require(not d.keys() & candidates, f"Unconditional dot row {row} writes an allocatable lane")
            analog[source].update(phases)
        require(phases, f"Unclassified FFN row {row}")
        row_phases.append(tuple(phases))
        work_reads = {col - s.WORK for col in g.keys() | u.keys() if s.WORK <= col < s.ACTION}
        read_mask = mask(work_reads & candidates)
        direct_reads = (g.keys() | u.keys()) - actions - {s.WORK + lane for lane in work_reads}
        require(direct_reads <= scratch | set(s.bits(s.READBACK, 57)), f"Unexpected direct FFN source in row {row}")
        for phase in phases:
            name = names[phase]
            for dest in d:
                if dest in circuit.dot_ephemeral:
                    ephemeral_writers[dest].add(phase)
                if dest not in candidates:
                    base_reads[phase] |= read_mask
                    declared_uses.update(work_reads & candidates)
                elif dest in circuit.effects[name]:
                    destinations[phase][dest] = destinations[phase].get(dest, 0) | read_mask
                    declared_uses.update(work_reads & candidates)
                else:
                    require(not (name in (s.SEED_MISS, s.SEED_WORD) or dest not in circuit.persistent),
                            f"Unexpected default write {name}/{dest}")
                    require(not read_mask or (name != s.START and read_mask == 1 << dest),
                            f"Default write {name}/{dest} uses another candidate")
                    default_reads[phase] |= read_mask
    for source, phases in analog.items():
        viable = {ids[target] for p in ephemeral_writers[source] for target in circuit.routes[names[p]]}
        require(viable == phases, f"Dot up support for lane {source} crosses its indicated action mask")
    return destinations, default_reads, base_reads, row_phases, analog, declared_uses


def solve(ids, circuit, destinations, defaults, base_reads, preserve_explicit):
    names = tuple(ids)
    # Include all declared edges, including targets which require an impossible
    # combination of flags. Every one of the 513 compiled IDs is accounted for.
    successors = [[ids[t] for t in circuit.routes[name]] for name in names]
    predecessors = [set() for _ in ids]
    for phase, targets in enumerate(successors):
        for target in targets:
            predecessors[target].add(phase)

    live_in = [0 for _ in ids]
    live_out = [0 for _ in ids]
    dep_masks = [tuple((1 << dest, reads) for dest, reads in values.items()) for values in destinations]
    explicit = [mask(values) for values in destinations]
    pending = set(range(len(ids)))
    while pending:
        phase = pending.pop()
        after = 0
        for target in successors[phase]:
            after |= live_in[target]
        live_out[phase] = after
        before = base_reads[phase] | (after & defaults[phase])
        retained = after | (explicit[phase] if preserve_explicit else 0)
        for dest, reads in dep_masks[phase]:
            if retained & dest:
                before |= reads
        if before != live_in[phase]:
            require((before | live_in[phase]) == before, "Liveness is not monotonic")
            live_in[phase] = before
            pending.update(predecessors[phase])
    return live_in, live_out, [after | (written if preserve_explicit else 0)
                               for after, written in zip(live_out, explicit, strict=True)]


def short_path_peaks(ids, circuit, destinations, defaults, base_reads, floor):
    """Use only phase sequences from completed short receipts, with branches fixed."""
    names = tuple(ids)
    evidence = full.HERE.parent / "validation"
    rows = []
    for fixture in ("scalar", "vector-short", "attention"):
        path = evidence / fixture / "guest.json"
        data = path.read_bytes()
        for index, run in enumerate(json.loads(data)["runs"]):
            events = run["actions"]
            require(events[-1]["action"] in (ids[full.s.HALT], ids[full.s.FAULT]), f"Incomplete receipt {fixture}")
            live = peak = 0
            for event, successor in zip(reversed(events[:-1]), reversed(events[1:]), strict=True):
                phase = event["action"]
                target = successor["action"]
                require(event["next"] == target and names[target] in circuit.routes[names[phase]],
                        f"Incorrect transition in receipt {fixture}")
                live = (base_reads[phase] | (live & defaults[phase]) |
                        union_bits(reads for dest, reads in destinations[phase].items() if live & (1 << dest)))
                if live.bit_count() > peak:
                    peak = live.bit_count()
                    witness = dict(phase=names[phase], following_phase=names[target], position=event["position"],
                                   lanes=list(members(live)))
            rows.append(dict(fixture=fixture, run=index, receipt_sha256=sha256(data).hexdigest(),
                             decoded_steps=len(events) - 1, peak_private_inputs=peak,
                             lower_bound_with_fixed_banks=floor + 2 * peak, peak=witness))
    return rows


def color(cliques):
    neighbors = defaultdict(int)
    for clique in cliques:
        for lane in members(clique):
            neighbors[lane] |= clique ^ (1 << lane)
    degree = {lane: links.bit_count() for lane, links in neighbors.items()}
    adjacent_colors = dict.fromkeys(neighbors, 0)
    colors = {}
    while adjacent_colors:
        lane = max(adjacent_colors, key=lambda i: (adjacent_colors[i].bit_count(), degree[i], -i))
        used = adjacent_colors.pop(lane)
        chosen = next(i for i in range(len(neighbors)) if not used & (1 << i))
        colors[lane] = chosen
        for other in members(neighbors[lane]):
            if other in adjacent_colors:
                adjacent_colors[other] |= 1 << chosen
    for clique in cliques:
        require(len({colors[lane] for lane in members(clique)}) == clique.bit_count(), "Interference collision")
    return colors


def layout(width, circuit, ids, raw, ingress, scratch, candidates, colors, dispatch_count):
    seed = set(circuit.dot_payload_ingress)
    fixed_raw = raw - candidates
    host_only = (ingress - raw) | seed
    score_blocks = [set(range(start, start + 32)) for start in range(0, width, 32)
                    if not set(range(start, start + 32)) & (fixed_raw | host_only)]
    action_blocks = (len(ids) + 31) // 32
    dispatch_blocks = (dispatch_count + 31) // 32
    require(len(score_blocks) >= action_blocks + dispatch_blocks, "Too few isolated Q4 score blocks")
    action = sorted(set.union(*score_blocks[:action_blocks]))[:len(ids)]
    dispatch = sorted(set.union(*score_blocks[action_blocks:action_blocks + dispatch_blocks]))[:dispatch_count]
    score_padding = set.union(*score_blocks[:action_blocks + dispatch_blocks]) - set(action) - set(dispatch)
    flexible = set(range(width)) - fixed_raw - set(action) - set(dispatch) - score_padding
    n = len(set(colors.values()))
    color_raw = sorted(flexible - host_only)[:n]
    residual = sorted(flexible - set(color_raw))
    projected_fixed = sorted(ingress - candidates)
    count = len(projected_fixed) + n + len(scratch) + len(seed)
    require(len(color_raw) == n and len(residual) >= count, "Physical budget exhausted")
    work = dict(zip(projected_fixed, residual[:len(projected_fixed)], strict=True))
    start = len(work)
    color_work = residual[start:start + n]
    start += n
    attn = dict(zip(sorted(scratch), residual[start:start + len(scratch)], strict=True))
    start += len(scratch)
    side = residual[start:start + len(seed)]
    outputs = fixed_raw | set(color_raw) | set(action) | set(dispatch)
    inputs = set(work.values()) | set(color_work) | set(attn.values()) | set(side)
    require(not (outputs & (inputs | host_only)), "Projected roles or host inputs enter feedback")
    require(len(inputs) == count and len(outputs) == len(fixed_raw) + n + len(action) + len(dispatch),
            "Physical lane collision")
    require(not ({v // 32 for v in action} & {v // 32 for v in dispatch}), "Q4 scale blocks collide")
    require(not score_padding & (outputs | inputs | host_only), "Live role in reserved Q4 score padding")
    return dict(raw={lane: color_raw[color] for lane, color in colors.items()},
                projected={**work, **{lane: color_work[color] for lane, color in colors.items()}},
                attn_scratch=attn, side=side, action=action, dispatch=dispatch,
                live_lanes=len(outputs | inputs), remaining=width - len(outputs | inputs),
                projected_lanes_in_raw_seed_region=len(inputs & seed), q4_blocks_available=len(score_blocks),
                q4_padding_reserved=len(score_padding))


def remap_rows(gate, up, down, phases, kept, candidates, allocation, dispatch_map):
    s = full.s
    phase_masks = defaultdict(int)
    for phase, lanes in enumerate(kept):
        for lane in members(lanes):
            phase_masks[lane] |= 1 << phase
    mapped = defaultdict(dict)
    row_fragments = removed = splits = 0
    for i, (g, u, d, active) in enumerate(zip(gate, up, down, phases, strict=True)):
        full_mask = mask(active)
        groups = defaultdict(dict)
        for destination, coefficient in d.items():
            keep = full_mask & phase_masks[destination] if destination in candidates else full_mask
            if keep:
                if destination in candidates:
                    new_dest = allocation["raw"][destination]
                elif destination >= s.DISPATCH:
                    new_dest = dispatch_map[destination]
                else:
                    new_dest = destination
                # Zero or repeated writes to a live color would invalidate row aliasing.
                require(new_dest not in groups[keep], f"Merged destinations in row {i}")
                groups[keep][new_dest] = coefficient
        removed += not groups
        splits += max(0, len(groups) - 1)
        row_fragments += len(groups)

        def remap_terms(terms, for_phases):
            result = {}
            for col, coeff in terms.items():
                if s.WORK <= col < s.ACTION:
                    lane = col - s.WORK
                    if lane not in allocation["projected"]:
                        raise ValueError(f"Retained row {i} uses dead lane {lane}")
                    new_col = allocation["projected"][lane]
                elif s.ACTION <= col < s.DISPATCH:
                    action = col - s.ACTION
                    if not for_phases & (1 << action):
                        continue
                    new_col = allocation["action"][action]
                else:
                    new_col = allocation["attn_scratch"].get(col, col)
                require(new_col not in result, f"Merged input columns in row {i}")
                result[new_col] = coeff
            return tuple(sorted(result.items()))

        for keep, outputs in groups.items():
            require(keep == full_mask or all(s.ACTION <= col < s.DISPATCH for col in u),
                    f"Split of an unconditional analogue up in row {i}")
            signature = remap_terms(g, keep), remap_terms(u, keep)
            for dest, coefficient in outputs.items():
                merged = mapped[signature]
                merged[dest] = merged.get(dest, 0) + coefficient
    return dict(row_fragments_before_deduplication=row_fragments, original_rows_removed=removed,
                added_fragments_from_splitting=splits,
                unique_gate_up_rows=sum(any(outputs.values()) for outputs in mapped.values()))


def analyze():
    circuit, (ids, scores, gate, up, down, rows), ports, gates, _ = full.build()
    gate = sparse_rows(gate, rows)
    up = sparse_rows(up, rows)
    down = sparse_rows(down, rows, transpose=True)
    ingress, raw, scratch, attn, fixed, candidates = inventory(circuit, ids, ports, gates, down)
    dest, defaults, base_reads, phase_rows, analog, used = summarize_rows(circuit, ids, gate, up, down, candidates, scratch)
    colors, merged, ineligible = full.coalesce_dispatch(circuit, ids, scores)
    names = tuple(ids)
    original = json.loads((full.HERE / "compile-audit.json").read_text())
    require((len(raw), len(ingress), len(scratch), rows, len(ids), len(merged)) ==
            tuple(original[key] for key in ("raw_output_lanes", "physical_work_lanes", "scratch_lanes",
                                            "ffn_rows_used", "actions", "physical_dispatch_columns")),
            "Source inventory differs from the retained audit")
    fixed_floor = (len(raw - candidates) + len(ingress - candidates) + len(scratch)
                   + len(ids) + len(merged) + len(circuit.dot_payload_ingress))
    max_colors = (WIDTH - fixed_floor) // 2
    direct = [pinned | union_bits(values.values()) for pinned, values in zip(base_reads, dest, strict=True)]
    direct_peak = max(range(len(ids)), key=lambda i: direct[i].bit_count())
    report = dict(width=WIDTH, original_width=original["width"], original_live=original["live_physical_lanes"],
                  actions=len(ids), original_ffn_rows=rows, raw=len(raw), projected=len(ingress), scratch=len(scratch),
                  q_k_v_work_lanes={name: len(lanes) for name, lanes in attn.items()},
                  pinned_raw_lanes=len(raw - candidates), pinned_projected_lanes=len(ingress - candidates),
                  pinned_both_lanes=len(fixed & raw & ingress),
                  private_feedback_candidates=len(candidates),
                  private_consumed_outside_defaults=len(used),
                  private_default_only=len(candidates - used),
                  analog_up_rows=sum(any(col < full.s.ACTION for col in row) for row in up),
                  analog_up_lanes=len(analog), dispatch_original_columns=len(colors),
                  dispatch_physical_columns=len(merged), ineligible_dispatch_factor_bound=ineligible,
                  private_color_budget_at_target=max_colors, preserve_all_explicit_direct_use_peak=direct[direct_peak].bit_count(),
                  preserve_all_explicit_direct_use_peak_phase=names[direct_peak],
                  preserve_all_explicit_direct_use_lanes=list(members(direct[direct_peak])),
                  short_path_peaks=short_path_peaks(ids, circuit, dest, defaults, base_reads, fixed_floor))
    for preserve_explicit in (True, False):
        incoming, outgoing, emitted = solve(ids, circuit, dest, defaults, base_reads, preserve_explicit)
        assignments = color(incoming + emitted)
        peak_in = max(range(len(ids)), key=lambda i: incoming[i].bit_count())
        peak_out = max(range(len(ids)), key=lambda i: emitted[i].bit_count())
        peak = max(incoming[peak_in].bit_count(), emitted[peak_out].bit_count())
        private = len(set(assignments.values()))
        result = dict(live_input_peak=incoming[peak_in].bit_count(), live_input_peak_phase=names[peak_in],
                      input_peak_lanes=list(members(incoming[peak_in])),
                      live_or_retained_output_peak=emitted[peak_out].bit_count(), live_or_retained_output_peak_phase=names[peak_out],
                      output_peak_lanes=list(members(emitted[peak_out])),
                      candidate_lanes_in_any_phase=len(assignments), candidate_colors=private,
                      candidate_lane_roles_saved=len(candidates) - private, phase_clique_lower_bound=peak,
                      disjoint_bank_floor=fixed_floor, physical_lane_lower_bound=fixed_floor + 2 * peak,
                      physical_lane_upper_bound=fixed_floor + 2 * private,
                      original_private_raw_or_projected_roles=2 * len(candidates))
        lane_count = result["physical_lane_upper_bound"]
        witness_width = (lane_count + WITNESS_ALIGNMENT - 1) // WITNESS_ALIGNMENT * WITNESS_ALIGNMENT
        allocation = layout(witness_width, circuit, ids, raw, ingress, scratch, candidates, assignments, len(merged))
        require(allocation["live_lanes"] == lane_count, "Physical count differs from allocation")
        result.update(target_lane_deficit=max(0, lane_count - WIDTH), coordinate_witness_width=witness_width,
                      coordinate_witness_unused=allocation["remaining"], q4_blocks_available=allocation["q4_blocks_available"],
                      q4_padding_reserved=allocation["q4_padding_reserved"],
                      coordinate_witness_free=allocation["remaining"] - allocation["q4_padding_reserved"],
                      projected_lanes_in_raw_seed_region=allocation["projected_lanes_in_raw_seed_region"])
        result.update(remap_rows(gate, up, down, phase_rows, emitted, candidates, allocation,
                                 {col: allocation["dispatch"][colour] for col, colour in colors.items()}))
        report["prune_defaults_only" if preserve_explicit else "prune_dead_private_outputs"] = result
    return report


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
