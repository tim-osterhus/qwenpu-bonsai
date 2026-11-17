"""Sparse proof of ADD return invariants and context-sensitive private liveness."""

from collections import Counter, defaultdict, deque
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "grid")]
import grid_census as grid

compact, live = grid.compact, grid.compact.live
full, s = live.full, live.full.s
ADD_CORE = ("ADD_INIT", "PREFIX_1", "PREFIX_2", "PREFIX_4", "PREFIX_8", "PREFIX_16", "ADD_FINISH")
ROOTS = ("SEED_MISS", "SEED_WORD", "START", "FETCH")


def prove_calls(c, ids, gates, ups, downs, phases, private):
    """Match every real sparse return write to its fully initialized source."""
    returns = c.returns
    flags = set(returns.values())
    assert len(flags) == len(returns) == 34 and flags <= private
    assert tuple(ids) == tuple(sorted(ids, key=ids.__getitem__))
    names = tuple(ids)
    assert c.routes["ADD_FINISH"] == {f"AFTER_{mode}": flag for mode, flag in returns.items()}
    for phase, target in zip(ADD_CORE, ADD_CORE[1:]):
        assert c.routes[phase] == {target: None}
    callers = {}
    flag_columns = {s.WORK + flag for flag in flags}
    for phase in names:
        effects = c.effects[phase]
        writes = effects.keys() & flags
        if writes:
            assert writes == flags and c.routes[phase] == {ADD_CORE[0]: None}
            modes = [mode for mode, lane in returns.items() if effects[lane] == s.constant(1)]
            assert len(modes) == 1
            assert all(effects[lane] == s.constant(1 if mode == modes[0] else -1)
                       for mode, lane in returns.items())
            callers[phase] = modes[0]
        for expression in effects.values():
            assert not any(src in flag_columns for _, terms in expression for src, _ in terms)
        for flag in c.routes[phase].values():
            assert flag is None or abs(flag) not in flags or phase == ADD_CORE[-1]
        assert (ADD_CORE[0] in c.routes[phase]) == (phase in callers)
        expected = set(ADD_CORE[1:]) & c.routes[phase].keys()
        assert not expected or (phase in ADD_CORE[:-1] and expected == {ADD_CORE[ADD_CORE.index(phase)+1]})
    found = defaultdict(dict)
    score_read_rows = set()
    for i, (g, u, d, source_phases) in enumerate(zip(gates, ups, downs, phases, strict=True)):
        flag_writes = d.keys() & flags
        reads = (g.keys() | u.keys()) & flag_columns
        if not (flag_writes or reads):
            continue
        if reads:
            assert len(reads) == 1
            lane = next(iter(reads)) - s.WORK
            assert all(dest == lane or dest >= s.DISPATCH for dest in d)
            if any(dest >= s.DISPATCH for dest in d):
                assert tuple(names[p] for p in source_phases) == (ADD_CORE[-1],)
                score_read_rows.add(i)
        if flag_writes:
            terms = compact.sparse({j: v for j, v in g.items() if not s.ACTION <= j < s.DISPATCH})
            for p in source_phases:
                col = s.ACTION + p
                assert u[col] == 16 and g[col] in (-4, 4)
                for dest in flag_writes:
                    values = found[p, dest]
                    sig = (terms, g[col])
                    values[sig] = values.get(sig, 0) + d[dest]
    checks = Counter()
    for p, phase in enumerate(names):
        for mode, dest in returns.items():
            if phase in (s.SEED_MISS, s.SEED_WORD):
                expected = {}
                label = "zero_seed"
            elif phase in callers or phase == s.START:
                val = 1 if callers.get(phase) == mode else -1
                expected = {((), 4): val, ((), -4): -val}
                label = "call_literal" if phase in callers else "start_literal"
            else:
                term = ((s.WORK + dest, 512),)
                expected = {(term, 4): 1, (term, -4): -1, ((), 4): -1, ((), -4): 1}
                label = "default_refresh"
            assert {k: v for k, v in found[p, dest].items() if v} == expected, (phase, mode)
            checks[label] += 1
    return callers, dict(checks), len(score_read_rows)


def graph(c, ids, callers, contextual):
    """Keep every declared branch except ADD_FINISH's proven return test."""
    roots = [(ids[p], None) for p in ROOTS]
    states, edges = list(roots), []
    reached = {state: i for i, state in enumerate(states)}
    names = tuple(ids)
    for p, mode in states:
        phase = names[p]
        choices = c.routes[phase]
        if contextual and phase == ADD_CORE[-1]:
            assert mode is not None
            choices = [f"AFTER_{mode}"]
        successors = []
        for nxt in choices:
            next_mode = ((callers[phase] if phase in callers else mode)
                         if contextual and nxt in ADD_CORE else None)
            if contextual and nxt in ADD_CORE:
                assert next_mode is not None
            key = (ids[nxt], next_mode)
            if key not in reached:
                reached[key] = len(states)
                states.append(key)
            successors.append(reached[key])
        edges.append(tuple(successors))
    return states, edges


def active_deps(dest, defaults):
    return [(tuple((1 << lane, mask) for lane, mask in writes.items()), refresh)
            for writes, refresh in zip(dest, defaults, strict=True)]


def solve(states, edges, deps, base_reads, forced=None):
    """Least backward fixed point. Optionally keep fixed-phase real writes live."""
    callers = [[] for _ in states]
    for caller, targets in enumerate(edges):
        for target in targets:
            callers[target].append(caller)
    incoming = [0] * len(states)
    outgoing = [0] * len(states)
    todo, queued = deque(range(len(states))), set(range(len(states)))
    while todo:
        i = todo.popleft()
        queued.remove(i)
        phase, _ = states[i]
        outgoing[i] = after = live.union_bits(incoming[j] for j in edges[i])
        emitted = after if forced is None else forced[phase]
        explicit, defaults = deps[phase]
        result = base_reads[phase] | (emitted & defaults)
        result |= live.union_bits(d for bit, d in explicit if emitted & bit)
        if incoming[i] != result:
            assert incoming[i] & result == incoming[i]
            incoming[i] = result
            for parent in callers[i]:
                if parent not in queued:
                    todo.append(parent)
                    queued.add(parent)
    # A fixed pruning schedule is invalid if an omitted output is newly needed.
    return incoming, outgoing


def summary(states, ins, outs, names):
    i = max(range(len(ins)), key=lambda j: ins[j].bit_count())
    o = max(range(len(outs)), key=lambda j: outs[j].bit_count())
    assert len(ins) == len(states) == len(outs)
    return dict(input_peak=ins[i].bit_count(), input_peak_phase=names[states[i][0]],
                input_peak_return=states[i][1], input_peak_lanes=list(live.members(ins[i])),
                output_peak=outs[o].bit_count(), output_peak_phase=names[states[o][0]],
                output_peak_return=states[o][1])


def phase_writes(down, phases, private, size):
    written = [0] * size
    for row, active in zip(down, phases, strict=True):
        destinations = live.mask(row.keys() & private)
        if destinations:
            for phase in active:
                written[phase] |= destinations
    return written


def phase_union(states, masks, size):
    merged = [0] * size
    for (phase, _), lanes in zip(states, masks, strict=True):
        merged[phase] |= lanes
    return merged


def fixed_phase_pruning(states, edges, deps, base, actual, ideal):
    """Find a keep mask usable with the original phase-only emit predicates."""
    forced = [0] * len(actual)
    ins, outs = ideal
    solves = 0
    while True:
        union = phase_union(states, outs, len(actual))
        keep = [prior | (wanted & writes) for prior, wanted, writes in zip(forced, union, actual, strict=True)]
        if keep == forced:
            assert all(not wanted & writes & ~saved
                       for wanted, writes, saved in zip(union, actual, keep, strict=True))
            return ins, outs, keep, solves
        forced = keep
        ins, outs = solve(states, edges, deps, base, forced)
        solves += 1


def verified_fixed_map_clique(cliques):
    """A checked clique in the fixed-color interference graph."""
    neighbors = defaultdict(int)
    distinct = sorted(set(cliques), key=lambda mask: (-mask.bit_count(), mask))
    for group in distinct:
        for lane in live.members(group):
            neighbors[lane] |= group ^ (1 << lane)
    indices = sorted(neighbors, key=lambda lane: (-neighbors[lane].bit_count(), lane))[:12]
    starts = [0, *distinct[:12], *(1 << lane for lane in indices)]
    best = starts[0]
    for candidate in starts:
        choices = live.mask(neighbors)
        for lane in live.members(candidate):
            choices &= neighbors[lane]
        while choices:
            chosen = max(live.members(choices), key=lambda lane: ((neighbors[lane] & choices).bit_count(), -lane))
            candidate |= 1 << chosen
            choices &= neighbors[chosen]
        if candidate.bit_count() > best.bit_count():
            best = candidate
    assert all((best ^ (1 << lane)) & ~neighbors[lane] == 0 for lane in live.members(best))
    return dict(slots=best.bit_count(), lanes=list(live.members(best)))


def certify_clique(clique, states, ins, outs, names):
    """Cover each clique edge with an actual state; encode subsets by clique index."""
    lanes = clique["lanes"]
    mask = live.mask(lanes)
    indices = {lane: i for i, lane in enumerate(lanes)}
    possible = {}
    for bank, support in (("input", ins), ("output", outs)):
        for (phase, ctx), slots in zip(states, support, strict=True):
            if (selected := slots & mask).bit_count() >= 2:
                dense = live.mask(indices[lane] for lane in live.members(selected))
                possible.setdefault(dense, (bank, names[phase], ctx))
    count = len(lanes)
    full_mask = (1 << count) - 1
    remaining = [full_mask ^ ((2 << i) - 1) for i in range(count)]
    total = count * (count - 1) // 2
    witness = []
    while total:
        chosen, covered = max(((bits, sum((remaining[i] & bits).bit_count() for i in live.members(bits)))
                               for bits in possible), key=lambda item: item[1])
        assert covered
        bank, phase, ctx = possible.pop(chosen)
        for i in live.members(chosen):
            remaining[i] &= ~chosen
        total -= covered
        witness.append(dict(bank=bank, phase=phase, context=ctx, slots=chosen.bit_count(),
                            dense_mask_hex=format(chosen, "x"), newly_covered_edges=covered))
    assert not any(remaining)
    clique["edge_count"] = count * (count - 1) // 2
    clique["actual_state_edge_witnesses"] = witness


def known_paths(ids, c, dest, defaults, old_base, grid_base, callers):
    """Replay only the saved phase schedule, leaving actual trace data untouched."""
    legacy = live.short_path_peaks(ids, c, dest, defaults, old_base, 2342)
    grid_paths = live.short_path_peaks(ids, c, dest, defaults, grid_base, 1737)
    fixture_root = full.HERE.parent / "validation"
    names = tuple(ids)
    witnesses = []
    for a, b in zip(legacy, grid_paths, strict=True):
        assert (a["fixture"], a["run"], a["receipt_sha256"]) == (b["fixture"], b["run"], b["receipt_sha256"])
        rows = json.loads((fixture_root / a["fixture"] / "guest.json").read_text())["runs"][a["run"]]["actions"]
        pending = None
        checked_calls = 0
        for row, after in zip(rows, rows[1:]):
            phase, next_phase = names[row["action"]], names[after["action"]]
            if phase in callers:
                assert pending is None and next_phase == ADD_CORE[0]
                pending = callers[phase]
            elif phase in ADD_CORE:
                assert pending is not None
                step = ADD_CORE.index(phase)
                if step + 1 == len(ADD_CORE):
                    assert next_phase == f"AFTER_{pending}"
                    pending = None
                    checked_calls += 1
                else:
                    assert next_phase == ADD_CORE[step+1]
        assert pending is None
        witnesses.append(dict(fixture=a["fixture"], run=a["run"], receipt_sha256=a["receipt_sha256"],
                              decoded_steps=a["decoded_steps"], exact_add_returns_checked=checked_calls,
                              legacy_peak=a["peak_private_inputs"], grouped_writer_peak=b["peak_private_inputs"],
                              peak_phase=b["peak"]["phase"]))
    return witnesses


def main():
    circuit, (ids, scores, gates, ups, downs, rows), ports, gates_dot, _ = full.build()
    gates, ups, downs = (live.sparse_rows(gates, rows), live.sparse_rows(ups, rows),
                         live.sparse_rows(downs, rows, transpose=True))
    ingress, raw, scratch, qkv, _, private = live.inventory(circuit, ids, ports, gates_dot, downs)
    dest, defaults, old_base, phases, _, _ = live.summarize_rows(circuit, ids, gates, ups, downs, private, scratch)
    calls, return_checks, score_rows = prove_calls(circuit, ids, gates, ups, downs, phases, private)
    old_graph = graph(circuit, ids, calls, False)
    new_graph = graph(circuit, ids, calls, True)
    names = tuple(ids)
    old_reachable = sorted({names[p] for p, _ in old_graph[0]}, key=ids.__getitem__)
    new_reachable = sorted({names[p] for p, _ in new_graph[0]}, key=ids.__getitem__)
    assert len(old_reachable) == 510 and set(new_reachable) <= set(old_reachable)
    q, k, v, _ = full.attention(circuit, ids, ports, gates_dot)
    ordinary = rows - len(compact.structured.dot_plane.compile_ff(False, True).g) - 4 * len(gates_dot)
    matrices = [q.items(), k.items(), v.items()]
    matrices += [[((i, lane), coeff) for i in range(ordinary, rows) for lane, coeff in kind[i].items()]
                 for kind in (gates, ups)]
    codes = {ids[name]: i for i, name in enumerate(old_reachable)}
    predicted = compact.structured.classes(
        [compact.structured.integer_rows(matrix, s.ACTION, codes) for matrix in matrices], codes)
    classes = [live.mask(codes[p] for p in group) for group in predicted]
    writer, transitions, _ = compact.transition_fragments(circuit, ids, old_reachable, classes, lambda lane: lane, False)
    # Strip existing dispatch before replacing it with the exact grid writer.
    data_down = [{col: weight for col, weight in d.items() if col < s.DISPATCH} for d in downs]
    ndest, ndefaults, base, *_ = live.summarize_rows(circuit, ids, gates, ups, data_down, private, scratch)
    assert ndest == dest and ndefaults == defaults
    for (terms, _bias, wanted), outputs in writer:
        assert outputs
        reads = live.mask(col - s.WORK for col, v in terms if v and col - s.WORK in private)
        for i in live.members(wanted):
            base[ids[old_reachable[i]]] |= reads
    assert all(n & o == n for n, o in zip(base, old_base, strict=True))
    deps = active_deps(dest, defaults)
    ideal, old_grid = solve(*new_graph, deps, base), solve(*old_graph, deps, base)
    old_legacy = solve(*old_graph, deps, old_base)
    report = {
        "source_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(grid.__file__), Path(compact.__file__), Path(live.__file__), Path(full.__file__))},
        "callers": calls,
        "return_invariant": {"return_slots": len(circuit.returns), "encoded_modes": circuit.returns,
                             "real_sparse_return_write_checks": return_checks,
                             "real_legacy_score_rows_reading_flags": score_rows,
                             "grid_writer_return_flag_reads": list(live.members(base[ids["ADD_FINISH"]]
                                                            & live.mask(circuit.returns.values())))},
        "graph": {"unrefined_phase_count": len(old_graph[0]), "contextual_state_count": len(new_graph[0]),
                  "contextual_phases": len(new_reachable), "unreached_after_context": sorted(set(old_reachable)-set(new_reachable)),
                  "unrefined_edges": sum(map(len, old_graph[1])), "contextual_edges": sum(map(len,new_graph[1])),
                  "expanded_core_contexts": dict(Counter(names[p] for p, mode in new_graph[0] if mode)),
                  "fault_and_halt_edges": {p: circuit.routes[p] for p in ("FAULT", "HALT")}},
        "grid_writer_route_cases": transitions,
        "prior_grid_ideal": summary(old_graph[0], *old_grid, names),
        "prior_legacy_pruned": summary(old_graph[0], *old_legacy, names),
        "context_ideal": summary(new_graph[0], *ideal, names),
    }
    for label, (ins, outs), (states, edges) in (
            ("prior_legacy_pruned", old_legacy, old_graph), ("context_ideal", ideal, new_graph)):
        colors = live.color(ins + outs)
        report[label]["fixed_map_greedy_colors_ignoring_extra_emits"] = len(set(colors.values()))
    clique = verified_fixed_map_clique(ideal[0] + ideal[1])
    certify_clique(clique, new_graph[0], *ideal, names)
    report["context_ideal"]["verified_fixed_map_clique"] = clique
    # The maximal input set must really be written on its exact predecessor;
    # a zero that is absent from the old FFN must not inflate this witness.
    actual = phase_writes(data_down, phases, private, len(ids))
    peak = report["context_ideal"]
    peak_predecessor = [new_graph[0][j] for j, targets in enumerate(new_graph[1])
                        if any(new_graph[0][i] == (ids[peak["input_peak_phase"]], peak["input_peak_return"])
                               for i in targets)]
    assert peak_predecessor == [(ids["ADD_INIT"], peak["input_peak_return"])]
    peak_mask = live.mask(peak["input_peak_lanes"])
    assert peak_mask & actual[ids["ADD_INIT"]] == peak_mask
    peak["input_peak_predecessor"] = "ADD_INIT"
    peak["input_peak_lanes_written_by_predecessor"] = peak_mask.bit_count()

    fixed = fixed_phase_pruning(*new_graph, deps, base, actual, ideal)
    fixed_in, fixed_out, kept, solves = fixed
    actual_fixed = summary(new_graph[0], fixed_in, fixed_out, names)
    actual_fixed["iterations"] = solves
    actual_fixed["candidate_phase_writes_total"] = sum(m.bit_count() for m in actual)
    actual_fixed["candidate_phase_writes_kept"] = sum(m.bit_count() for m in kept)
    actual_fixed["maximum_same_phase_retained_writes"] = max(m.bit_count() for m in kept)
    actual_fixed["maximum_same_phase_retained_writes_at"] = names[max(range(len(kept)), key=lambda i: kept[i].bit_count())]
    actual_fixed["fixed_map_colors_under_previous_no_duplicate_emits_remap"] = len(set(
        live.color(fixed_in + fixed_out + kept).values()))
    actual_fixed["scope"] = ("One global keep mask per original action; count all surviving output writes and their reads "
                             "in every context. Same-color duplicate default refresh would need a new expression-aware rewriter.")
    report["fixed_phase_pruning"] = actual_fixed
    report["known_paths"] = known_paths(ids, circuit, dest, defaults, old_base, base, calls)
    selected_fit = {}
    # 2342 = original fixed roles including 513 action + 63 dispatch.
    # Replace those banks with 67 grouped input roles and one isolated Q4 tile.
    fixed_floor = 2342 - 513 - 63 + 67 + 32
    slots = report["context_ideal"]["verified_fixed_map_clique"]["slots"]
    for label, saving in (("tested_half_wrom", 128), ("proposed_quarter_wrom", 192)):
        floor = fixed_floor - saving
        selected_fit[label] = dict(optimistic_fixed_roles=floor, available_paired_colors=(2560-floor)//2,
                                   single_state_witness=peak["input_peak"],
                                   single_state_roles=floor+2*peak["input_peak"],
                                   single_state_excess=floor+2*peak["input_peak"]-2560,
                                   fixed_map_clique=slots, fixed_map_min_roles=floor+2*slots,
                                   fixed_map_min_excess=floor+2*slots-2560)
    report["width_gate"] = {"target": 2560, "grid_input": 67, "isolated_q4_score_tile": 32,
                           "q4_score_unused_positions": 23, "guard_rows": 1,
                           "grid_input_pq2_tile_unused_positions": 61, "wrom_cases": selected_fit}
    # Return the proof inputs so follow-up analyses consume this exact graph.
    return report, (circuit, ids, new_graph, dest, defaults, base, deps, ideal,
                    private, ingress, raw, scratch, qkv, phases, gates, ups, data_down)


if __name__ == "__main__":
    print(json.dumps(main()[0], indent=2))
