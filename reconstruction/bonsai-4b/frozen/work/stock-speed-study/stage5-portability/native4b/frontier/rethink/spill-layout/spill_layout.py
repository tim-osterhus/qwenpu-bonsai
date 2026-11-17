"""Sparse production spill rewrite and fixed raw/projected lane pricing."""

from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "return-context"))
import return_context as ret

grid, compact, full, live, s = ret.grid, ret.compact, ret.full, ret.live, ret.s
M_LOW, M_HIGH, M_DIVISOR = 1104, 1136, 1168
POLICIES = {
    "unspilled_control": (),
    "division_high": (("M_DIV_STEP", "M_DIV_STEP", M_HIGH),),
    "division_high_and_shared_mul_low": (("M_DIV_STEP", "M_DIV_STEP", M_HIGH), ("M_MUL_ADD", "M_MUL", M_LOW)),
}


def transplant(circuit, ids, callers, policy):
    c = deepcopy(circuit)
    continuations = {mode: f"AFTER_{mode}" for mode in c.returns}
    calls = dict(callers)
    stores, reads, releases, slots = set(), set(), set(), []
    if policy:
        assert len(ids) == 513
        c.route("SPILL_RESERVED_ZERO_GUARD", s.FAULT)
    for index, (caller, mode, word) in enumerate(policy):
        assert [p for p, value in callers.items() if value == mode] == [caller]
        assert c.routes[caller] == {"ADD_INIT": None}
        address, version = 0x78000000 + 8 * index, 1792 + index
        c.reserve_state(version)
        c.fix(s.START, version, 0, 1)
        phases = tuple(f"SPILL_{step}_{mode}" for step in
                       ("STORE", "PREP_READ", "READ_NEW", "READ_OLD", "RETIRE"))
        store, prep, reload, old, release = phases
        word_lanes = s.bits(word)
        hit = 1742
        c.fix(caller, s.WRITE_ADDR, address, 33)
        c.put(caller, s.WRITE_ADDR + 2, s.buffer(s.WORK + version, -1))
        c.put(caller, version, s.buffer(s.WORK + version, -1))
        c.move(caller, s.WRITE_WORD, s.bits(s.WORK + word))
        c.routes[caller] = {}
        c.route(caller, store)
        # A/B and the return word were prepared in the existing caller.
        # The retained record becomes authoritative at this store.
        for lane in word_lanes:
            c.put(store, lane, ())
        c.route(store, "ADD_INIT")
        calls.pop(caller)
        calls[store] = mode
        stores.add(store)

        target = continuations[mode]
        del c.routes["ADD_FINISH"][target]
        c.route("ADD_FINISH", (prep, c.returns[mode]))
        continuations[mode] = prep
        for setup, sense in ((prep, 1), (reload, -1)):
            c.fix(setup, s.READ_ADDR, address, 33)
            c.put(setup, s.READ_ADDR + 2, s.buffer(s.WORK + version, sense))
            c.fix(setup, s.READ_MARK, 1, 1)
        c.route(prep, reload)
        c.move(reload, word, s.bits(s.READBACK))
        # Reuse the existing retained-record marker V[56]. This symbolic
        # hit test needs a stock margin, including the exact missing case.
        hit_expression = s.threshold(((s.READBACK + 56, 128), (s.WORK + s.ONE, -1)))
        c.put(reload, hit, hit_expression)
        c.route(reload, (old, hit), "SET_LW_RANGE")
        c.move(old, s.OLD_HANDLE, s.bits(s.READBACK + 32, s.HANDLE_BITS))
        c.put(old, s.OLD_HIT, hit_expression)
        c.route(old, release)
        c.route(release, target)
        c.vx.reads.update((reload, old))
        reads.update((reload, old))
        releases.add(release)
        slots.append(dict(caller=caller, mode=mode, data_lanes=list(word_lanes), version_lane=version,
                          phases=list(phases), old_continuation=target, miss_fault="SET_LW_RANGE",
                          typed_address_hex=[f"0x{address:08x}", f"0x{address+4:08x}"]))
    return c, calls, continuations, stores, reads, releases, slots


def flow(c, ids, calls, continuations):
    states = [(ids[p], None) for p in ret.ROOTS]
    indices = {key: i for i, key in enumerate(states)}
    names = tuple(ids)
    edges = []
    for p, mode in states:
        label = names[p]
        targets = (continuations[mode],) if label == "ADD_FINISH" else tuple(c.routes[label])
        links = []
        for target in targets:
            ctx = (calls.get(label, mode) if target in ret.ADD_CORE else None)
            assert ctx is not None or target not in ret.ADD_CORE
            key = (ids[target], ctx)
            if key not in indices:
                indices[key] = len(states)
                states.append(key)
            links.append(indices[key])
        edges.append(tuple(links))
    return states, edges


def reachable(c, ids):
    visited, todo = set(ret.ROOTS), list(ret.ROOTS)
    for label in todo:
        for target in c.routes[label]:
            if target not in visited:
                todo.append(target)
                visited.add(target)
    return tuple(label for label in ids if label in visited)


def next_fragments(c, ids, names, bit_count):
    """Compile the existing bipolar route blend for an expanded score word."""
    codes = {phase: i for i, phase in enumerate(names)}

    def code(phase, bit):
        return 1 if codes[phase] & (1 << bit) else -1

    terms_by_phase = defaultdict(dict)
    verified = 0
    for phase in names:
        routes = c.routes[phase]
        fallback = [target for target, flag in routes.items() if flag is None]
        assert len(fallback) <= 1
        base = fallback[0] if fallback else min(routes, key=ids.__getitem__)
        for bit in range(bit_count):
            dest = compact.NEXT + bit
            def add(terms, weight):
                weights = terms_by_phase[dest, terms]
                weights[phase] = weights.get(phase, 0) + weight
            base_sign = code(base, bit)
            changes = {}
            baseline = base_sign
            for target, flag in routes.items():
                if target == base:
                    continue
                assert flag is not None
                delta = (code(target, bit) - base_sign) // 2
                if delta:
                    changes[target] = delta
                    baseline += delta
                    expr = c.effects[phase].get(abs(flag), s.buffer(s.WORK + abs(flag)))
                    for factor, terms in expr:
                        add(terms, factor * delta * (1 if flag > 0 else -1))
            add((), baseline)
            for target in routes:
                assert baseline + sum(delta * (1 if t == target else -1) for t, delta in changes.items()) == code(target, bit)
                verified += 1
    groups = defaultdict(dict)
    for (dest, terms), weights in terms_by_phase.items():
        supports = defaultdict(int)
        for phase, value in weights.items():
            if value:
                supports[value] |= 1 << codes[phase]
        g = defaultdict(float)
        for lane, coeff in terms:
            g[lane] += 512 * coeff
        support = compact.sparse(g)
        for value, mask in supports.items():
            for bias, coef in ((4, value), (-4, -value)):
                for coeffs, factor in ((support, coef), ((), -coef)) if terms else (((), coef),):
                    outputs = groups[coeffs, bias, mask]
                    outputs[dest] = outputs.get(dest, 0) + factor
    assert verified % bit_count == 0
    return [(key, compact.sparse(out)) for key, out in groups.items() if any(out.values())], verified // bit_count


def ordinary_compile(c, original, cut):
    s.DISPATCH = s.ACTION + len(c.phases)
    ids, _, g, u, d, ordinary = full.vx.compile_program(c)
    return (ids, live.sparse_rows(g, ordinary) + original[14][cut:],
            live.sparse_rows(u, ordinary) + original[15][cut:],
            live.sparse_rows(d, ordinary, True) + original[16][cut:], ordinary)


def classify_grid(c, ids, codes, ports, gates, cut, gate, up, stores):
    q, k, v, o = full.attention(c, ids, ports, gates)
    for phase in stores:
        v[56, s.ACTION + ids[phase]] = 1 / 16
    matrices = [q.items(), k.items(), v.items()]
    matrices += [[((i, lane), value) for i, row in enumerate(kind) if i >= cut for lane, value in row.items()]
                 for kind in (gate, up)]
    classes = compact.structured.classes([compact.structured.integer_rows(matrix, s.ACTION, codes)
                                           for matrix in matrices], codes)
    masks = [live.mask(codes[p] for p in group) for group in classes]
    return (q, k, v, o), classes, grid.GridSelector(masks, 32, len(codes))


def core_metrics(states, ideal, ids, word, ctx):
    mask = live.mask(s.bits(word))
    return {phase: (ideal[0][states.index((ids[phase], ctx))] & mask).bit_count()
            for phase in ret.ADD_CORE}


def prove_old_handle_cold(c, ids, states, edges, releases):
    """A guest retirement must see a complete guest lookup after scratch use."""
    tracked = {*s.bits(s.OLD_HANDLE, s.HANDLE_BITS), s.OLD_HIT}
    work = {s.WORK + lane for lane in tracked}
    names = tuple(ids)
    dirty = set(i for j, links in enumerate(edges) if names[states[j][0]] in releases for i in links)
    frontier = list(dirty)
    for i in frontier:
        p, _ = states[i]
        name = names[p]
        assert name != s.RETIRE
        effects = c.effects[name]
        assert not any(lane in work for expr in effects.values() for _, terms in expr for lane, _ in terms)
        assert not any(flag is not None and abs(flag) in tracked and abs(flag) not in effects
                       for flag in c.routes[name].values())
        if tracked <= effects.keys():
            continue
        for nxt in edges[i]:
            if nxt not in dirty:
                frontier.append(nxt)
                dirty.add(nxt)
    return len(dirty)


def size_case(policy, original, prior, ports, gates_dot, original_ord):
    old_c, old_ids, old_graph, _, _, _, _, old_ideal, _, _, _, scratch, old_qkv, *_ = original
    assert not (set(range(1792, 1792 + len(policy))) | {1742}) & (original[9] | original[10])
    c, calls, continuations, stores, reads, releases, slots = transplant(old_c, old_ids, prior["callers"], policy)
    ids, g, u, d, ordinary = ordinary_compile(c, original, original_ord)
    assert all(ids[phase] == i for phase, i in old_ids.items())
    names = reachable(c, ids)
    codes = {ids[label]: i for i, label in enumerate(names)}
    score_bits = (len(codes) - 1).bit_length()
    tensors, groups, selector = classify_grid(c, ids, codes, ports, gates_dot, ordinary, g, u, stores)
    assert not policy or ids["SPILL_RESERVED_ZERO_GUARD"] == 513 and 513 not in codes
    ingress, raw, attn_scratch, qkv, _, private = live.inventory(c, ids, ports, gates_dot, d)
    assert qkv == old_qkv and attn_scratch == scratch
    data = [{dest: value for dest, value in row.items() if dest < s.DISPATCH} for row in d]
    dst, defaults, base, phases, _, _ = live.summarize_rows(c, ids, g, u, data, private, scratch)
    writer, writer_cases = next_fragments(c, ids, names, score_bits)
    for (terms, _bias, support), outputs in writer:
        assert outputs
        reads_private = live.mask(lane - s.WORK for lane, value in terms if value and lane - s.WORK in private)
        for code in live.members(support):
            base[ids[names[code]]] |= reads_private
    states, edges = flow(c, ids, calls, continuations)
    deps = ret.active_deps(dst, defaults)
    ideal = ret.solve(states, edges, deps, base)
    lower = ret.verified_fixed_map_clique(ideal[0] + ideal[1])
    ret.certify_clique(lower, states, *ideal, tuple(ids))
    actual = ret.phase_writes(data, phases, private, len(ids))
    fixed_in, fixed_out, kept, solves = ret.fixed_phase_pruning(states, edges, deps, base, actual, ideal)
    colors = live.color(fixed_in + fixed_out + kept)
    ncolor = len(set(colors.values()))
    old_clique = live.mask(prior["context_ideal"]["verified_fixed_map_clique"]["lanes"])
    new_clique = live.mask(lower["lanes"])
    qkvo_before = full.attention(old_c, old_ids, ports, gates_dot)
    changed = {}
    for kind, before, after in zip(("q", "k", "v", "o"), qkvo_before, tensors, strict=True):
        assert all(after.get(key) == value for key, value in before.items())
        changed[kind] = len(after.keys() - before.keys())
    for slot in slots:
        word, ctx = slot["data_lanes"][0], slot["mode"]
        slot.update(overlap_with_prior_504=(old_clique & live.mask(slot["data_lanes"])).bit_count(),
                    core_liveness_before=core_metrics(old_graph[0], old_ideal, old_ids, word, ctx),
                    core_liveness_after=core_metrics(states, ideal, ids, word, ctx))
    floor_no_rom = (len(raw - private) + len(ingress - private) + len(scratch)
                    + len(c.dot_payload_ingress) + selector.input_roles + 32)
    rom = {}
    for name, saving in (("half", 128), ("quarter", 192)):
        floor = floor_no_rom - saving
        rom[name] = dict(fixed_roles=floor, paired_limit=(2560-floor)//2,
                         ideal_fixed_map_lower_bound_roles=floor+2*lower["slots"],
                         fixed_emission_witness_roles=floor+2*ncolor, excess_at_2560=floor+2*ncolor-2560)
    result = dict(source_actions=len(ids), packed_reachable_actions=len(names), return_qualified_states=len(states),
                  grouped_rails=selector.input_roles, protected_classes=len(groups), grid_groups=selector.groups,
                  score_signs=score_bits, score_tile_unused_positions=32-score_bits, score_guard_token=513,
                  transition_cases_in_bipolar_domain=writer_cases, slots=slots,
                  data_prune_iterations=solves, fixed_phase_kept_writes=sum(x.bit_count() for x in kept),
                  maximum_retained_writes_single_phase=max(x.bit_count() for x in kept),
                  conditional_ideal_fixed_clique=lower["slots"], fixed_emission_global_colors=ncolor,
                  prior_clique_vertices_recovered=(new_clique & old_clique).bit_count(),
                  verified_clique=lower,
                  ideal_peak=max(x.bit_count() for x in ideal[0]+ideal[1]),
                  fixed_peak=max(x.bit_count() for x in fixed_in+fixed_out+kept), role_budget=rom,
                  full_wrom_roles_with_fixed_emission=floor_no_rom+2*ncolor,
                  typed_ports=dict(write_actions=sorted(stores), read_actions=sorted(reads),
                    retire_actions=sorted(releases), qkv_work_source_counts={name: len(lanes) for name, lanes in qkv.items()},
                    existing_read_write_address_bits=33, existing_word_bits=32, existing_position_handle_bits=24,
                    existing_hit_bits=1, additional_private_version_lanes=len(policy),
                    additional_fixed_hit_test_egress=int(bool(policy)), added_qkvo_coefficients=changed,
                    old_handle_taint_states_checked=prove_old_handle_cold(c, ids, states, edges, releases),
                    version_zero_seed_records_required=len(policy),
                    additional_retained_records_peak_success=len(policy)+1 if policy else 0))
    return result, (c, ids, names, codes, groups, selector, g, u, data, phases, ordinary, writer, kept,
                    colors, ingress, raw, private, scratch, tensors, floor_no_rom)


def physical_witness(details):
    c, _, _, _, _, selector, _, _, _, _, _, _, _, colors, ingress, raw, private, scratch, _, floor = details
    # Build a single oversized physical witness, with original raw field pins
    # and all 257 WROM projections. Half/quarter figures are separate budgets.
    ncolors = len(set(colors.values()))
    width = (floor + 2 * ncolors + 511) // 512 * 512
    fixed_raw = raw - private
    host_only = (ingress - raw) | set(c.dot_payload_ingress)
    score_start = next(start for start in reversed(range(0, width, 32))
                       if not set(range(start, start + 32)) & (fixed_raw | host_only))
    score_tile = set(range(score_start, score_start + 32))
    controls = [*range(compact.CLASS, compact.CLASS+len(selector.class_masks)),
                *range(grid.GROUP, grid.GROUP+selector.groups), *range(grid.MEMBER, grid.MEMBER+32), grid.REFERENCE]
    assert len(controls) == selector.input_roles
    for start in reversed(range(0, width, 128)):
        free = sorted(set(range(start, start + 128)) - fixed_raw - host_only - score_tile)
        if len(free) >= len(controls):
            embedding = dict(zip(controls, free[:len(controls)], strict=True))
            break
    else:
        raise ValueError("No packed input tile in oversized allocation")
    occupied = fixed_raw | score_tile | set(embedding.values())
    flex = sorted(set(range(width)) - occupied)
    raw_colors = [lane for lane in flex if lane not in host_only][:ncolors]
    assert len(raw_colors) == ncolors
    free = iter(sorted(set(flex) - set(raw_colors)))
    fixed_work = {lane: next(free) for lane in sorted(ingress - private)}
    work_colors = [next(free) for _ in range(ncolors)]
    scratch_map = {lane: next(free) for lane in sorted(scratch)}
    side = [next(free) for _ in c.dot_payload_ingress]
    outputs = fixed_raw | set(raw_colors) | score_tile
    inputs = set(fixed_work.values()) | set(work_colors) | set(scratch_map.values()) | set(side)
    assert not outputs & (inputs | host_only)
    assert not set(embedding.values()) & (inputs | outputs)
    assert len(outputs | inputs | set(embedding.values())) == floor+2*ncolors
    return dict(width=width, raw={lane: raw_colors[col] for lane, col in colors.items()},
                work={**fixed_work, **{lane: work_colors[col] for lane, col in colors.items()}},
                scratch=scratch_map, side=side, input=embedding, score_start=score_start,
                active_roles_with_score_padding=floor+2*ncolors)


def finite_packs(details, placement, score_signs):
    from export_tensors import q4_head
    _, ids, names, _, _, selector, *_ = details
    encoded = sha256()
    mask = placement["input"]
    for code in range(len(names)):
        terms = selector.embedding(code)
        assert len(terms) == 4 and sum(v*v for _, v in terms) == 1
        tile = bytearray(struct.pack("<e", 0.5) + bytes([0x55]) * 32)
        positions = {mask[lane] % 128 for lane, value in terms if value == .5}
        assert len(positions) == 4 and len({mask[lane] // 128 for lane, _ in terms}) == 1
        for slot in positions:
            tile[2 + slot // 4] ^= 3 << (2 * (slot % 4))
        assert sum(x != 0 for x in compact.structured.decode_tiles(tile + tile)[:128]) == 4
        encoded.update(tile)
    weights = {(code, bit): 1 if code & (1 << bit) else -1
               for code in range(len(names)) for bit in range(score_signs)}
    packed, tiles = q4_head(weights, (len(names)+1, 32))
    assert tiles == len(names)
    for code, cells in enumerate(packed[:, 0]):
        expected = b"\x00\x34" if code < tiles else b"\x00\x00"
        assert cells[:2].tobytes() == expected
        for i in range(32):
            value = (int(cells[2+i % 16]) >> (4 if i >= 16 else 0)) & 15
            assert value == 8 + 4 * weights.get((code, i), 0)
    return dict(pq2_tile_sha256=encoded.hexdigest(), q4_tile_sha256=sha256(packed.tobytes()).hexdigest(),
                head_token_map_sha256=sha256(b"".join(struct.pack("<H", ids[phase]) for phase in names)
                    + struct.pack("<H", 513)).hexdigest(), packed_guard_zero=True)


def regroup_rows(details, place):
    (_, _ids, _, codes, groups, selector, gates, ups, downs, phases,
     dot_start, writer, kept, _colors, _ingress, _raw, private, _, _, _) = details
    phase_masks = defaultdict(int)
    for phase, mask in enumerate(kept):
        for lane in live.members(mask):
            phase_masks[lane] |= 1 << phase
    reachable_mask = live.mask(codes)
    rows = {}
    counts = Counter()

    def in_lane(lane):
        if s.WORK <= lane < s.ACTION:
            return place["work"][lane-s.WORK]
        return place["scratch"].get(lane, lane)

    def out_lane(lane):
        if lane >= compact.NEXT:
            return place["score_start"] + lane - compact.NEXT
        return place["raw"][lane] if lane in private else lane

    def rewrite_terms(items):
        result = tuple(sorted((in_lane(lane), value) for lane, value in items))
        assert len({lane for lane, _ in result}) == len(result)
        return result

    def emit(terms, bias, wanted, out):
        for select in selector.cover(wanted):
            g, u = selector.coefficients(select, bias, grid.penalty_for(terms, bias, 4))
            compact.add_row(rows, (tuple(sorted((*terms, *((place["input"][k], v) for k, v in g)))),
                                   tuple(sorted((place["input"][k], v) for k, v in u))), out)
            counts["expanded_grid_fragments"] += 1
        counts["source_fragments"] += 1

    for i, (g, u, d, active) in enumerate(zip(gates, ups, downs, phases, strict=True)):
        mask = live.mask(active) & reachable_mask
        if not mask or not d:
            continue
        fragments = defaultdict(dict)
        for lane, value in d.items():
            keep = mask & phase_masks[lane] if lane in private else mask
            if keep:
                destination = out_lane(lane)
                assert destination not in fragments[keep]
                fragments[keep][destination] = value
        for keep, outputs in fragments.items():
            out = compact.sparse(outputs)
            if i >= dot_start:
                assert keep == mask and not (d.keys() & private)
                terms = []
                for source in (g, u):
                    mapped = dict(rewrite_terms((col, v) for col, v in source.items() if not s.ACTION <= col < s.DISPATCH))
                    for group, members in enumerate(groups):
                        values = {source.get(s.ACTION+phase, 0) for phase in members}
                        assert len(values) == 1
                        if value := values.pop():
                            mapped[place["input"][compact.CLASS+group]] = 2*value
                    terms.append(compact.sparse(mapped))
                compact.add_row(rows, tuple(terms), out)
                counts["protected_dot_fragments"] += 1
            else:
                bias, = {g[s.ACTION+phase] for phase in live.members(keep)}
                assert bias in (-4, 4) and all(u[s.ACTION+phase] == 16 for phase in live.members(keep))
                terms = rewrite_terms((col, value) for col, value in g.items() if not s.ACTION <= col < s.DISPATCH)
                emit(terms, bias, live.mask(codes[phase] for phase in live.members(keep)), out)
    for (terms, bias, wanted), outputs in writer:
        emit(rewrite_terms(terms), bias, wanted, tuple((out_lane(lane), weight) for lane, weight in outputs))
    result = {**counts, **compact.row_summary(rows)}
    result["activation_decisions_checked"] = selector.count * counts["source_fragments"]
    result["storage"] = grid.storage(result["rows"])
    return result


def main():
    prior, original = ret.main()
    _before, (_, _, _, _, _, total), ports, gates, _ = full.build()
    cut = total - len(compact.structured.dot_plane.compile_ff(False, True).g) - 4 * len(gates)
    comparison = {}
    for name, policy in POLICIES.items():
        case, details = size_case(policy, original, prior, ports, gates, cut)
        placement = physical_witness(details)
        case["full_wrom_physical_witness"] = dict(width=placement["width"],
            physical_roles=placement["active_roles_with_score_padding"],
            score_q4_tile_start=placement["score_start"],
            input_pq2_tile_start=min(placement["input"].values())//128*128,
            raw_private_map_entries=len(placement["raw"]),
            exact_coordinate_map_sha256=sha256(json.dumps(placement, sort_keys=True).encode("ascii")).hexdigest(),
            **finite_packs(details, placement, case["score_signs"]))
        case["grouped_ffn"] = regroup_rows(details, placement)
        comparison[name] = case
    return dict(source_sha256={path.name: sha256(path.read_bytes()).hexdigest() for path in (
                    Path(__file__), Path(ret.__file__), Path(grid.__file__), Path(full.__file__))},
                scope="Symbolic production-derived transfer rewrite with all unproved branch outcomes.",
                prior_clique=504, clean_start_opcode_refinement_clique=501,
                original_pair_limits={"half": 411, "quarter": 443},
                rejected_vector_candidate="EQUAL32 has zero live lanes at PREFIX_1/VX_VRED; VX_RED_READ rewrites it before each next byte.",
                policy_cases=comparison)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
