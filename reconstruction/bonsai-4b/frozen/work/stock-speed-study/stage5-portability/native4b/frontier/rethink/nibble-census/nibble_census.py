"""Emit and audit a CPU-only production sparse radix-16 shared adder rewrite."""

from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from random import Random
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public-lanes"))
import public_lanes as public

spill, ret, full, compact, live, s = public.spill, public.ret, public.full, public.compact, public.live, public.s
CORE = ("ADD_INIT", "PREFIX_1", "ADD_FINISH")
RETIRED = tuple(p for p in ret.ADD_CORE if p not in CORE)
GENERATES = s.bits(s.G, 8)
CARRIES = (s.CARRY, *s.bits(s.G + 8, 7), s.G + 31)
# PREP_SEQUENTIAL also uses P[3:32] for the separate VX_PC4_FINISH path.
OLD_SCRATCH = set(s.bits(s.P, 3) + s.bits(s.G, 31))
KEEP_SCRATCH = set(GENERATES + CARRIES[1:-1])


def expressions():
    initial, gather, finish = {}, {}, {}
    for nibble, gen in enumerate(GENERATES):
        offset = nibble * 4
        # Only the low bit+1 input digits can affect this output bit.
        # The bipolar sum is 2*(a+b) minus the maximum unsigned total.
        for bit in range(4):
            terms = tuple((s.WORK + operand + offset + digit, 1 << digit)
                          for operand in (s.A, s.B) for digit in range(bit + 1))
            step = 1 << bit
            maximum = 4 * step - 2
            thresholds = range(step, maximum + 1, step)
            baseline = () if bit else s.constant(-1)
            initial[s.X + offset + bit] = (*baseline, *(
                s.threshold(terms, maximum + 1 - 2 * threshold, 1 if i % 2 == 0 else -1)[0]
                for i, threshold in enumerate(thresholds)))
        initial[gen] = s.threshold(terms, -1)

    for end, dest in enumerate(CARRIES[1:], 1):
        # Each term selects the last lower generator (or the input carry)
        # followed by nibbles whose local four sum bits are all one.
        result = s.constant(end)
        for start, gen in enumerate((s.CARRY, *GENERATES[:end])):
            terms = ((s.WORK + gen, 1), *((lane, 1) for lane in
                     s.bits(s.WORK + s.X + 4 * start, 4 * (end - start))))
            result += s.match(terms)
        gather[dest] = result

    for nibble, carry in enumerate(CARRIES[:-1]):
        lower = ((s.WORK + carry, 1),)
        for bit in range(4):
            lane = s.WORK + s.X + 4 * nibble + bit
            finish[s.SUM + 4 * nibble + bit] = (
                *s.buffer(lane), *s.match((*lower, (lane, -1))), *s.match((*lower, (lane, 1)), -1))
            lower = (*lower, (lane, 1))
    return initial, gather, finish


def transplant(original, callers):
    c = deepcopy(original)
    obsolete = OLD_SCRATCH - KEEP_SCRATCH
    old_outputs = set(s.bits(s.X) + s.bits(s.P) + s.bits(s.G) + s.bits(s.SUM))
    for phase in ret.ADD_CORE:
        assert c.effects[phase].keys() <= old_outputs
        c.effects[phase].clear()
    c.persistent = tuple(lane for lane in c.persistent if lane not in obsolete)
    external_reads = {lane for effects in c.effects.values() for expr in effects.values()
                      for _, terms in expr for lane, _ in terms}
    assert not external_reads & {s.WORK + lane for lane in OLD_SCRATCH}
    assert {s.WORK + lane for lane in s.bits(s.P + 3, 29)} <= external_reads
    assert s.WORK + CARRIES[-1] in external_reads
    for phase, effects in zip(CORE, expressions(), strict=True):
        c.effects[phase].update(effects)
    assert c.routes[CORE[0]] == {CORE[1]: None}
    c.routes[CORE[1]] = {CORE[2]: None}
    for phase in RETIRED:
        c.routes[phase] = {"SET_ILLEGAL": None}
    assert set(c.phases) == set(original.phases)
    assert all(c.effects[phase] == original.effects[phase] and c.routes[phase] == original.routes[phase]
               for phase in c.phases if phase not in ret.ADD_CORE)
    assert all(c.effects[phase] == original.effects[phase] and c.routes[phase] == {CORE[0]: None}
               for phase in callers)
    assert c.routes[CORE[-1]] == original.routes[CORE[-1]]
    assert all(not (set(routes) & set(RETIRED)) for routes in c.routes.values())
    return c


def arithmetic_proof():
    initial, gather, finish = expressions()

    def word(lanes, val):
        return {s.WORK + lane: s.sign(val, bit) for bit, lane in enumerate(lanes)}

    def evaluate(expr, inputs):
        total = 0
        for factor, terms in expr:
            value = sum(inputs[lane] * coeff for lane, coeff in terms)
            assert not terms or value
            total += factor if value >= 0 else -factor
        assert total in (-1, 1)
        return total

    nibble_pairs = []
    for a, b in product(range(16), repeat=2):
        inputs = {**word(s.bits(s.A, 4), a), **word(s.bits(s.B, 4), b), s.WORK + s.ONE: 1}
        sums = {s.WORK + lane: evaluate(initial[lane], inputs) for lane in s.bits(s.X, 4) + GENERATES[:1]}
        assert all(sums[s.WORK + lane] == val for lane, val in
                   zip(s.bits(s.X, 4) + GENERATES[:1], (s.sign(a+b, bit) for bit in range(5)), strict=True))
        assert not (sums[s.WORK + GENERATES[0]] == 1 and all(sums[s.WORK + lane] == 1 for lane in s.bits(s.X, 4)))
        for carry in range(2):
            case = {**sums, s.WORK + s.CARRY: s.sign(carry, 0), s.WORK + s.ONE: 1}
            for bit in range(4):
                assert evaluate(finish[s.SUM + bit], case) == s.sign(a + b + carry, bit)
            assert evaluate(gather[CARRIES[1]], case) == s.sign(a + b + carry, 4)
        nibble_pairs.append(sums)

    # A nibble is either kill, propagate, or generate. Exercise every
    # eight-nibble sequence of these categories at both input carries.
    representatives = (nibble_pairs[0], nibble_pairs[15], nibble_pairs[31])
    assert [(case[s.WORK + GENERATES[0]] > 0, all(case[s.WORK + x] > 0 for x in s.bits(s.X, 4)))
            for case in representatives] == [(False, False), (False, True), (True, False)]

    def gather_and_check(choices, states, carry):
        inputs = {s.WORK + s.CARRY: s.sign(carry, 0), s.WORK + s.ONE: 1}
        for index, case in enumerate(states):
            for bit in range(4):
                inputs[s.WORK + s.X + index * 4 + bit] = case[s.WORK + s.X + bit]
            inputs[s.WORK + GENERATES[index]] = case[s.WORK + GENERATES[0]]
        for index, kind in enumerate(choices, 1):
            carry = kind == 2 or (kind == 1 and carry)
            inputs[s.WORK + CARRIES[index]] = got = evaluate(gather[CARRIES[index]], inputs)
            assert got == s.sign(carry, 0)
        return inputs

    for choices in product(range(3), repeat=8):
        states = [representatives[choice] for choice in choices]
        for carry in range(2):
            gather_and_check(choices, states, carry)
    random = Random(0xB12128)
    directed = (0, 1, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff, 0xaaaa_aaaa, 0x5555_5555,
                *(1 << bit for bit in range(32)), *((1 << bit) - 1 for bit in range(1, 32)))
    pairs = [*product(directed, repeat=2), *((random.getrandbits(32), random.getrandbits(32)) for _ in range(4096))]
    for a, b in pairs:
        totals = [((a >> n) & 15) + ((b >> n) & 15) for n in range(0, 32, 4)]
        choices = [2 if total >= 16 else int(total == 15) for total in totals]
        states = [nibble_pairs[((a >> n) & 15) * 16 + ((b >> n) & 15)] for n in range(0, 32, 4)]
        for carry in range(2):
            inputs = gather_and_check(choices, states, carry)
            expected = a + b + carry
            assert all(evaluate(finish[s.SUM + bit], inputs) == s.sign(expected, bit) for bit in range(32))
            assert inputs[s.WORK + CARRIES[-1]] == s.sign(expected, 32)
    return dict(local_nibble_cases=512, exhaustive_carry_network_cases=2 * 3**8, tested_full_words=len(pairs) * 2)


def compact_returns(original, ids, callers):
    c = deepcopy(original)
    flags = set(c.returns.values())
    pending = tuple(sorted(flags)[:9])
    codes = {phase: code for code, phase in enumerate(spill.reachable(c, ids))}
    c.persistent = tuple(lane for lane in c.persistent if lane not in flags or lane in pending)
    for phase, mode in callers.items():
        for flag in flags:
            del c.effects[phase][flag]
        for bit, lane in enumerate(pending):
            c.put(phase, lane, s.constant(s.sign(codes[f"AFTER_{mode}"], bit)))
        readback = sum((1 << bit) for bit, lane in enumerate(pending) if c.effects[phase][lane] == s.constant(1))
        assert tuple(codes)[readback] == f"AFTER_{mode}"
    assert all(not (effects.keys() & flags) for phase, effects in c.effects.items() if phase not in callers)
    return c, pending


def source_compile(c, ids, g, u, d, dot_start, ports, gates_dot, pending):
    active = spill.reachable(c, ids)
    codes = {ids[phase]: code for code, phase in enumerate(active)}
    tensors, groups, selector = spill.classify_grid(c, ids, codes, ports, gates_dot, dot_start, g, u, ())
    writer, cases = spill.next_fragments(c, ids, active, 9)
    if pending:
        finish = 1 << codes[ids[CORE[-1]]]
        writer = [((terms, bias, support & ~finish), outputs) for (terms, bias, support), outputs in writer
                  if support & ~finish]
        for bit, lane in enumerate(pending):
            terms = ((s.WORK + lane, 512),)
            dest = compact.NEXT + bit
            for gate, bias, coef in ((terms, 4, 1), (terms, -4, -1), ((), 4, -1), ((), -4, 1)):
                writer.append(((gate, bias, finish), ((dest, coef),)))
    ingress, raw, scratch, qkv, _, private = live.inventory(c, ids, ports, gates_dot, d)
    public_candidates = (raw & ingress) - set().union(*qkv.values()) - {
        lane for value in ports.values() for lane in ((value,) if isinstance(value, int) else value)}
    assert public_candidates <= set(c.persistent) and private <= public_candidates
    assert len(public_candidates - private) == 402 and selector.input_roles == 67
    return active, codes, tensors, groups, selector, writer, cases, ingress, raw, scratch, qkv, private, public_candidates


def analyze(c, ids, g, u, d, dot_start, ports, gates_dot, callers, host_public, pending=(), emit=False):
    active, codes, tensors, groups, selector, writer, cases, ingress, raw, scratch, qkv, private, public_candidates = (
        source_compile(c, ids, g, u, d, dot_start, ports, gates_dot, pending))
    candidates = public_candidates if host_public else private
    states, edges = spill.flow(c, ids, callers, {mode: f"AFTER_{mode}" for mode in c.returns})
    active_names = tuple(ids)
    dst, defaults, base, phases, _, _ = live.summarize_rows(c, ids, g, u, d, candidates, scratch)
    for (terms, _, wanted), _ in writer:
        reads = live.mask(col - s.WORK for col, val in terms if val and col - s.WORK in candidates)
        for code in live.members(wanted):
            base[ids[active[code]]] |= reads
    actual = ret.phase_writes(d, phases, candidates, len(ids))
    deps = ret.active_deps(dst, defaults)

    if host_public:
        edges = [() if active_names[p] in public.TERMINAL else links for (p, _), links in zip(states, edges, strict=True)]
        fields, read_tokens, tile, range_guard = public.contract(ids)
        host_masks, _ = public.host_needs(c, ids, states, edges, fields, read_tokens, tile, range_guard)
        candidate_mask = live.mask(candidates)
        host_masks = [mask & candidate_mask for mask in host_masks]
        ideal, ins, outs, kept, solves = public.close(states, edges, deps, base, actual, host_masks)
    else:
        ideal = ret.solve(states, edges, deps, base)
        ins, outs, kept, solves = ret.fixed_phase_pruning(states, edges, deps, base, actual, ideal)

    protected = [out | kept[p] for out, (p, _) in zip(outs, states, strict=True)]
    colors = live.color(ins + protected)
    ncolor = len(set(colors.values()))
    peak = ret.summary(states, ins, protected, active_names)
    print(f"{len(active)} actions, {'public' if host_public else 'private'}: {ncolor} colors, "
          f"{peak['input_peak']} input / {peak['output_peak']} output", file=sys.stderr, flush=True)
    assert all(len({colors[lane] for lane in live.members(mask)}) == mask.bit_count() for mask in ins + protected)
    fixed_floor = (len(raw - candidates) + len(ingress - candidates) + len(scratch)
                   + len(c.dot_payload_ingress) + selector.input_roles + 32)
    lower = ret.verified_fixed_map_clique(ins + protected)
    ret.certify_clique(lower, states, ins, protected, active_names)
    ideal_lower = ret.verified_fixed_map_clique(ideal[0] + ideal[1])
    ret.certify_clique(ideal_lower, states, *ideal, active_names)
    floor_budget = {name: dict(fixed_roles=fixed_floor-saving, color_limit=(2560-fixed_floor+saving)//2,
                                ideal_clique_min_roles=fixed_floor-saving+2*ideal_lower["slots"],
                                fixed_clique_min_roles=fixed_floor-saving+2*lower["slots"],
                                checked_colors_roles=fixed_floor-saving+2*ncolor)
                    for name, saving in (("full", 0), ("half_ledger", 128), ("quarter_conditional", 192))}
    result = dict(
        active_actions=len(active), context_states=len(states), edges=sum(map(len, edges)), candidate_pairs=len(candidates),
        fixed_floor=fixed_floor, attention_pins={name: len(lanes) for name, lanes in qkv.items()},
        selector_roles=selector.input_roles, protected_classes=len(groups), score_signs=9, route_cases=cases,
        closure_solves=solves, kept_phase_writes=sum(map(int.bit_count, kept)), fixed_emission_colors=ncolor,
        fixed_emission_peak=peak, ideal_peak=ret.summary(states, *ideal, active_names),
        fixed_clique=lower, ideal_clique=ideal_lower, width_budget=floor_budget,
        input_output_conflicts_checked=sum(mask.bit_count() * (mask.bit_count()-1) // 2 for mask in ins + protected),
    )
    if host_public:
        result["actual_emission_and_next_work_witness"] = public.emitted_witness(
            states, edges, active_names, kept, ins, outs, protected, dst, defaults, actual)
    details = (c, ids, active, codes, groups, selector, g, u, d, phases, dot_start, writer, kept,
               colors, ingress, raw, candidates, scratch, tensors, fixed_floor)
    if emit:
        placement = spill.physical_witness(details)
        result["full_wrom_packed_control"] = verify_placement(details, placement)
        result["grouped_ffn"] = spill.regroup_rows(details, placement)
        result["grouped_ffn"]["control_width_mtp_ffn_bytes"] = (
            placement["width"] * result["grouped_ffn"]["storage"]["shared_h512_rows"] * 8)
    return result, details, (dst, defaults, base, actual)


def verify_placement(details, place):
    (c, ids, active, _codes, groups, selector, _, _, _, _, _, _, _, _, ingress, raw, candidates, scratch,
     (q, k, v, o), floor) = details
    work, raw_map = place["work"], place["raw"]
    side = {full.dot.SIDE_PAYLOAD + i: physical for i, physical in enumerate(place["side"])}
    projected = {s.WORK + lane: physical for lane, physical in work.items()} | side
    fixed_raw = {lane: lane for lane in raw - candidates}
    assert set(work) <= ingress and set(raw_map) <= candidates
    assert set(place["scratch"]) == scratch
    eh = {(place["input"][lane], place["input"][lane]): 1 for lane in place["input"]}
    eh.update({(physical, place["width"] + raw_map.get(lane, lane)): 1 for lane, physical in work.items()})
    eh.update({(physical, place["width"] + lane): 1
               for lane, physical in zip(c.dot_payload_ingress, place["side"], strict=True)})
    assert len({i for i, _ in eh}) == len(eh) == len({j for _, j in eh})
    assert set(fixed_raw.values()).isdisjoint(set(projected.values()) | set(place["input"].values()))
    assert all(0 <= i < place["width"] and 0 <= j < 2 * place["width"] for i, j in eh)
    assert len(eh) == len(set(work.values())) + len(place["side"]) + selector.input_roles

    def attention_input(matrix):
        by_row = defaultdict(dict)
        mapped = {}
        for (row, col), weight in matrix.items():
            if s.ACTION <= col < s.DISPATCH:
                by_row[row][col-s.ACTION] = weight
            else:
                mapped[row, projected[col]] = weight
        assert len(mapped) + sum(map(len, by_row.values())) == len(matrix)
        for row, values in by_row.items():
            for group, ids_in_class in enumerate(groups):
                weights = {values.get(action, 0) for action in ids_in_class}
                assert len(weights) == 1
                if weight := weights.pop():
                    key = row, place["input"][compact.CLASS + group]
                    assert key not in mapped
                    mapped[key] = weight * 2
        return mapped

    o_phys = {(place["scratch"].get(dest, dest), col): weight for (dest, col), weight in o.items()}
    assert len(o_phys) == len(o) and {r for r, _ in o_phys} <= set(place["scratch"].values()) | set(fixed_raw.values())
    sparse = {"EH": eh, "Q": attention_input(q), "K": attention_input(k), "V": attention_input(v), "O": o_phys}
    control = spill.finite_packs(details, place, 9)
    assert len(active) == 506 and len(ids) == 513
    return dict(width=place["width"], roles=place["active_roles_with_score_padding"], fixed_floor=floor,
                score_start=place["score_start"], source_tile_start=min(place["input"].values())//128*128,
                raw_work_placement=place, **control,
                sparse_projection_sha256={kind: digest_matrix(matrix) for kind, matrix in sparse.items()},
                sparse_projection_coefficients={kind: len(matrix) for kind, matrix in sparse.items()},
                abi_raw_remap={lane: raw_map[lane] for lane in sorted(set(raw_map) & set().union(
                    *map(set, public.contract(ids)[0].values())))})


def digest_matrix(matrix):
    return sha256(json.dumps(sorted((i, j, value) for (i, j), value in matrix.items()),
                             separators=(",", ":")).encode("ascii")).hexdigest()


def changed_rows(details, dependencies):
    (c, ids, _active, codes, _groups, _selector, g, u, d, phases, dot_start, writer, kept, _colors, _, _, _, _, _, _) = details
    dst, defaults, base, actual = dependencies
    core_ids = {ids[phase]: phase for phase in CORE}
    core_mask = live.mask(core_ids)
    report = {phase: dict(token=ids[phase], routes=c.routes[phase], arithmetic_outputs={}, rows=[]) for phase in CORE}
    for p, phase in core_ids.items():
        effects = c.effects[phase]
        report[phase].update(fixed_emitted_mask_hex=format(kept[p], "x"), source_emitted_mask_hex=format(actual[p], "x"),
                             candidate_base_read_mask_hex=format(base[p], "x"),
                             candidate_refresh_read_mask_hex=format(defaults[p], "x"))
        for lane, expr in effects.items():
            assert kept[p] >> lane & 1
            report[phase]["arithmetic_outputs"][lane] = dict(expression=expr, dependency_mask_hex=format(dst[p][lane], "x"))
    for row, (gate, up, down, phase_ids) in enumerate(zip(g[:dot_start], u[:dot_start], d[:dot_start], phases[:dot_start], strict=True)):
        mask = live.mask(phase_ids)
        for p in live.members(mask & core_mask):
            phase = core_ids[p]
            touched = {lane: weight for lane, weight in down.items() if lane in c.effects[phase]}
            if touched:
                assert up[s.ACTION + p] == 16
                report[phase]["rows"].append(dict(row=row, source_action_mask_hex=format(mask, "x"),
                    gate_terms={col: val for col, val in gate.items() if not s.ACTION <= col < s.DISPATCH},
                    gate_action_bias=gate[s.ACTION + p], up_action=up[s.ACTION + p], arithmetic_down=touched))
    score_touched = live.mask(codes[p] for p in core_ids)
    score = [dict(terms=terms, bias=bias, score_input_code_mask_hex=format(support, "x"), outputs=outputs)
             for (terms, bias, support), outputs in writer if support & score_touched]
    return dict(arithmetic_phases=report, score_writer=score)


def main():
    baseline, original = ret.main()
    assert baseline["fixed_phase_pruning"]["fixed_map_colors_under_previous_no_duplicate_emits_remap"] == 555
    assert baseline["fixed_phase_pruning"]["candidate_phase_writes_kept"] == 212485
    print("Reproduced original 555 private colors / 212485 writes", file=sys.stderr, flush=True)
    old, ids = original[:2]
    _, (_, _, _, _, _, row_count), ports, gates_dot, _ = full.build()
    cut = row_count - len(compact.structured.dot_plane.compile_ff(False, True).g) - 4 * len(gates_dot)
    callers = baseline["callers"]
    assert len(callers) == 54
    old_public, _, _ = analyze(old, ids, *original[14:], cut, ports, gates_dot, callers, True)
    assert old_public["fixed_emission_colors"] == 897
    rewritten = transplant(old, callers)
    arithmetic = arithmetic_proof()
    new_ids, g, u, d, dot_start = spill.ordinary_compile(rewritten, original, cut)
    assert ids == new_ids
    assert full.attention(old, ids, ports, gates_dot) == full.attention(rewritten, ids, ports, gates_dot)
    assert set(spill.reachable(old, ids)) - set(spill.reachable(rewritten, ids)) == set(RETIRED)
    compact_circuit, pending = compact_returns(rewritten, ids, callers)
    variants = {}
    rows = {}
    for name, circuit, pending in (("radix16", rewritten, ()), ("radix16_compact_return", compact_circuit, pending)):
        if pending:
            new_ids, g, u, d, dot_start = spill.ordinary_compile(circuit, original, cut)
            assert ids == new_ids
        data = [{lane: weight for lane, weight in row.items() if lane < s.DISPATCH} for row in d]
        private, _, _ = analyze(circuit, ids, g, u, data, dot_start, ports, gates_dot, callers, False, pending, emit=True)
        guarded, details, deps = analyze(circuit, ids, g, u, data, dot_start, ports, gates_dot, callers, True, pending, emit=True)
        rows[name] = changed_rows(details, deps)
        variants[name] = dict(private=private, guarded_public=guarded, pending_return_word=list(pending),
                              source_ordinary_rows=dot_start, source_total_rows=len(d),
                              emitted_arithmetic_rows_per_phase={phase: len(phase_rows["rows"]) for phase, phase_rows
                                                                 in rows[name]["arithmetic_phases"].items()})
    rows_path = HERE / "sparse-rows.json"
    rows_path.write_text(json.dumps(rows, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    sources = (Path(__file__), Path(public.__file__), Path(spill.__file__), Path(ret.__file__), Path(full.__file__),
               Path(full.vx.__file__), Path(s.__file__), Path(live.__file__), Path(compact.__file__), Path(spill.grid.__file__),
               public.OBSERVER, public.PROBE, HERE.parent / "next-approaches.md", rows_path)
    return dict(arithmetic=arithmetic, action_tokens=ids, retired_actions=list(RETIRED), shared_add_callers=callers,
                live_control=dict(private_colors=555, private_kept_writes=212485, guarded_public=old_public),
                variants=variants,
                source_and_row_artifact_sha256={path.name: sha256(path.read_bytes()).hexdigest() for path in sources})


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
