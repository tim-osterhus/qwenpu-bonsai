"""Compile and verify sparse action selectors without opening a model file."""

from collections import Counter, defaultdict, deque
from functools import cache
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "selector"), str(HERE.parents[1] / "audit")]
import phase_liveness as live
import structured_action_audit as structured

full, s = live.full, live.full.s
CLASS, ID_BIT, NEXT = 10000, 10018, 10112
BIT_COUNT = 9


def sparse(values):
    return tuple(sorted((lane, v) for lane, v in values.items() if v))


def add_row(rows, key, down):
    outputs = rows.setdefault(key, {})
    for dest, weight in down:
        outputs[dest] = outputs.get(dest, 0) + weight


def source():
    circuit, (ids, scores, g, u, d, total), ports, gates, _ = full.build()
    ordinary = total - len(structured.dot_plane.compile_ff(False, True).g) - 4 * len(gates)
    g, u, d = (live.sparse_rows(g, total), live.sparse_rows(u, total),
               live.sparse_rows(d, total, transpose=True))
    raw_reachable = {s.SEED_MISS, s.SEED_WORD, s.START, s.FETCH}
    pending = deque(sorted(raw_reachable))
    while pending:
        for target in circuit.routes[pending.popleft()]:
            if target not in raw_reachable:
                raw_reachable.add(target)
                pending.append(target)
    reachable = [phase for phase in ids if phase in raw_reachable]
    codes = {ids[phase]: i for i, phase in enumerate(reachable)}
    q, k, v, _ = full.attention(circuit, ids, ports, gates)
    matrices = [q.items(), k.items(), v.items()]
    matrices += [[((i, lane), weight) for i in range(ordinary, total) for lane, weight in rows[i].items()]
                 for rows in (g, u)]
    groups = structured.classes([structured.integer_rows(matrix, s.ACTION, codes) for matrix in matrices], codes)
    assert len(codes) == 510 and len(groups) == 18 and ordinary == 18040 and total == 19952

    ingress, raw, scratch, _, _, priv = live.inventory(circuit, ids, ports, gates, d)
    dest, defaults, reads, phase_rows, _, _ = live.summarize_rows(circuit, ids, g, u, d, priv, scratch)
    dispatch, dispatch_weights, _ = full.coalesce_dispatch(circuit, ids, scores)
    incoming, _, emitted = live.solve(ids, circuit, dest, defaults, reads, False)
    colors = live.color(incoming + emitted)
    allocation = live.layout(3584, circuit, ids, raw, ingress, scratch, priv, colors, len(dispatch_weights))
    dispatch = {lane: allocation["dispatch"][color] for lane, color in dispatch.items()}
    assert len(set(colors.values())) == 556 and len(dispatch_weights) == 63 and allocation["live_lanes"] == 3454
    return (circuit, ids, reachable, codes, groups, ordinary, g, u, d,
            phase_rows, emitted, priv, allocation, dispatch)


def remap_source(rows):
    _, ids, _, codes, groups, ordinary, gate, up, down, phases, emitted, private, layout, dispatch = rows
    cls_of = {action: i for i, group in enumerate(groups) for action in group}
    class_masks = [live.mask(codes[action] for action in group) for group in groups]
    phase_masks = defaultdict(int)
    for phase, lanes in enumerate(emitted):
        for lane in live.members(lanes):
            phase_masks[lane] |= 1 << phase

    def input_lane(lane):
        if s.WORK <= lane < s.ACTION:
            return layout["projected"][lane - s.WORK]
        return layout["attn_scratch"].get(lane, lane)

    def output_lane(lane):
        return layout["raw"][lane] if lane in private else dispatch.get(lane, lane)

    def action_lane(lane):
        return s.ACTION <= lane < s.DISPATCH

    legacy_all, legacy_live, dot = {}, {}, {}
    data, legacy_predicate = [], []
    for i, (g, u, d, phase_ids) in enumerate(zip(gate, up, down, phases, strict=True)):
        mask = live.mask(phase_ids)
        fragments = defaultdict(dict)
        for lane, coef in d.items():
            retained = mask & phase_masks[lane] if lane in private else mask
            if retained:
                fragments[retained][lane] = coef
        for retained, outputs in fragments.items():
            active = list(live.members(retained))
            active_live = [phase for phase in active if phase in codes]
            # Reproduce the prior concrete liveness-witness count before the rewrite.
            for actions, target in ((set(active), legacy_all), (set(active_live), legacy_live)):
                if not actions:
                    continue
                terms = []
                for weights in (g, u):
                    terms.append(sparse({
                        (layout["action"][lane - s.ACTION] if action_lane(lane) else input_lane(lane)): w
                        for lane, w in weights.items() if not action_lane(lane) or lane - s.ACTION in actions
                    }))
                add_row(target, tuple(terms), ((output_lane(lane), w) for lane, w in outputs.items()))
            if not active_live:
                continue
            if i >= ordinary:
                terms = []
                for weights in (g, u):
                    mapped = {input_lane(lane): w for lane, w in weights.items() if not action_lane(lane)}
                    for group in groups:
                        vals = {weights.get(s.ACTION + action, 0) for action in group}
                        assert len(vals) == 1 and all(cls_of[action] == cls_of[group[0]] for action in group)
                        if value := vals.pop():
                            mapped[CLASS + cls_of[group[0]]] = 2 * value
                    terms.append(sparse(mapped))
                assert retained == mask and not any(lane >= s.DISPATCH for lane in outputs)
                add_row(dot, tuple(terms), ((output_lane(lane), w) for lane, w in outputs.items()))
                continue
            assert set(u.values()) == {16} and all(s.ACTION <= lane < s.DISPATCH for lane in u)
            biases = {g[s.ACTION + action] for action in active_live}
            assert len(biases) == 1 and next(iter(biases)) in (-4, 4)
            terms = sparse({input_lane(lane): w for lane, w in g.items() if not action_lane(lane)})
            key = (terms, biases.pop(), live.mask(codes[action] for action in active_live))
            data_outputs = tuple((output_lane(lane), w) for lane, w in outputs.items() if lane < s.DISPATCH)
            pred_outputs = tuple((output_lane(lane), w) for lane, w in outputs.items() if lane >= s.DISPATCH)
            if data_outputs:
                data.append((key, data_outputs))
            if pred_outputs:
                legacy_predicate.append((key, pred_outputs))
    result = {
        "all_513_legacy_liveness_rows": sum(any(d.values()) for d in legacy_all.values()),
        "reachable_legacy_liveness_rows": sum(any(d.values()) for d in legacy_live.values()),
        "reachable_legacy_data_fragments": len(data),
        "reachable_legacy_branch_predicate_fragments": len(legacy_predicate),
        "protected_dot_rows": sum(any(d.values()) for d in dot.values()),
    }
    assert result["all_513_legacy_liveness_rows"] == 18416, result
    return data, legacy_predicate, dot, class_masks, input_lane, result


def transition_fragments(circuit, ids, reachable, class_masks, input_lane, emit_classes):
    codes = {phase: i for i, phase in enumerate(reachable)}

    def code(phase):
        i = codes[phase]
        bits = tuple(1 if i & (1 << bit) else -1 for bit in range(BIT_COUNT))
        if not emit_classes:
            return bits
        classes = tuple(0.5 if mask & (1 << i) else 0 for mask in class_masks)
        return (0.5 * bits[0], *(0.25 * v for v in bits[1:]), *classes)

    fanout = defaultdict(dict)
    transitions = Counter()
    overlapping_flags_example = None
    head_codes = [tuple(1 if i & (1 << bit) else -1 for bit in range(BIT_COUNT))
                  for i in range(len(reachable))]

    @cache
    def score_signs(signs):
        # Independent symbolic version of the scoring convention used by the
        # separate Q4_0 fixture. Include its zero guard.
        logits = [sum(value * weight for value, weight in zip(signs, row, strict=True)) for row in head_codes]
        logits.append(0)
        winner, runner = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)[:2]
        return winner, logits[winner] - logits[runner]

    for phase in reachable:
        routes = circuit.routes[phase]
        defaults = [target for target, flag in routes.items() if flag is None]
        assert len(defaults) <= 1
        fallback = defaults[0] if defaults else min(routes, key=ids.__getitem__)
        base = code(fallback)
        constant = list(base)
        changes = {}
        for target, flag in routes.items():
            if target == fallback:
                continue
            assert flag is not None
            expr = circuit.effects[phase].get(abs(flag), s.buffer(s.WORK + abs(flag)))
            delta = tuple((to - current) / 2 for to, current in zip(code(target), base, strict=True))
            changes[target] = delta
            for bit, value in enumerate(delta):
                constant[bit] += value
                if value:
                    for factor, terms in expr:
                        lane = (NEXT + bit, terms)
                        local = fanout[lane]
                        local[phase] = local.get(phase, 0) + value * factor * (1 if flag > 0 else -1)
        for bit, value in enumerate(constant):
            if value:
                local = fanout[NEXT + bit, ()]
                local[phase] = local.get(phase, 0) + value
        for winner in routes:
            # Exact arithmetic on signed legacy predicates. On this domain only the
            # winner may contribute a positive non-fallback predicate.
            result = tuple(c + sum(delta[i] * (1 if target == winner else -1)
                                   for target, delta in changes.items())
                           for i, c in enumerate(constant))
            assert result == code(winner)
            signs = tuple(1 if value > 0 else -1 for value in result[:BIT_COUNT])
            assert score_signs(signs) == (codes[winner], 2)
            transitions["signed_route_cases"] += 1
        if phase == "CLASSIFY" and not emit_classes:
            # Two abstract positive flags are deliberately outside the domain.
            # This example does not claim that the pinned producer can emit them.
            targets = sorted(changes, key=ids.__getitem__)
            for i, left in enumerate(targets):
                for right in targets[i+1:]:
                    mixed = tuple(b + 2 * (l + r) for b, l, r in zip(base, changes[left], changes[right], strict=True))
                    if mixed != code(left):
                        overlapping_flags_example = dict(source=phase, positive_routes=[left, right],
                            original_winner_for_equal_margins=left, expected_signs=list(code(left)),
                            emitted_values=list(mixed))
                        break
                if overlapping_flags_example:
                    break
        transitions["unconditional_phases" if len(routes) == 1 and defaults else "conditional_phases"] += 1
        if len(routes) > 2:
            transitions["phases_with_more_than_two_routes"] += 1
            transitions["max_routes_from_one_phase"] = max(transitions["max_routes_from_one_phase"], len(routes))
    groups = defaultdict(dict)
    for (destination, terms), phases in fanout.items():
        by_weight = defaultdict(int)
        for phase, weight in phases.items():
            if weight:
                by_weight[weight] |= 1 << codes[phase]
        for weight, mask in by_weight.items():
            vars = ((4, weight), (-4, -weight))
            for (bias, down) in vars:
                term_weights = defaultdict(float)
                for lane, factor in terms:
                    term_weights[input_lane(lane)] += 512 * factor
                nonlin = sparse(term_weights)
                for gate, coef in ((nonlin, down), ((), -down)) if terms else (((), down),):
                    key = gate, bias, mask
                    outputs = groups[key]
                    outputs[destination] = outputs.get(destination, 0) + coef
    assert len(reachable) == 510 and transitions["signed_route_cases"] == 862
    return ([(key, sparse(out)) for key, out in groups.items() if any(out.values())],
            dict(transitions), overlapping_flags_example)


class Selector:
    """A union of whole protected classes, followed by a disjoint ID cover."""

    def __init__(self, class_masks, count):
        self.class_masks = class_masks
        self.count = count
        self.full = (1 << count) - 1
        self.id_code = [(2 if i & 1 else -2, *(1 if i & (1 << b) else -1
                         for b in range(1, BIT_COUNT))) for i in range(count)]
        cube_masks = defaultdict(int)
        for i in range(count):
            for care in range(1 << BIT_COUNT):
                cube_masks[care, i & care] |= 1 << i
        self.cubes = sorted(((care, want, mask) for (care, want), mask in cube_masks.items()),
                            key=lambda item: (-item[2].bit_count(), item[0].bit_count(), item[:2]))

    @cache
    def decomposition(self, mask):
        classes = tuple(i for i, bits in enumerate(self.class_masks) if bits & mask == bits)
        rest = mask
        for i in classes:
            rest ^= self.class_masks[i]
        return classes, rest

    @cache
    def cube_cover(self, remaining):
        if not remaining:
            return ()
        result = []
        for care, want, bits in self.cubes:
            if bits & remaining == bits:
                result.append(("bits", (care, want)))
                remaining ^= bits
                if not remaining:
                    break
        return tuple(result)

    @cache
    def cover(self, mask, mode):
        classes, remaining = self.decomposition(mask)
        direct = ((("class", classes),) if classes else ()) + self.cube_cover(remaining)
        choices = [tuple((control, 1) for control in direct)]
        if mode == "signed":
            choices.append(tuple((control, 1) for control in self.cube_cover(mask)))
            majority = [
                tuple(range(len(self.class_masks))),
                tuple(i for i, bits in enumerate(self.class_masks) if (bits & mask).bit_count() * 2 > bits.bit_count()),
            ]
            for classes in majority:
                if not classes:
                    continue
                base = live.union_bits(self.class_masks[i] for i in classes)
                assert base & self.full == base
                choices.append(((("class", classes), 1),)
                    + tuple((control, 1) for control in self.cube_cover(mask & ~base))
                    + tuple((control, -1) for control in self.cube_cover(base & ~mask)))
        return min(choices, key=len)

    @cache
    def evaluate(self, control):
        """Decode synthesized control coefficients on all source embeddings."""
        kind, spec = control
        if kind == "plain_minterm":
            control = ("bits", ((1 << BIT_COUNT) - 1, spec))
            kind, spec = control
        g, u = self.coefficients(control, 0, 2)
        selected, minimum_off = 0, None
        for i, bits in enumerate(self.id_code):
            cls = next(j for j, mask in enumerate(self.class_masks) if mask & (1 << i))
            code = {CLASS + cls: 2, **{ID_BIT + j: value for j, value in enumerate(bits)}}
            up = sum(code.get(lane, 0) * coef for lane, coef in u)
            g_shift = sum(code.get(lane, 0) * coef for lane, coef in g)
            if up == 64 and g_shift == 0:
                selected |= 1 << i
            elif kind == "class":
                assert up == 0 and g_shift == 0
            else:
                assert up == 64 and g_shift <= -8 and g_shift % 8 == 0
                nearest_mismatch = -g_shift // 8
                minimum_off = nearest_mismatch if minimum_off is None else min(minimum_off, nearest_mismatch)
        return selected, minimum_off

    def coefficients(self, control, bias, penalty):
        kind, spec = control
        if kind == "class":
            return (tuple((CLASS + c, 2 * bias) for c in spec if bias),
                    tuple((CLASS + c, 32) for c in spec))
        care, want = spec
        gate = [(CLASS + c, 2 * bias - penalty * care.bit_count()) for c in range(len(self.class_masks))]
        gate.extend((ID_BIT + b, (1 if want & (1 << b) else -1) * penalty * (1 if b == 0 else 2))
                    for b in live.members(care))
        return tuple(gate), tuple((CLASS + c, 32) for c in range(len(self.class_masks)))


def row_summary(rows):
    sizes = Counter()
    digest = sha256()
    for gate, up in sorted(rows):
        outputs = sparse(rows[gate, up])
        if not outputs:
            continue
        # Order rows numerically and hash type-stable encodings. Equal float
        # and integer weights can otherwise acquire different printable forms.
        digest.update(struct.pack("<3H", len(gate), len(up), len(outputs)))
        packed = bytearray()
        for lane, value in gate:
            word = struct.pack("<f", value)
            assert struct.unpack("<f", word)[0] == value
            packed.extend(struct.pack("<H", lane) + word)
        for lane, value in up:
            word = struct.pack("<f", value)
            assert struct.unpack("<I", word)[0] & 65535 == 0
            packed.extend(struct.pack("<H", lane) + word[2:])
        for lane, value in outputs:
            word = struct.pack("<e", value)
            assert struct.unpack("<e", word)[0] == value
            packed.extend(struct.pack("<H", lane) + word)
        digest.update(packed)
        sizes.update(rows=1, gate_nonzeros=len(gate), up_nonzeros=len(up), down_nonzeros=len(outputs))
    return {**sizes, "coefficient_stream_sha256": digest.hexdigest()}


def compile_cover(fragments, dot, selector, mode):
    rows = {key: dict(down) for key, down in dot.items()}
    statistics = Counter()
    controls = set()
    for (terms, bias, wanted), outputs in fragments:
        # Compared with |work[i]| <= 8*a, a one-bit mismatch leaves >4096*a
        # of negative headroom. Actual amplitudes and finite stock SiLU are external checks.
        bound = 8 * sum(abs(v) for _, v in terms) + 4096 + abs(bias)
        penalty = 1 << int(bound).bit_length()
        admitted = rejected = 0
        for control, sign in selector.cover(wanted, mode):
            selected, minimum_off = selector.evaluate(control)
            assert selected
            if sign > 0:
                assert not admitted & selected
                admitted |= selected
            else:
                assert not rejected & selected
                rejected |= selected
                statistics["subtractive_fragments"] += 1
            gate, up = selector.coefficients(control, bias, penalty)
            if minimum_off is not None:
                assert penalty * minimum_off > bound
            add_row(rows, (terms + gate, up), outputs if sign > 0 else ((lane, -w) for lane, w in outputs))
            controls.add(control)
            statistics["row_fragments"] += 1
            statistics["direct_class_fragments" if control[0] == "class" else "hamming_fragments"] += 1
        assert rejected & admitted == rejected and admitted ^ rejected == wanted
        statistics["activation_decisions_checked"] += selector.count
    return rows, {**statistics, "unique_selectors": len(controls), "distinct_source_masks": len({key[2] for key, _ in fragments})}


def minterm_count(fragments, dot, selector, retain_classes):
    """Materialize and deduplicate actual down fanout one action at a time."""
    by_action = [[] for _ in range(selector.count)]
    classes = {key: dict(down) for key, down in dot.items()}
    copies = 0
    for (terms, bias, mask), outputs in fragments:
        if retain_classes:
            group, mask = selector.decomposition(mask)
            if group:
                g, u = selector.coefficients(("class", group), bias, 0)
                add_row(classes, (terms + g, u), outputs)
        for i in live.members(mask):
            by_action[i].append((terms, bias, outputs))
            copies += 1
    rows = sum(any(d.values()) for d in classes.values())
    duplicate_merges = zeros = 0
    for i, fragments in enumerate(by_action):
        control = ("plain_minterm", i)
        assert selector.evaluate(control)[0] == 1 << i
        merged = {}
        for terms, bias, outputs in fragments:
            # Minterm ID and up coefficients are fixed within this action.
            # The penalty depends only on terms and |bias|.
            add_row(merged, (terms, bias), outputs)
        surviving = sum(any(d.values()) for d in merged.values())
        rows += surviving
        duplicate_merges += len(fragments) - len(merged)
        zeros += len(merged) - surviving
    return dict(minterm_fragments=copies, deduplicated_rows=rows,
                minterm_duplicate_merges=duplicate_merges, minterm_cancelled_rows=zeros,
                protected_classes_reused=retain_classes)


def geometry(rows, language_rows, alignment, ffn_weights, width):
    shared = (max(rows, language_rows) + alignment - 1) // alignment * alignment
    grow = shared - language_rows
    padding = {name: grow * n for name, n in ffn_weights.items()}
    # F32 gate + BF16 up + F16 down, matching the pinned 9B MTP tensors.
    circuit = shared * width * 8
    return dict(ordinary_and_dot_rows=rows, shared_h512_rows=shared, circuit_unused_rows=shared-rows,
                language_padding_rows_per_matrix=grow, language_padding_bytes=padding,
                language_padding_total_bytes=sum(padding.values()), circuit_ffn_bytes=circuit,
                language_padding_plus_circuit_ffn_bytes=sum(padding.values()) + circuit)


def audit():
    source_rows = source()
    circuit, ids, reachable, codes, groups, ordinary, g, *_ = source_rows
    data, pred, dot, class_masks, input_lane, baseline = remap_source(source_rows)
    selector = Selector(class_masks, len(codes))
    turn, cases, overlapping = transition_fragments(circuit, ids, reachable, class_masks, input_lane, False)
    all_turn, more_cases, _ = transition_fragments(circuit, ids, reachable, class_masks, input_lane, True)
    assert cases == more_cases and overlapping
    modes = {}
    for name, fragments, method in (("data_only", data, "disjoint"), ("legacy_predicate_output", data + pred, "disjoint"),
                                    ("nine_sign_next_code", data + turn, "disjoint"),
                                    ("all_27_next_code", data + all_turn, "disjoint"),
                                    ("signed_complement_nine_sign_next_code", data + turn, "signed")):
        rows, count = compile_cover(fragments, dot, selector, method)
        modes[name] = {**count, **row_summary(rows)}
    direct = {name: minterm_count(data + turn, dot, selector, keep)
              for name, keep in (("literal_minterm", False), ("class_plus_minterm", True))}
    plan_root = HERE.parents[2]
    headers = json.loads((plan_root / "headers.json").read_text(encoding="utf-8"))
    header = headers["remote"]["Ternary-Bonsai-2-4B-PQ2_0.gguf"]
    metadata = header["metadata"]
    width, language, align = (metadata[key] for key in
                             ("qwen35.embedding_length", "qwen35.feed_forward_length", "prism.hadamard.block_size"))
    ffn_weights = Counter()
    for name, desc in header["tensors"].items():
        if name.endswith((".ffn_gate.weight", ".ffn_up.weight", ".ffn_down.weight")):
            ffn_kind = name.split(".")[-2].removeprefix("ffn_")
            assert desc["type_id"] == 142 and sorted(desc["dims"]) == [width, language]
            ffn_weights[ffn_kind] += width // 128 * 34
    assert ffn_weights == {"gate": 21760, "up": 21760, "down": 21760}
    sizes = {"original_9b_live_rows_at_4b_width": len(g),
             "legacy_liveness": baseline["all_513_legacy_liveness_rows"],
             "legacy_reachable_liveness": baseline["reachable_legacy_liveness_rows"]}
    sizes.update({key: value["rows"] for key, value in modes.items()})
    sizes.update({key: value["deduplicated_rows"] for key, value in direct.items()})
    geom = {name: geometry(val, language, align, ffn_weights, width) for name, val in sizes.items()}
    def lane_case(rom_saving=0, reserve_pq2=False, emitted_class=0):
        # Reserve the complete Q4_0 score tile. The PQ2 padding is token-only and
        # can be used by other tensors only when placement can prove exclusivity.
        fixed = 2342 - 513 - 63 + 27 + 32 + 229 * reserve_pq2 + emitted_class - rom_saving
        roles = fixed + 2 * 556
        return dict(roles=roles, deficit=roles-width, fixed_roles=fixed,
                    private_color_budget=(width-fixed)//2, rom_projected_lanes_reclaimed=rom_saving,
                    next_sign_lanes=9, next_score_padding=23, input_action_lanes=27,
                    input_pq2_tiles=2, input_pq2_padding_reserved=229 * reserve_pq2,
                    legacy_dispatch_removed=63, optional_next_class_lanes=emitted_class)
    return {
        "source_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in
                          (Path(__file__), full.HERE / "compile_full9b.py", Path(full.dot.__file__),
                           Path(full.vx.__file__), Path(live.__file__), Path(structured.__file__),
                           plan_root / "headers.json")},
        "source": dict(original_ids=len(ids), reachable_ids=len(codes), ordinary_rows=ordinary, dot_rows=len(g)-ordinary,
                       unreachable_ids=[phase for phase in ids if phase not in reachable],
                       protected_class_sizes=list(map(len, groups)), **baseline),
        "compact_cube_rewrites": modes,
        "minterm_comparisons_nine_sign_next_code": direct,
        "symbolic_transitions": {**cases, "invalid_two_positive_example": overlapping},
        "h512_pq2_geometry": geom,
        "lane_ledger": {
            "coded_input_legacy_dispatch_roles_before_padding": 3454 - 513 + 27,
            "coded_input_separate_nine_sign_score_tile": lane_case(),
            "same_reserving_both_complete_pq2_input_tiles": lane_case(reserve_pq2=True),
            "same_with_explicit_class_egress": lane_case(emitted_class=18),
            "conditional_two_half_wrom": {**lane_case(128), "extra_retained_cells": 4608,
                                         "extra_f16_kv_bytes": 4608 * 4096},
            "conditional_full_wrom_projection_sharing": lane_case(257),
        },
        "scope": {
            "proof": "Exact signed-input selector masks and matching affine gate offsets before rounding. Exact next code on signed legacy predicate cases.",
            "limitations": [
                "A zero inactive SiLU is the selected symbolic plateau convention. Stock finite activation, reduction and rounding parity have not been measured.",
                "Multi-positive routing, non-bipolar or zero predicates and real recurrence are outside the symbolic route proof.",
                "The egress is distinct from the old token residual. The old code is masked from feedback and selected token embedding regenerates its protected class.",
                "Lane counts price existing bank/liveness policy and complete score padding. There is no native-width coordinate witness.",
                "FFN cost uses the existing F32/BF16/F16 precision layout. No runtime cost, weights, checkpoints, KV transport, or trunk parity were tested.",
            ],
        },
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
