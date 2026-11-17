"""Optional opcode-only refinement of the conservative ADD context graph."""

from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import return_context as returns

s, live = returns.s, returns.live


def decoder(c):
    flags = c.op_flag
    columns = {s.WORK + lane: 1 << i for i, lane in enumerate(flags.values())}
    assert len(flags) == len(columns) == 63 and set(flags.values()) <= set(c.persistent)
    one = s.WORK + s.ONE
    assert s.ONE in c.persistent
    patterns = {}
    for op, lane in flags.items():
        (factor, terms), = c.effects["DECODE"][lane]
        assert factor == 1 and len(terms) == len(dict(terms))
        pattern = dict(terms)
        bias = pattern.pop(one, 0)
        assert pattern and bias == 1 - len(pattern)
        assert all(s.WORK + s.RAW <= source < s.WORK + s.RAW + 32 and abs(want) == 1
                   for source, want in pattern.items())
        patterns[op] = pattern
    for a, b in combinations(patterns.values(), 2):
        assert any(a[lane] != b[lane] for lane in a.keys() & b.keys())

    writers = {}
    one_writes = {}
    for phase, effects in c.effects.items():
        if s.ONE in effects:
            one_writes[phase] = effects[s.ONE]
        if phase == "DECODE":
            continue
        on, off = set(), set()
        for op, lane in flags.items():
            if lane not in effects:
                continue
            expr = effects[lane]
            assert expr in (s.constant(1), s.constant(-1))
            (on if expr == s.constant(1) else off).add(op)
        if on or off:
            writers[phase] = ({op: (op in on) for op in sorted(on | off)},
                              sum(columns[s.WORK + flags[op]] for op in on),
                              sum(columns[s.WORK + flags[op]] for op in off))
    assert one_writes == {s.START: s.constant(1)}
    return flags, columns, writers, len(patterns) * (len(patterns) - 1) // 2


def verify_sparse_writers(c, ids, tracked, phases, gates, ups, downs):
    """Check that the actual augmented compiler has no extra flag writers."""
    written = defaultdict(dict)
    for gate, up, down, active in zip(gates, ups, downs, phases, strict=True):
        outputs = down.keys() & tracked
        if not outputs:
            continue
        terms = tuple(sorted((col, v) for col, v in gate.items() if not s.ACTION <= col < s.DISPATCH))
        for phase in active:
            col = s.ACTION + phase
            assert up[col] == 16 and gate[col] in (-4, 4)
            for lane in outputs:
                key = (terms, gate[col])
                holder = written[phase, lane]
                holder[key] = holder.get(key, 0) + down[lane]
    for phase, p in ids.items():
        for lane in tracked:
            if phase in (s.SEED_MISS, s.SEED_WORD):
                default = ()
            elif phase == s.START:
                default = s.constant(-1)
            else:
                default = s.buffer(s.WORK + lane)
            source = c.effects[phase].get(lane, default)
            expected = Counter()
            for value, terms in source:
                pred = defaultdict(int)
                for src, weight in terms:
                    pred[src] += 512 * weight
                gate = tuple(sorted((src, weight) for src, weight in pred.items() if weight))
                expected[gate, 4] += value
                expected[gate, -4] -= value
                if terms:
                    expected[(), 4] -= value
                    expected[(), -4] += value
            assert {k: v for k, v in written[p, lane].items() if v} == {k: v for k, v in expected.items() if v}
    return len(ids) * len(tracked)


def branch_value(c, phase, flag, columns, op_mask, one_known):
    if flag is None:
        return 0
    expr = c.effects[phase].get(abs(flag), s.buffer(s.WORK + abs(flag)))
    result = 0
    for factor, terms in expr:
        total = 0
        for lane, weight in terms:
            if lane == s.WORK + s.ONE and one_known:
                input_sign = 1
            elif lane in columns and op_mask is not None:
                input_sign = 1 if columns[lane] & op_mask else -1
            else:
                return None
            total += weight * input_sign
        if terms and not total:
            return None
        result += factor * ((1 if total > 0 else -1) if terms else 1)
    result *= 1 if flag > 0 else -1
    return result if result in (-1, 1) else None


def opcode_graph(c, ids, callers, columns, writers, diagnostic_fetch):
    """Keep unproved route choices and distinguish known initialization."""
    phases = tuple(ids)
    roots = returns.ROOTS if diagnostic_fetch else tuple(p for p in returns.ROOTS if p != s.FETCH)
    states = [(ids[p], None, None, False) for p in roots]
    indices = {state: i for i, state in enumerate(states)}
    edges = []
    usage = Counter()
    for phase_id, context, active, one_known in states:
        phase = phases[phase_id]
        next_one = phase == s.START or (one_known and phase not in (s.SEED_MISS, s.SEED_WORD))
        if phase == "DECODE" and one_known:
            next_masks = (0, *columns.values())
        elif phase == s.START:
            next_masks = (0,)
        elif active is not None and phase in writers:
            _, on, off = writers[phase]
            next_masks = ((active & ~off) | on,)
        elif phase in (s.SEED_MISS, s.SEED_WORD) or phase == "DECODE":
            next_masks = (None,)
        else:
            next_masks = (active,)
        routes = c.routes[phase]
        if phase == "ADD_FINISH":
            assert context is not None
            choices = (f"AFTER_{context}",)
        elif len(routes) == 1:
            choices = tuple(routes)
        else:
            values = {target: branch_value(c, phase, flag, columns, active, one_known)
                      for target, flag in routes.items()}
            positive = tuple(t for t, v in values.items() if v == 1)
            fallback = tuple(t for t, v in values.items() if v == 0)
            if None not in values.values() and (len(positive) == 1 or (not positive and len(fallback) == 1)):
                choices = positive or fallback
                usage[phase] += len(routes) - 1
            else:
                choices = tuple(routes)
        targets = []
        for choice in choices:
            next_ctx = (callers.get(phase, context) if choice in returns.ADD_CORE else None)
            assert next_ctx is not None or choice not in returns.ADD_CORE
            for next_mask in next_masks:
                key = (ids[choice], next_ctx, next_mask, next_one)
                if key not in indices:
                    indices[key] = len(states)
                    states.append(key)
                targets.append(indices[key])
        edges.append(tuple(targets))
    return states, edges, usage


def audit():
    before, data = returns.main()
    c, ids, _, _, _, base, deps, _, _, _, _, _, _, phases, gates, ups, downs = data
    callers = before["callers"]
    flags, columns, writers, tested = decoder(c)
    checked = verify_sparse_writers(c, ids, set(flags.values()) | {s.ONE}, phases, gates, ups, downs)
    output = {}
    for diagnostic in (False, True):
        ls, edges, usage = opcode_graph(c, ids, callers, columns, writers, diagnostic)
        reduced = [(p, mode) for p, mode, _, _ in ls]
        ins, outs = returns.solve(reduced, edges, deps, base)
        result = returns.summary(reduced, ins, outs, tuple(ids))
        state = max(range(len(ins)), key=lambda i: ins[i].bit_count())
        truth = ls[state][2]
        result.update(states=len(ls), edges=sum(map(len, edges)), recognized_route_prunes=dict(sorted(usage.items())),
                      peak_known_active_op_flags=[op for op, lane in flags.items()
                          if truth is not None and truth & columns[s.WORK + lane]], peak_flags_known=truth is not None,
                      fixed_map_clique=returns.verified_fixed_map_clique(ins + outs),
                      fixed_map_greedy_colors=len(set(live.color(ins + outs).values())))
        output["with_unknown_fetch_root" if diagnostic else "after_symbolic_start"] = result
    return dict(source_sha256={p.name: sha256(p.read_bytes()).hexdigest()
                    for p in (Path(__file__), Path(returns.__file__), Path(returns.full.__file__))},
                decode_flag_count=len(flags), disjoint_decode_pattern_pair_checks=tested,
                real_sparse_opcode_and_reference_write_checks=checked,
                literal_opcode_writers={phase: v[0] for phase, v in writers.items()},
                results=output,
                limits=("Refines all fully known signed opcode predicates, keeps ambiguous scores and all unknown data predicates. "
                        "The initialized-only sensitivity drops the unconstrained FETCH diagnostic root. "
                        "Nested non-opcode selectors and exact stock arithmetic remain open."))


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
