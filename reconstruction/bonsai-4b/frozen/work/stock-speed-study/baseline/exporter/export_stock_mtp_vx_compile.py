"""Scalar-compatible circuit compiler with shared conditional dispatch egress."""

from collections import defaultdict
import math


def compile_circuit(c, logic):
    ids = {phase: index for index, phase in enumerate(c.phases)}
    if logic.ACTION + len(ids) > logic.DISPATCH:
        raise ValueError('Action and dispatch lanes overlap')
    if (len(set(c.persistent)) != len(c.persistent) or max(c.persistent) >= logic.WORK
            or logic.WORK + max(c.persistent) >= logic.ACTION):
        raise ValueError('Overlapping feedback lanes')
    if missing := set(c.phases) - set(c.routes):
        raise ValueError(f'Missing transitions: {sorted(missing)}')
    coefficients, extra_effects = {}, defaultdict(dict)
    next_dispatch = logic.DISPATCH
    for phase, routes in c.routes.items():
        shared = {}
        for target, flag in routes.items():
            coefficients[ids[target], logic.ACTION + ids[phase]] = 64
            if flag is None:
                continue
            expression = c.effects[phase].get(abs(flag), logic.buffer(logic.WORK + abs(flag)))
            if expression not in shared:
                shared[expression] = next_dispatch
                extra_effects[phase][next_dispatch] = expression
                next_dispatch += 1
            coefficients[ids[target], shared[expression]] = math.copysign(1 / 16, flag)
    if next_dispatch > logic.TINY_WIDTH:
        raise ValueError(f'Dispatch lanes exceed embedding width: {next_dispatch} > {logic.TINY_WIDTH}')
    effects = defaultdict(list)
    for phase, op in ids.items():
        phase_effects = c.effects[phase] | extra_effects[phase]
        if phase not in (logic.SEED_MISS, logic.SEED_WORD):
            for lane in c.persistent:
                phase_effects.setdefault(lane, logic.constant(-1) if phase == logic.START else logic.buffer(logic.WORK + lane))
        for dest, expression in phase_effects.items():
            for factor, terms in expression:
                effects[dest, factor, terms].append(logic.ACTION + op)
    groups = defaultdict(list)
    for (dest, factor, terms), op_lanes in effects.items():
        groups[factor, terms, tuple(op_lanes)].append(dest)
    gate, up, down, row = [], [], [], 0
    for (factor, terms, op_lanes), destinations in groups.items():
        variants = ((True, 4, 1), (True, -4, -1), (False, 4, -1), (False, -4, 1))
        if not terms:
            variants = ((False, 4, 1), (False, -4, -1))
        for include, bias, polarity in variants:
            if include:
                gate.extend(((row, lane), weight * 512) for lane, weight in terms)
            gate.extend(((row, lane), bias) for lane in op_lanes)
            up.extend(((row, lane), 16) for lane in op_lanes)
            down.extend(((dest, row), factor * polarity) for dest in destinations)
            row += 1
    return ids, coefficients, gate, up, down, row
