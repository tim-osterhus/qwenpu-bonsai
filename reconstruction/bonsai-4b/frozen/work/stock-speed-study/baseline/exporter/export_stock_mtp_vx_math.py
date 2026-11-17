"""Exact VX bit microcode for the stock recurrent threshold circuit.

All values are little-endian tuples of signed Boolean lanes (-1/1).  ``src``
arguments contain read/work lanes; ``dst`` and scratch arguments contain physical
feedback lanes.  The caller owns/reserves scratch, registers the returned adder
return modes, and supplies the existing scalar ``logic`` module and ``state``
mapping.  No arithmetic here is a host runtime implementation of VX.
"""

import math


SILU512 = tuple(round((x := (i - 256) / 32) / (1 + math.exp(-x)) * 4096)
                for i in range(512))
EXP256 = tuple(round(127 * math.exp(-i / 32)) for i in range(256))


def _move(c, phase, dst, src, logic):
    for dest, source in zip(dst, src, strict=True):
        c.put(phase, dest, logic.buffer(source))


def _fix(c, phase, dst, value, logic):
    for bit, dest in enumerate(dst):
        c.put(phase, dest, logic.constant(1 if value >> bit & 1 else -1))


def resize(src, width, zero, signed=True):
    """A static slice/sign extension; does not allocate or emit microcode."""
    src = tuple(src)
    return src[:width] + ((src[-1] if signed else zero),) * max(0, width - len(src))


def window(src, shift, width, zero, signed=True):
    """Bits of arithmetic ``src >> shift`` (negative shift means left)."""
    src = tuple(src)
    fill = src[-1] if signed else zero
    return tuple(zero if i + shift < 0 else src[i + shift] if i + shift < len(src) else fill
                 for i in range(width))


def emit_lut(c, phase, inputs, dst, values, logic):
    """Emit an exact constant LUT, sharing one match across every output bit.

    Inputs are lanes, or ``(lane, polarity)`` pairs.  For a bit with N positive
    entries, ``N - 1 + sum(match(entry))`` is exactly -1/1.  The compiler shares
    matches across all output bits, so a 512-entry SiLU table costs at most 2048
    predicate rows, rather than one separate table per bit.
    """
    inputs = tuple((source, 1) if isinstance(source, int) else source for source in inputs)
    if len(values) != 1 << len(inputs):
        raise ValueError('LUT values must cover exactly every input')
    matches = tuple(logic.match(tuple((source, polarity if value >> bit & 1 else -polarity)
                                     for bit, (source, polarity) in enumerate(inputs)))
                    for value in range(len(values)))
    for bit, dest in enumerate(dst):
        positive = [matches[index] for index, value in enumerate(values) if value >> bit & 1]
        c.put(phase, dest, (*logic.constant(len(positive) - 1),
                            *(term for expression in positive for term in expression)))


def emit_clamp16(c, phase, src, dst, logic):
    """Clamp a signed >=16-bit value to int16, exactly as pinned vxcirc.py."""
    if len(src) < 16 or len(dst) != 16:
        raise ValueError('clamp16 requires a signed >=16-bit source and 16-bit destination')
    zeros = tuple((lane, -1) for lane in src[15:])
    ones = tuple((lane, 1) for lane in src[15:])
    for bit, dest in enumerate(dst[:15]):
        c.put(phase, dest, (*logic.match(((src[bit], 1), *zeros)),
                            *logic.match(((src[bit], 1), *ones)),
                            *logic.buffer(src[-1], -1), *logic.match(zeros, -1),
                            *logic.constant(1)))
    c.put(phase, dst[15], logic.buffer(src[-1]))


def emit_sat127(c, phase, src, dst, logic):
    """Clamp a signed >=8-bit value to [-127,127] (including -128 -> -127)."""
    if len(src) < 8 or len(dst) != 8:
        raise ValueError('sat127 requires a signed >=8-bit source and 8-bit destination')
    zeros = tuple((lane, -1) for lane in src[7:])
    ones = tuple((lane, 1) for lane in src[7:])
    minus128 = (*ones, *((lane, -1) for lane in src[:7]))
    for bit, dest in enumerate(dst[:7]):
        expression = (*logic.match(((src[bit], 1), *zeros)),
                      *logic.match(((src[bit], 1), *ones)),
                      *logic.buffer(src[-1], -1), *logic.match(zeros, -1),
                      *logic.constant(1))
        if bit == 0:
            expression += (*logic.buffer(src[-1]), *logic.match(ones, -1),
                           *logic.match(minus128), *logic.constant(1))
        c.put(phase, dest, expression)
    c.put(phase, dst[7], logic.buffer(src[-1]))


def emit_vq_round(c, *, entry, finish, product, rounded, dst, flag, state, logic):
    """Exact ties-away rounding of signed clamp16*r, followed by sat127.

    ``product`` is a read tuple of 32 bits; ``rounded`` needs 16 physical lanes,
    ``dst`` 8, and ``flag`` one.  This does not use the prefix adder: round-up is
    bit14 AND (nonnegative OR any lower bit), then an increment of bits15..30.
    """
    if len(product) != 32 or len(rounded) != 16 or len(dst) != 8:
        raise ValueError('VQ rounding widths must be 32,16,8')
    do_round, do_clamp = entry + '_BITS', entry + '_CLAMP'
    negative_tie = ((product[14], 1), (product[31], 1),
                    *((lane, -1) for lane in product[:14]))
    c.put(entry, flag, (*logic.buffer(product[14]), *logic.match(negative_tie, -1),
                        *logic.constant(-1)))
    c.route(entry, do_round)
    for bit, dest in enumerate(rounded):
        source = product[15 + bit]
        carry = ((state((flag,))[0], 1), *((lane, 1) for lane in product[15:15 + bit]))
        c.put(do_round, dest, (*logic.buffer(source),
                              *logic.match((*carry, (source, -1))),
                              *logic.match((*carry, (source, 1)), -1)))
    c.route(do_round, do_clamp)
    emit_sat127(c, do_clamp, state(rounded), dst, logic)
    c.route(do_clamp, finish)


def emit_add_constant(c, phase, src, dst, value, logic):
    """A small-width modular constant add using at most two thresholds per bit."""
    if len(src) != len(dst):
        raise ValueError('constant add widths must match')
    for bit, dest in enumerate(dst):
        period = 1 << (bit + 1)
        initial = 1 if value >> bit & 1 else -1
        points = sorted({(-value) % period, ((1 << bit) - value) % period} - {0})
        expression = list(logic.constant(initial))
        previous = initial
        for point in points:
            current = 1 if (point + value) >> bit & 1 else -1
            delta = (current - previous) // 2
            if delta:
                terms = tuple((source, 1 << i) for i, source in enumerate(src[:bit + 1]))
                expression += logic.threshold(terms, (period - 1) - 2 * point + 1, delta)
                expression += logic.constant(delta)
            previous = current
        c.put(phase, dest, tuple(expression))


def _tick(c, phase, counter, direction, state, logic):
    """Counter +/-1 in one recurrent phase, without prefix-adder scratch."""
    source = state(counter)
    for bit, dest in enumerate(counter):
        if bit == 0:
            c.put(phase, dest, logic.buffer(source[0], -1))
        else:
            carry = tuple((lane, 1 if direction > 0 else -1) for lane in source[:bit])
            c.put(phase, dest, (*logic.buffer(source[bit]),
                                *logic.match((*carry, (source[bit], -1))),
                                *logic.match((*carry, (source[bit], 1)), -1)))


def emit_signed_shift(c, *, entry, finish, value, count, amount, flag, zero, state, logic):
    """Shift an in-place signed value by a signed amount (left positive).

    Scratch is only the caller's signed counter and flag.  Arithmetic right
    shifting happens before any 32-bit truncation.  The loop's rows are constant
    in the supported shift range; a 40/64-bit barrel shifter is unnecessary.
    """
    if len(count) != len(amount):
        raise ValueError('shift counter and amount widths must match')
    test, direction, left, right = (entry + suffix for suffix in ('_TEST', '_DIR', '_LEFT', '_RIGHT'))
    _move(c, entry, count, amount, logic)
    c.route(entry, test)
    c.put(test, flag, logic.match(tuple((lane, -1) for lane in state(count))))
    c.branch(test, flag, finish, direction)
    c.put(direction, flag, logic.buffer(state(count)[-1]))
    c.branch(direction, flag, right, left)
    source = state(value)
    _move(c, left, value, (zero, *source[:-1]), logic)
    _tick(c, left, count, -1, state, logic)
    c.route(left, test)
    _move(c, right, value, (*source[1:], source[-1]), logic)
    _tick(c, right, count, 1, state, logic)
    c.route(right, test)


def mul32_returns(prefix):
    return tuple(prefix + suffix for suffix in ('_ADD', '_CORRECT_A', '_CORRECT_B'))


def emit_mul32(c, *, entry, finish, prefix, left, right, low, high, counter,
               flag, result, carry_out, zero, state, logic, signed_a=True, signed_b=True):
    """Full exact 32x32 ->64 product using the existing 32-bit prefix adder.

    Scratch: low32, high32, counter5, flag1 (70 bits).  Inputs must remain live
    until completion and must not overlap low/high/counter.  The caller adds
    ``mul32_returns(prefix)`` to Circuit's return modes before scalar program
    construction.  Signed correction changes only the high word, so there is no
    lossy abs(INT_MIN) special case.  Wider VX products are truncated only after
    the required arithmetic shift/window has been taken.
    """
    if any(len(word) != 32 for word in (left, right, low, high, result)) or len(counter) != 5:
        raise ValueError('mul32 requires 32-bit words and a 5-bit counter')
    ret_add, ret_a, ret_b = mul32_returns(prefix)
    pick, add, skip, done, sign_a, sign_b = (prefix + suffix for suffix in
                                            ('_PICK', '_STEP', '_SKIP', '_DONE', '_SIGN_A', '_SIGN_B'))
    after_add, after_a, after_b = ('AFTER_' + mode for mode in (ret_add, ret_a, ret_b))
    _move(c, entry, low, right, logic)
    _fix(c, entry, high, 0, logic)
    _fix(c, entry, counter, 0, logic)
    c.route(entry, pick)
    c.put(pick, flag, logic.buffer(state(low)[0]))
    c.branch(pick, flag, add, skip)
    c.add(add, state(high), left, ret_add)
    for phase, upper, carry in ((after_add, result, carry_out), (skip, state(high), zero)):
        _move(c, phase, low, (*state(low)[1:], upper[0]), logic)
        _move(c, phase, high, (*upper[1:], carry), logic)
        _tick(c, phase, counter, 1, state, logic)
        c.put(phase, flag, logic.match(tuple((lane, 1) for lane in state(counter))))
        c.branch(phase, flag, done, pick)
    if signed_a:
        c.put(done, flag, logic.buffer(left[-1]))
        c.branch(done, flag, sign_a, sign_b)
        c.add(sign_a, state(high), right, ret_a, subtract=True)
        _move(c, after_a, high, result, logic)
        c.route(after_a, sign_b)
    else:
        c.route(done, sign_b)
    if signed_b:
        c.put(sign_b, flag, logic.buffer(right[-1]))
        c.branch(sign_b, flag, prefix + '_CORRECT', finish)
        c.add(prefix + '_CORRECT', state(high), left, ret_b, subtract=True)
        _move(c, after_b, high, result, logic)
        c.route(after_b, finish)
    else:
        c.route(sign_b, finish)


def emit_silu(c, *, entry, finish, src, dst, flag, state, logic):
    """Pinned Q16 ->Q12 SILU512, including large-input saturation."""
    if len(src) != 32 or len(dst) != 32:
        raise ValueError('SiLU requires 32-bit source and destination')
    small, big, table, zero, saturate, linear = (entry + suffix for suffix in
                                               ('_SMALL', '_BIG', '_TABLE', '_ZERO', '_SAT', '_LINEAR'))
    c.put(entry, flag, (*logic.buffer(src[31], -1),
                        *logic.match(tuple((lane, -1) for lane in src[19:]), -1),
                        *logic.constant(-1)))
    c.branch(entry, flag, big, small)
    c.put(small, flag, (*logic.buffer(src[31]),
                        *logic.match(tuple((lane, 1) for lane in src[19:]), -1),
                        *logic.constant(-1)))
    c.branch(small, flag, zero, table)
    c.put(big, flag, logic.match(tuple((lane, -1) for lane in src[27:])))
    c.branch(big, flag, linear, saturate)
    emit_lut(c, table, (*src[11:19], (src[19], -1)), dst, SILU512, logic)
    c.route(table, finish)
    _fix(c, zero, dst, 0, logic)
    c.route(zero, finish)
    _fix(c, saturate, dst, 1 << 23, logic)
    c.route(saturate, finish)
    for bit, dest in enumerate(dst):
        c.put(linear, dest, logic.buffer(src[bit + 4]) if bit + 4 < len(src) else logic.constant(-1))
    c.route(linear, finish)


def emit_exp(c, *, entry, finish, src, dst, flag, logic):
    """Pinned EXP256 for a nonnegative integer d; d >=256 returns zero."""
    if len(src) < 8 or len(dst) != 8:
        raise ValueError('EXP requires >=8-bit input and 8-bit output')
    table, zero = entry + '_TABLE', entry + '_ZERO'
    c.put(entry, flag, logic.match(tuple((lane, -1) for lane in src[8:])))
    c.branch(entry, flag, table, zero)
    emit_lut(c, table, src[:8], dst, EXP256, logic)
    c.route(table, finish)
    _fix(c, zero, dst, 0, logic)
    c.route(zero, finish)
