"""Build the recurrent RV32M phases around the scalar prefix adder."""

M_OPS = ('MUL', 'MULH', 'MULHSU', 'MULHU', 'DIV', 'DIVU', 'REM', 'REMU')
M_RETURN = ('M_MUL', 'M_MUL_CORRECT', 'M_DIV_NEG_A', 'M_DIV_NEG_B', 'M_DIV_STEP')
M_LOW, M_HIGH, M_DIVISOR = 1104, 1136, 1168
M_COUNT, M_FLAG = 1200, 1205


def add_m_program(c, *, state, rs1, rs2, zero, carry_out, result, finish, op_flags):
    from export_stock_mtp_scalar import RD_WORD, WORD, buffer, match, threshold, xor

    low, high, divisor = state(M_LOW), state(M_HIGH), state(M_DIVISOR)
    counter = state(M_COUNT, 5)
    c.reserve_state(*range(M_LOW, M_FLAG + 1))
    op = {name: state(op_flags[name], 1)[0] for name in M_OPS}
    zeros = (zero,) * WORD

    def decision(phase, expression, yes, no):
        c.put(phase, M_FLAG, expression)
        c.branch(phase, M_FLAG, yes, no)

    def select(*sources):
        return match(tuple((source, -1) for source in sources), -1)

    def output(phase, sources):
        c.move(phase, RD_WORD, sources)
        c.route(phase, finish)

    def tick(phase, repeat, done):
        # The iteration that starts at 31 writes the final 32-bit result.
        for bit, lane in enumerate(counter):
            if bit == 0:
                c.put(phase, M_COUNT, buffer(lane, -1))
                continue
            lower = tuple((source, 1) for source in counter[:bit])
            c.put(phase, M_COUNT + bit, (*buffer(lane), *match((*lower, (lane, -1))),
                                         *match((*lower, (lane, 1)), -1)))
        decision(phase, match(tuple((lane, 1) for lane in counter)), done, repeat)

    for name in M_OPS[:4]:
        c.route(f'EXEC_{name}', 'M_MUL_INIT')
    c.move('M_MUL_INIT', M_LOW, rs2)
    c.fix('M_MUL_INIT', M_HIGH, 0)
    c.fix('M_MUL_INIT', M_COUNT, 0, 5)
    c.route('M_MUL_INIT', 'M_MUL_PICK')

    # In each unsigned shift-add step, the low bit selects an add into the high
    # half. Shift the combined 65-bit carry/high/low value right by one.
    c.branch('M_MUL_PICK', M_LOW, 'M_MUL_ADD', 'M_MUL_SKIP')
    c.add('M_MUL_ADD', high, rs1, 'M_MUL')
    for phase, upper, carry in (('AFTER_M_MUL', result, carry_out), ('M_MUL_SKIP', high, zero)):
        c.move(phase, M_LOW, (*low[1:], upper[0]))
        c.move(phase, M_HIGH, (*upper[1:], carry))
        tick(phase, 'M_MUL_PICK', 'M_MUL_DONE')

    c.branch('M_MUL_DONE', op_flags['MUL'], 'M_MUL_LOW', 'M_MUL_SIGN_A')
    output('M_MUL_LOW', low)
    # A signed input changes the upper word by subtracting the opposite input.
    decision('M_MUL_SIGN_A', threshold(((rs1[-1], 2), (op['MULH'], 1), (op['MULHSU'], 1)), -1),
             'M_MUL_CORRECT_A', 'M_MUL_SIGN_B')
    c.add('M_MUL_CORRECT_A', high, rs2, 'M_MUL_CORRECT', subtract=True)
    c.move('AFTER_M_MUL_CORRECT', M_HIGH, result)
    c.route('AFTER_M_MUL_CORRECT', 'M_MUL_SIGN_B')
    decision('M_MUL_SIGN_B', match(((rs2[-1], 1), (op['MULH'], 1))), 'M_MUL_CORRECT_B', 'M_MUL_HIGH')
    c.add('M_MUL_CORRECT_B', high, rs1, 'RD', subtract=True)
    output('M_MUL_HIGH', high)

    for name in M_OPS[4:]:
        c.route(f'EXEC_{name}', 'M_DIV_ZERO')
    decision('M_DIV_ZERO', match(tuple((lane, -1) for lane in rs2)), 'M_DIV_ZERO_RESULT', 'M_DIV_INIT')
    decision('M_DIV_ZERO_RESULT', select(op['REM'], op['REMU']), 'M_DIV_ZERO_REM', 'M_DIV_ZERO_QUOT')
    output('M_DIV_ZERO_REM', rs1)
    c.fix('M_DIV_ZERO_QUOT', RD_WORD, -1)
    c.route('M_DIV_ZERO_QUOT', finish)

    # Signed division uses unsigned magnitudes. INT_MIN stays 0x80000000 when
    # negated, so the magnitude loop also yields INT_MIN / -1 and its remainder.
    c.move('M_DIV_INIT', M_LOW, rs1)
    c.move('M_DIV_INIT', M_DIVISOR, rs2)
    c.fix('M_DIV_INIT', M_HIGH, 0)
    c.fix('M_DIV_INIT', M_COUNT, 0, 5)
    decision('M_DIV_INIT', threshold(((rs1[-1], 2), (op['DIV'], 1), (op['REM'], 1)), -1),
             'M_DIV_NEG_A', 'M_DIV_SIGN_B')
    c.add('M_DIV_NEG_A', zeros, rs1, 'M_DIV_NEG_A', subtract=True)
    c.move('AFTER_M_DIV_NEG_A', M_LOW, result)
    c.route('AFTER_M_DIV_NEG_A', 'M_DIV_SIGN_B')
    decision('M_DIV_SIGN_B', threshold(((rs2[-1], 2), (op['DIV'], 1), (op['REM'], 1)), -1),
             'M_DIV_NEG_B', 'M_DIV_STEP')
    c.add('M_DIV_NEG_B', zeros, rs2, 'M_DIV_NEG_B', subtract=True)
    c.move('AFTER_M_DIV_NEG_B', M_DIVISOR, result)
    c.route('AFTER_M_DIV_NEG_B', 'M_DIV_STEP')

    # Shift the remainder and next dividend bit, then try the subtraction.
    # The shifted remainder is at least the divisor if bit 32 is set or the adder
    # reports no borrow. Retain the shifted remainder when subtraction borrows.
    shifted_high = (low[-1], *high[:-1])
    c.add('M_DIV_STEP', shifted_high, divisor, 'M_DIV_STEP', subtract=True)
    decision('AFTER_M_DIV_STEP', select(high[-1], carry_out), 'M_DIV_ACCEPT', 'M_DIV_RESTORE')
    for phase, accepted, upper in (('M_DIV_ACCEPT', 1, result), ('M_DIV_RESTORE', 0, shifted_high)):
        c.fix(phase, M_LOW, accepted, 1)
        c.move(phase, M_LOW + 1, low[:-1])
        c.move(phase, M_HIGH, upper)
        tick(phase, 'M_DIV_STEP', 'M_DIV_DONE')

    decision('M_DIV_DONE', select(op['REM'], op['REMU']), 'M_DIV_REM_SIGN', 'M_DIV_QUOT_SIGN')
    decision('M_DIV_REM_SIGN', match(((rs1[-1], 1), (op['REM'], 1))), 'M_DIV_NEG_REM', 'M_MUL_HIGH')
    c.add('M_DIV_NEG_REM', zeros, high, 'RD', subtract=True)
    c.put('M_DIV_QUOT_SIGN', M_FLAG, xor(rs1[-1], rs2[-1]))
    c.route('M_DIV_QUOT_SIGN', 'M_DIV_QUOT_SIGNED')
    decision('M_DIV_QUOT_SIGNED', match(((state(M_FLAG, 1)[0], 1), (op['DIV'], 1))), 'M_DIV_NEG_QUOT', 'M_MUL_LOW')
    c.add('M_DIV_NEG_QUOT', zeros, low, 'RD', subtract=True)
