"""Pinned VEW, VMATR and VEXP microcode for the stock MTP VX exporter.

Only recurrent Boolean/threshold operations and the existing neural scalar
adder/multiplier are used.  All scratch is borrowed from the idle VMAT tile.
"""

import export_stock_mtp_vx_math as mathunit


RETURNS = ('VX_OP_ADD',)


def add_ops(b, p):
    c, s, m = b.c, p['logic'], p['m']
    cfg, elem = p['cfg'], p['elem']
    prod, px, pp, pg, act, weights = (p[name] for name in ('prod', 'px', 'pp', 'pg', 'act', 'weights'))
    flag, auxflag = p['flag'], p['auxflag']
    result, acc = p['result'], p['acc']
    state = lambda base, n=32: s.bits(s.WORK + base, n)
    zero, one = s.WORK + s.ZERO, s.WORK + s.ONE
    cs, es = b.state(cfg), b.state(elem)
    param1, param2 = act[:32], act[32:]
    p1, p2 = b.state(param1), b.state(param2)
    left, right = prod[:32], prod[32:64]
    u, w = prod[64:80], prod[80:96]
    # Keep full products directly in the scalar multiplier's live words.  Both
    # words remain scratch until the shared tracker consumes the final result.
    wide, saved = (*s.bits(m.M_LOW), *s.bits(m.M_HIGH)), pp[:32]
    cosine, sine = pp[32:48], pp[48:64]
    lo, hi = b.state(wide[:32]), b.state(wide[32:])
    dot = weights[16:22]
    ds = b.state(dot)
    amount, shift_counter = pg[11:17], pg[5:11]
    carry = pg[17]
    limit = pg[20:26]
    add_sel, mul_sel, shift_sel, wide_sel = weights[:4], weights[4:8], weights[8:12], weights[12:16]
    add_targets, mul_targets, shift_targets, wide_targets = {}, {}, {}, {}
    clamp_sel, byte_offset, byte_sel = pg[26:30], pg[30:32], pg[48:52]
    rope_u = pg[32:48]
    clamp_targets, byte_targets, byte_continuations = {}, {}, {}

    def selector(phase, physical, targets, target):
        code = targets.setdefault(target, len(targets))
        if code >= 16:
            raise ValueError('VX arithmetic continuation field overflow')
        b.fix(phase, physical, code)

    def dispatch(phase, physical, targets):
        for target, code in targets.items():
            physical_flag = s.EXTRACT_FLAG + code
            c.put(phase, physical_flag, s.match(tuple((lane, s.sign(code, bit))
                for bit, lane in enumerate(b.state(physical)))))
            c.route(phase, (target, physical_flag))

    def add(phase, a, d, target, *, subtract=False, carry_in=None):
        selector(phase, add_sel, add_targets, target)
        c.add(phase, a, d, 'VX_OP_ADD', subtract=subtract, carry_in=carry_in)

    def multiply(phase, a, d, target):
        c.move(phase, s.RS1, b.resize(a, signed=True))
        c.move(phase, s.RS2, b.resize(d, signed=True))
        for name in ('MUL', 'MULHSU', 'MULHU'):
            c.fix(phase, c.op_flag[name], 0, 1)
        c.fix(phase, c.op_flag['MULH'], 1, 1)
        selector(phase, mul_sel, mul_targets, target)
        c.route(phase, 'M_MUL_INIT')

    def shift32(phase, source, shift, target, *, signed=True):
        c.move(phase, s.A, b.resize(source, signed=signed))
        c.move(phase, s.B, b.resize(shift))
        c.fix(phase, s.SHIFT_LEFT, 0, 1)
        c.fix(phase, s.SHIFT_ARITH, int(signed), 1)
        selector(phase, shift_sel, shift_targets, target)
        c.route(phase, 'SHIFT_TEST_0')

    def shift_wide(phase, shift, target):
        b.move(phase, amount, shift)
        selector(phase, wide_sel, wide_targets, target)
        c.route(phase, 'VX_OP_WSHIFT')

    def or_index(phase, physical, a, d):
        for dest, av, dv in zip(physical, a, d):
            c.put(phase, dest, s.threshold(((av, 1), (dv, 1)), 1))

    def read_vm(phase, reg, offset, element, target):
        or_index(phase, s.bits(s.A, 6), offset, element)
        c.route(phase, phase + '_QUERY')
        b.read(phase + '_QUERY', 'vm', (*state(s.A, 6), *reg), target)

    def packed_byte(phase, space, index, byte, destination, target):
        key = tuple(destination), target
        if key not in byte_continuations:
            continuation = phase + '_BYTE_DONE'
            byte_continuations[key] = continuation
            b.move(continuation, destination, state(s.RS1))
            c.route(continuation, target)
        selector(phase, byte_sel, byte_targets, byte_continuations[key])
        b.move(phase, byte_offset, byte)
        b.read(phase, space, index, 'VX_OP_BYTE_READ')

    def clamp(phase, target):
        selector(phase, clamp_sel, clamp_targets, target)
        c.route(phase, 'VX_OP_CLAMP16')

    def meta(phase, payload, reg, offset, keep, signed, next_phase):
        # VM words carry the sign in bit30; normalize bit31 after all wide work.
        b.move(phase, result, (*payload[:31], payload[30]))
        or_index(phase, p['result_index'][:6], offset, es[:6])
        b.move(phase, p['result_index'][6:], reg)
        b.move(phase, p['result_keep'], (keep,))
        b.fix(phase, p['result_signed'], int(signed))
        code = b.result_targets.setdefault(next_phase, len(b.result_targets))
        b.fix(phase, p['result_next'], code)
        c.route(phase, 'VX_ABS_TEST')

    # The scalar MULH already leaves an exact signed64 result in M_HIGH:M_LOW.
    # Keep the unqualified scalar path as fallback; a true VX route wins its
    # dispatch score by 1/16, as for existing scalar/VX shared continuations.
    vector_multiply = tuple((s.WORK + c.op_flag[name], 1) for name in ('VEW', 'VMATR', 'VEXP'))
    c.put('M_MUL_HIGH', auxflag, s.threshold(vector_multiply, 2))
    c.route('M_MUL_HIGH', ('VX_OP_MUL_OUT', auxflag))
    # The scalar signed-B correction returns via RD with its high word in SUM;
    # commit that word before taking the shared full-product continuation.
    c.put('AFTER_RD', auxflag, s.threshold(vector_multiply, 2))
    c.route('AFTER_RD', ('VX_OP_MUL_NEG_B', auxflag))
    c.move('VX_OP_MUL_NEG_B', m.M_HIGH, state(s.SUM))
    c.route('VX_OP_MUL_NEG_B', 'VX_OP_MUL_OUT')
    c.route('VX_OP_MUL_OUT', 'VX_OP_MUL_RETURN')
    c.put('SHIFT_FINISH', auxflag, s.threshold(tuple((s.WORK + c.op_flag[name], 1)
        for name in ('VEW', 'VEXP')), 1))
    c.route('SHIFT_FINISH', ('VX_OP_SHIFT_RETURN', auxflag))

    # -------------------------- VMATR: signed scores and wide p.V scaling.
    b.move('VX_EXEC_VMATR', param1, state(s.RS1))
    b.move('VX_EXEC_VMATR', param2, state(s.RS2))
    b.fix('VX_EXEC_VMATR', elem, 0)
    c.routes['VX_EXEC_VMATR'] = {'VX_R_INIT': None}
    b.fix('VX_R_INIT', acc, 0)
    b.fix('VX_R_INIT', dot, 0)
    c.branch('VX_R_INIT', cfg[10], 'VX_R_LIMIT_PV', 'VX_R_LIMIT_SCORE')
    b.move('VX_R_LIMIT_PV', limit, cs[4:10])
    c.route('VX_R_LIMIT_PV', 'VX_R_VA')
    b.fix('VX_R_LIMIT_SCORE', limit, 8)
    c.route('VX_R_LIMIT_SCORE', 'VX_R_VA')
    packed_byte('VX_R_VA', 'va', (*ds[2:6], *cs[12:15]), ds[:2], left, 'VX_R_CACHE')
    c.branch('VX_R_CACHE', cfg[10], 'VX_R_VC', 'VX_R_KC')
    packed_byte('VX_R_KC', 'kc', (*ds[2:3], *es[:5], *p1[5:10]), ds[:2], right, 'VX_R_MUL')
    packed_byte('VX_R_VC', 'vc', (*es[2:3], *ds[:5], *p1[5:10]), es[:2], right, 'VX_R_MUL')
    multiply('VX_R_MUL', b.state(left), b.state(right), 'VX_R_PRODUCT')
    add('VX_R_PRODUCT', b.state(acc), lo, 'VX_R_ACC')
    b.move('VX_R_ACC', acc, state(s.SUM))
    b.inc('VX_R_ACC', dot)
    c.route('VX_R_ACC', 'VX_R_DOT_LAST')
    after_eq = b.eq('VX_R_DOT_LAST', flag, ds, b.state(limit))
    c.branch(after_eq, flag, 'VX_R_INNER_DONE', 'VX_R_VA')
    c.branch('VX_R_INNER_DONE', cfg[10], 'VX_R_SCALE', 'VX_R_SCORE')
    multiply('VX_R_SCALE', b.state(acc), b.resize(p2[:17]), 'VX_R_SCALED')
    meta('VX_R_SCALED', mathunit.window((*lo, *hi), 12, 32, zero), cs[:4], p1[10:16], cs[11], False, 'VX_R_NEXT')
    meta('VX_R_SCORE', b.state(acc), cs[:4], p1[10:16], cs[11], True, 'VX_R_NEXT')
    b.inc('VX_R_NEXT', elem)
    c.route('VX_R_NEXT', 'VX_R_LAST')
    c.branch('VX_R_LAST', cfg[10], 'VX_R_LAST_PV', 'VX_R_LAST_SCORE')
    after_eq = b.eq('VX_R_LAST_PV', flag, es, b.const(8, 7))
    c.branch(after_eq, flag, s.COMMIT, 'VX_R_INIT')
    after_eq = b.eq('VX_R_LAST_SCORE', flag, es, (*cs[4:10], zero))
    c.branch(after_eq, flag, 'VX_R_PAD_CHECK', 'VX_R_INIT')
    c.branch('VX_R_PAD_CHECK', cfg[4], 'VX_R_PAD_INDEX', s.COMMIT)
    or_index('VX_R_PAD_INDEX', s.bits(s.A, 6), p1[10:16], cs[4:10])
    c.route('VX_R_PAD_INDEX', 'VX_R_PAD_STORE')
    b.write('VX_R_PAD_STORE', 'vm', (*state(s.A, 6), *cs[:4]), b.const(0), s.COMMIT)

    # -------------------------- VEW: ADD, GLU and pair-preserving RoPE.
    b.move('VX_EXEC_VEW', param1, state(s.RS1))
    b.move('VX_EXEC_VEW', param2, state(s.RS2))
    b.fix('VX_EXEC_VEW', elem, 0)
    c.routes['VX_EXEC_VEW'] = {'VX_W_INIT': None}
    c.branch('VX_W_INIT', cfg[16], 'VX_W_REVEN', 'VX_W_A')
    read_vm('VX_W_A', cs[:4], cs[18:24], es[:6], 'VX_W_A_READ')
    b.move('VX_W_A_READ', pp[:32], s.bits(s.READBACK))
    c.branch('VX_W_A_READ', cfg[15], 'VX_W_A_SHIFT', 'VX_W_B')
    shift32('VX_W_A_SHIFT', b.state(pp[:32]), p1[:5], 'VX_W_CLAMP_P')
    read_vm('VX_W_B', cs[4:8], cs[18:24], es[:6], 'VX_W_B_READ')
    c.move('VX_W_B_READ', s.EQUAL, s.bits(s.READBACK))
    c.branch('VX_W_B_READ', cfg[15], 'VX_W_B_SHIFT_G', 'VX_W_B_SHIFT_A')
    shift32('VX_W_B_SHIFT_A', state(s.EQUAL), p1[:5], 'VX_W_CLAMP_P')
    shift32('VX_W_B_SHIFT_G', state(s.EQUAL), p1[11:16], 'VX_W_CLAMP_GB')
    clamp('VX_W_CLAMP_P', 'VX_W_P_MUL')
    multiply('VX_W_P_MUL', b.resize(b.state(u), signed=True), b.resize(p2[:15]), 'VX_W_P_PRODUCT')
    shift_wide('VX_W_P_PRODUCT', p1[5:11], 'VX_W_P_SHIFTED')
    c.branch('VX_W_P_SHIFTED', cfg[15], 'VX_W_SILU', 'VX_W_ADD_RESULT')
    add('VX_W_ADD_RESULT', lo, b.state(pp[:32]), 'VX_W_ADD_DONE')
    meta('VX_W_ADD_DONE', state(s.SUM), cs[:4], cs[18:24], cs[17], False, 'VX_W_NEXT')
    mathunit.emit_silu(c, entry='VX_W_SILU', finish='VX_W_B', src=lo, dst=pp[:32],
                       flag=flag, state=b.state, logic=s)
    clamp('VX_W_CLAMP_GB', 'VX_W_G_MUL')
    multiply('VX_W_G_MUL', b.state(pp[:32]), b.resize(b.state(u), signed=True), 'VX_W_G_PRODUCT')
    meta('VX_W_G_PRODUCT', mathunit.window((*lo, *hi), 8, 32, zero), cs[:4], cs[18:24], cs[17], False, 'VX_W_NEXT')

    read_vm('VX_W_REVEN', cs[:4], cs[18:24], (zero, *es[1:6]), 'VX_W_REVEN_READ')
    shift32('VX_W_REVEN_READ', s.bits(s.READBACK), p1[:5], 'VX_W_REVEN_CLAMP')
    clamp('VX_W_REVEN_CLAMP', 'VX_W_REVEN_SAVE')
    b.move('VX_W_REVEN_SAVE', rope_u, b.state(u))
    c.route('VX_W_REVEN_SAVE', 'VX_W_RODD')
    read_vm('VX_W_RODD', cs[:4], cs[18:24], (one, *es[1:6]), 'VX_W_RODD_READ')
    shift32('VX_W_RODD_READ', s.bits(s.READBACK), p1[:5], 'VX_W_RODD_CLAMP')
    clamp('VX_W_RODD_CLAMP', 'VX_W_RODD_SAVE')
    b.move('VX_W_RODD_SAVE', w, b.state(u))
    c.route('VX_W_RODD_SAVE', 'VX_W_COS')
    b.read('VX_W_COS', 'rope', (zero, *es[1:3]), 'VX_W_COS_READ')
    b.move('VX_W_COS_READ', cosine, s.bits(s.READBACK, 16))
    c.route('VX_W_COS_READ', 'VX_W_SIN')
    b.read('VX_W_SIN', 'rope', (one, *es[1:3]), 'VX_W_SIN_READ')
    b.move('VX_W_SIN_READ', sine, s.bits(s.READBACK, 16))
    c.branch('VX_W_SIN_READ', elem[0], 'VX_W_RF_ODD', 'VX_W_RF_EVEN')
    multiply('VX_W_RF_ODD', b.resize(b.state(rope_u), signed=True),
             b.resize(b.state(sine), signed=True), 'VX_W_RF_SAVE')
    multiply('VX_W_RF_EVEN', b.resize(b.state(rope_u), signed=True),
             b.resize(b.state(cosine), signed=True), 'VX_W_RF_SAVE')
    # One signed16*int16 product always fits signed32; its high word is the
    # sign extension of bit31.  The sum/difference still keeps all64 bits.
    b.move('VX_W_RF_SAVE', saved, lo)
    c.branch('VX_W_RF_SAVE', elem[0], 'VX_W_RS_ODD', 'VX_W_RS_EVEN')
    multiply('VX_W_RS_ODD', b.resize(b.state(w), signed=True),
             b.resize(b.state(cosine), signed=True), 'VX_W_RS_PRODUCT')
    multiply('VX_W_RS_EVEN', b.resize(b.state(w), signed=True),
             b.resize(b.state(sine), signed=True), 'VX_W_RS_PRODUCT')
    c.branch('VX_W_RS_PRODUCT', elem[0], 'VX_W_R_ADD_LOW', 'VX_W_R_SUB_LOW')
    add('VX_W_R_ADD_LOW', b.state(saved), lo, 'VX_W_R_LOW')
    add('VX_W_R_SUB_LOW', b.state(saved), lo, 'VX_W_R_LOW', subtract=True)
    b.move('VX_W_R_LOW', wide[:32], state(s.SUM))
    b.move('VX_W_R_LOW', (carry,), (s.WORK + s.G + 31,))
    c.branch('VX_W_R_LOW', elem[0], 'VX_W_R_ADD_HIGH', 'VX_W_R_SUB_HIGH')
    saved_high = (b.state(saved)[31],) * 32
    add('VX_W_R_ADD_HIGH', saved_high, hi, 'VX_W_R_HIGH', carry_in=b.state((carry,))[0])
    add('VX_W_R_SUB_HIGH', saved_high, hi, 'VX_W_R_HIGH', subtract=True,
        carry_in=b.state((carry,))[0])
    b.move('VX_W_R_HIGH', wide[32:], state(s.SUM))
    mathunit.emit_add_constant(c, 'VX_W_R_HIGH', (*p1[:5], zero), amount, -14, s)
    c.route('VX_W_R_HIGH', 'VX_W_R_SHIFT')
    shift_wide('VX_W_R_SHIFT', b.state(amount), 'VX_W_R_SHIFTED')
    meta('VX_W_R_SHIFTED', lo, cs[4:8], cs[18:24], cs[17], False, 'VX_W_NEXT')
    b.inc('VX_W_NEXT', elem)
    c.route('VX_W_NEXT', 'VX_W_LAST')
    after_eq = b.eq('VX_W_LAST', flag, es, cs[8:15])
    c.branch(after_eq, flag, s.COMMIT, 'VX_W_INIT')

    # -------------------------- VEXP: signed score difference and exact EXP256.
    b.move('VX_EXEC_VEXP', param1, state(s.RS1))
    b.move('VX_EXEC_VEXP', param2, state(s.RS2))
    b.fix('VX_EXEC_VEXP', elem, 0)
    c.routes['VX_EXEC_VEXP'] = {'VX_E_VM': None}
    b.read('VX_E_VM', 'vm', (*es[:6], *cs[:4]), 'VX_E_VM_READ')
    add('VX_E_VM_READ', b.resize(p1[:31], signed=True), s.bits(s.READBACK), 'VX_E_DIFFERENCE', subtract=True)
    c.put('VX_E_DIFFERENCE', flag, s.match(tuple((lane, -1) for lane in state(s.SUM)[16:])))
    c.branch('VX_E_DIFFERENCE', flag, 'VX_E_COPY', 'VX_E_SAT')
    b.move('VX_E_COPY', left, b.resize(state(s.SUM, 16)))
    c.route('VX_E_COPY', 'VX_E_FACTOR')
    b.fix('VX_E_SAT', left, 65535)
    c.route('VX_E_SAT', 'VX_E_FACTOR')
    multiply('VX_E_FACTOR', b.state(left), b.resize(p2[:16]), 'VX_E_PRODUCT')
    shift32('VX_E_PRODUCT', lo, cs[20:25], 'VX_E_EXP', signed=False)
    mathunit.emit_exp(c, entry='VX_E_EXP', finish='VX_E_ADDRESS', src=state(s.RD_WORD),
                      dst=prod[64:72], flag=flag, logic=s)
    or_index('VX_E_ADDRESS', s.bits(s.A, 9), cs[11:20], b.resize(es, 9))
    c.route('VX_E_ADDRESS', 'VX_E_WRITE_QUERY')
    b.read('VX_E_WRITE_QUERY', 'va', state(s.A)[2:9], 'VX_E_WRITE_READ')
    c.move('VX_E_WRITE_READ', s.EQUAL, s.bits(s.READBACK))
    c.move('VX_E_WRITE_READ', s.READ_ADDR, state(s.READ_ADDR, 33))
    c.fix('VX_E_WRITE_READ', s.READ_MARK, 1, 1)
    for offset in range(4):
        physical_flag = s.EXTRACT_FLAG + offset
        c.put('VX_E_WRITE_READ', physical_flag, s.match(tuple((lane, s.sign(offset, bit))
            for bit, lane in enumerate(state(s.A, 2)))))
        phase = 'VX_E_MERGE_' + str(offset)
        c.route('VX_E_WRITE_READ', (phase, physical_flag))
        payload = (*state(s.EQUAL)[:8 * offset], *b.state(prod[64:72]), *state(s.EQUAL)[8 * offset + 8:])
        b.write_payload(phase, payload, 'VX_E_WRITTEN', existing=True)
    # A padded odd element has zero probability and never reads VM or EXP(0).
    c.put('VX_E_WRITTEN', flag, s.match(tuple((lane, s.sign(1, bit))
        for bit, lane in enumerate(b.state(weights[24:25])))))
    c.branch('VX_E_WRITTEN', flag, s.COMMIT, 'VX_E_NEXT')
    b.fix('VX_EXEC_VEXP', weights[24:25], 0)
    b.inc('VX_E_NEXT', elem)
    c.route('VX_E_NEXT', 'VX_E_LAST')
    after_eq = b.eq('VX_E_LAST', flag, es, cs[4:11])
    c.branch(after_eq, flag, 'VX_E_PAD_CHECK', 'VX_E_VM')
    c.branch('VX_E_PAD_CHECK', cfg[4], 'VX_E_PAD', s.COMMIT)
    b.fix('VX_E_PAD', prod[64:72], 0)
    b.fix('VX_E_PAD', weights[24:25], 1)
    c.route('VX_E_PAD', 'VX_E_ADDRESS')

    # Shared arithmetic and continuation selectors are emitted after all callers
    # have registered their targets.  None require new persistent allocations.
    mathunit.emit_signed_shift(c, entry='VX_OP_WSHIFT', finish='VX_OP_WIDE_RETURN',
        value=wide, count=shift_counter, amount=b.state(amount), flag=flag,
        zero=zero, state=b.state, logic=s)
    mathunit.emit_clamp16(c, 'VX_OP_CLAMP16', state(s.RD_WORD), u, s)
    c.route('VX_OP_CLAMP16', 'VX_OP_CLAMP_RETURN')
    c.move('VX_OP_BYTE_READ', s.EQUAL, s.bits(s.READBACK))
    for offset in range(4):
        physical_flag = s.EXTRACT_FLAG + offset
        c.put('VX_OP_BYTE_READ', physical_flag, s.match(tuple((lane, s.sign(offset, bit))
            for bit, lane in enumerate(b.state(byte_offset)))))
        phase = 'VX_OP_BYTE_' + str(offset)
        c.route('VX_OP_BYTE_READ', (phase, physical_flag))
        c.move(phase, s.RS1, b.resize(state(s.EQUAL + 8 * offset, 8), signed=True))
        c.route(phase, 'VX_OP_BYTE_RETURN')
    dispatch('AFTER_VX_OP_ADD', add_sel, add_targets)
    dispatch('VX_OP_MUL_RETURN', mul_sel, mul_targets)
    dispatch('VX_OP_SHIFT_RETURN', shift_sel, shift_targets)
    dispatch('VX_OP_WIDE_RETURN', wide_sel, wide_targets)
    dispatch('VX_OP_CLAMP_RETURN', clamp_sel, clamp_targets)
    dispatch('VX_OP_BYTE_RETURN', byte_sel, byte_targets)
