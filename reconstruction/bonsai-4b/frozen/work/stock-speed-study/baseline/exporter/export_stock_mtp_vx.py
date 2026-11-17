"""Stock MTP RV32IM/Zba with a recurrent, memory-backed pinned VX datapath.

This module deliberately leaves the scalar exporter untouched.  The extra memory
addresses are transport addresses, not guest RAM.  WROM is losslessly reframed as
eight low nibbles per word; the other byte memories use four bytes per word.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import export_stock_mtp_scalar as s
import export_stock_mtp_scalar_m as m
import export_stock_mtp_vx_math as mathunit
import export_stock_mtp_vx_ops as ops
from export_stock_mtp_vx_compile import compile_circuit
from export_stock_mtp_vx_fast_add import emit_add as fast_add


ADDRESS = {'vm': 0x10000000, 'va': 0x20000000, 'kc': 0x30000000,
           'vc': 0x40000000, 'rope': 0x50000000, 'wrom': 0x60000000}
VX_OPS = ('VMAT', 'VMATR', 'VQ', 'VEW', 'VEXP', 'VGET', 'VRED', 'VPUT')
RETURNS = ('VX_WBASE', 'VX_WSTART', 'VX_ACC', 'VX_SCALE', 'VX_ABS', 'VX_MAX',
           'VX_ROW', 'VX_VQ_NEG', 'VX_VQ_MUL', 'VX_VQ_ROUND', 'VX_VRED',
           'VX_VRED_MUL', 'VX_GENERAL', 'VX_GENERAL2', 'VX_WSTEP', 'VX_ARGCOMPARE')
RETURNS += ops.RETURNS
EXTRA_READBACK = (985, 1017)


def configure():
    s.WORK, s.ACTION, s.TINY_WIDTH, s.TINY_VOCAB = 2048, 4096, 5120, 1024


def deduplicate(compiled):
    """Merge exactly identical gate/up rows, summing their down coefficients."""
    ids, scores, gate, up, down, rows = compiled
    gs, us, ds = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(float)), defaultdict(list)
    for (r, col), value in gate:
        gs[r][col] += value
    for (r, col), value in up:
        us[r][col] += value
    for (dest, r), value in down:
        ds[r].append((dest, value))
    unique, grouped = {}, defaultdict(lambda: defaultdict(float))
    for r in range(rows):
        key = tuple(sorted((col, value) for col, value in gs[r].items() if value)), tuple(sorted(
            (col, value) for col, value in us[r].items() if value))
        target = unique.setdefault(key, len(unique))
        for dest, value in ds[r]:
            grouped[target][dest] += value
    ng, nu, nd = [], [], []
    for (g, u), r in unique.items():
        ng.extend(((r, col), value) for col, value in g)
        nu.extend(((r, col), value) for col, value in u)
        nd.extend(((dest, r), value) for dest, value in grouped[r].items() if value)
    return ids, scores, ng, nu, nd, len(unique)


def normalize_expressions(c):
    """Combine linear expression factors before nonlinear phase gating."""
    for effects in c.effects.values():
        for destination, expression in tuple(effects.items()):
            factors = defaultdict(float)
            for factor, terms in expression:
                factors[terms] += factor
            effects[destination] = tuple((factor, terms) for terms, factor in factors.items() if factor)


def validate_readback_contract(c, b):
    ports = (s.bits(s.READBACK, s.WORD + s.HANDLE_BITS + 1), *(s.bits(base) for base in EXTRA_READBACK))
    physical = tuple(lane for port in ports for lane in port)
    if len(physical) != len(set(physical)):
        raise ValueError('Attention readback ports overlap')
    direct = set(physical)
    permitted = {s.FETCH, 'READ_RS1', 'READ_RS2', 'READ_A7', 'READ_A0',
                 'READ_LW', 'STORE_LOOKUP', 'REG_LOOKUP'} | b.reads
    for phase, effects in c.effects.items():
        if phase in permitted:
            continue
        for destination, expression in effects.items():
            for factor, terms in expression:
                if invalid := direct & {source for source, weight in terms}:
                    raise ValueError(f'Attention input used outside a read phase: {phase} -> {destination}: {sorted(invalid)}')


class Builder:
    def __init__(self, c):
        self.c = c
        blocked = set(c.persistent)
        blocked.update(range(64, 352))  # scalar seed/query ingress
        blocked.update(range(s.READBACK, EXTRA_READBACK[-1] + 32))
        blocked.update(range(s.EXTRACT_FLAG, s.EXTRACT_FLAG + 16))
        blocked.update(range(1536, 1600))  # all scalar and VX adder returns
        blocked.update(range(1600, 1600 + len(c.op_flag) + len(VX_OPS)))
        self.available = [lane for lane in range(2048) if lane not in blocked]
        self.fields = {}
        self.reads = set()
        self.streams = set()
        self.store_returns = {}
        self.store_flag = self.alloc('store_return', 5)
        self.pending_address = self.pending_word = None

    def alloc(self, name, n):
        if len(self.available) < n:
            raise ValueError(f'VX feedback overflow allocating {name}: {n} > {len(self.available)}')
        lanes, self.available = tuple(self.available[:n]), self.available[n:]
        self.fields[name] = lanes
        self.c.reserve_state(*lanes)
        return lanes

    def state(self, lanes):
        return tuple(s.WORK + lane for lane in lanes)

    def move(self, phase, dest, src, polarity=1):
        for lane, source in zip(dest, src):
            self.c.put(phase, lane, s.buffer(source, polarity))

    def fix(self, phase, dest, value):
        for bit, lane in enumerate(dest):
            self.c.put(phase, lane, s.constant(s.sign(value, bit)))

    def const(self, value, n=32):
        return tuple(s.WORK + (s.ONE if value >> bit & 1 else s.ZERO) for bit in range(n))

    def resize(self, source, n=32, signed=False):
        return (*source[:n], *((source[-1] if signed else s.WORK + s.ZERO),) * max(0, n - len(source)))

    def inc(self, phase, dest, step=1):
        if step <= 0 or step & (step - 1):
            raise ValueError('Increment must be a power of two')
        start = step.bit_length() - 1
        src = self.state(dest)
        for bit in range(start, len(dest)):
            if bit == start:
                self.c.put(phase, dest[bit], s.buffer(src[bit], -1))
            else:
                lower = tuple((lane, 1) for lane in src[start:bit])
                self.c.put(phase, dest[bit], (*s.buffer(src[bit]),
                    *s.match((*lower, (src[bit], -1))), *s.match((*lower, (src[bit], 1)), -1)))

    def eq(self, phase, flag, left, right):
        # Scratch scalar EQUAL is safe across VX micro-ops.
        for bit, (a, b) in enumerate(zip(left, right)):
            self.c.put(phase, s.EQUAL + bit, s.xor(a, b, -1))
        next_phase = phase + '_EQ'
        self.c.route(phase, next_phase)
        self.c.put(next_phase, flag, s.match(tuple((s.WORK + s.EQUAL + bit, 1) for bit in range(len(left)))))
        return next_phase

    def addr(self, phase, space, index, *, write=False):
        self.c.address(phase, s.WRITE_ADDR if write else s.READ_ADDR, self.addr_sources(space, index))

    def addr_sources(self, space, index):
        source = list(self.resize((s.WORK + s.ZERO, s.WORK + s.ZERO, *index)))
        for bit in range(28, 32):
            source[bit] = self.const(ADDRESS[space])[bit]
        return tuple(source)

    def pending(self):
        if self.pending_address is None:
            self.pending_address = self.alloc('pending_write_address', 33)
            self.pending_word = self.alloc('pending_write_word', 32)

    def read(self, phase, space, index, target):
        self.addr(phase, space, index)
        self.c.route(phase, target)
        self.reads.add(target)

    def write(self, phase, space, index, payload, target):
        self.pending()
        self.move(phase, self.pending_address, (*self.addr_sources(space, index), s.WORK + s.ZERO))
        self.write_payload(phase, payload, target)

    def write_payload(self, phase, payload, target, *, existing=False):
        self.pending()
        if existing:
            self.move(phase, self.pending_address, s.bits(s.WORK + s.READ_ADDR, 33))
        self.move(phase, self.pending_word, self.resize(payload))
        code = self.store_returns.setdefault(target, len(self.store_returns))
        self.fix(phase, self.store_flag, code)
        self.c.route(phase, 'VX_STORE_LOOKUP')

    def finish_stores(self):
        c = self.c
        c.address('VX_STORE_LOOKUP', s.READ_ADDR, self.state(self.pending_address)[:32])
        c.route('VX_STORE_LOOKUP', 'VX_STORE_READ')
        self.reads.add('VX_STORE_READ')
        c.move('VX_STORE_READ', s.OLD_HANDLE, s.bits(s.READBACK + s.WORD, s.HANDLE_BITS))
        c.move('VX_STORE_READ', s.OLD_HIT, (s.READBACK + s.WORD + s.HANDLE_BITS,))
        c.address('VX_STORE_READ', s.WRITE_ADDR, self.state(self.pending_address)[:32])
        c.move('VX_STORE_READ', s.WRITE_WORD, self.state(self.pending_word))
        c.route('VX_STORE_READ', s.STORE)
        # Every published write passes the public retirement/invalidation event.
        c.routes[s.STORE] = {s.RETIRE: None}
        c.routes[s.RETIRE] = {'VX_STORE_RETURN': None}
        for target, code in self.store_returns.items():
            flag = self.alloc('store_case_' + target, 1)[0]
            c.put('VX_STORE_RETURN', flag, s.match(tuple((lane, s.sign(code, bit))
                for bit, lane in enumerate(self.state(self.store_flag)))))
            c.route('VX_STORE_RETURN', (target, flag))
        # Scalar SW and register writes keep their existing retirement paths.
        c.route('VX_STORE_RETURN', s.COMMIT)
        self.fix(s.FETCH, self.store_flag, 31)
        for phase in c.phases:
            if phase.startswith('MERGE_'):
                self.fix(phase, self.store_flag, 31)

    def emit(self, phase, family, payload):
        self.streams.add(phase)
        self.fix(phase, self.fields['stream_valid'], 1)
        self.fix(phase, self.fields['stream_family'], family)
        self.move(phase, self.fields['stream_payload'], self.resize(payload, 16))


def program():
    configure()
    extension = SimpleNamespace(M_RETURN=m.M_RETURN + RETURNS, add_m_program=m.add_m_program)
    c = s.program(extension)
    b = Builder(c)
    state = lambda base, n=32: s.bits(s.WORK + base, n)
    zero, one = s.WORK + s.ZERO, s.WORK + s.ONE
    cfg = b.alloc('configuration_rd', 32)
    elem = b.alloc('element', 7)
    tile = b.alloc('tile', 3)
    waddr = b.alloc('wrom_word', 16)
    wbase = b.alloc('wrom_base', 16)
    acc = b.alloc('accumulator', 32)
    result = b.alloc('result', 32)
    result_index = b.alloc('result_index', 10)
    result_keep = b.alloc('result_keep', 1)
    result_signed = b.alloc('result_signed', 1)
    result_next = b.alloc('result_next', 3)
    b.result_targets = {'VX_ROW_NEXT': 0}
    chunk_current = b.alloc('chunk_current', 2)
    vmx = b.alloc('vmx', 32)
    vsum = b.alloc('vsum', 32)
    varg = b.alloc('varg', 32)
    vbest = b.alloc('vbest', 32)
    vbest_valid = b.alloc('vbest_valid', 1)
    flag = b.alloc('flag', 1)[0]
    auxflag = b.alloc('auxiliary_flag', 1)[0]
    # The eight products and their small prefix adder share scratch across every tile.
    prod = b.alloc('tile_products', 96)
    px = b.alloc('tile_xor', 96)
    pp = b.alloc('tile_propagate', 96)
    pg = b.alloc('tile_generate', 96)
    act = b.alloc('tile_activations', 64)
    weights = b.alloc('tile_weights', 32)
    cfgs, es, ts = b.state(cfg), b.state(elem), b.state(tile)
    r1, r2 = state(s.RS1), state(s.RS2)
    raw = state(s.RAW)
    z32 = (zero,) * 32
    # A two-phase PC+4 carry tree replaces the generic eight-phase scalar adder.
    c.routes['PREP_SEQUENTIAL'] = {'VX_PC4_FINISH': None}
    c.effects['PREP_SEQUENTIAL'] = {}
    for bit in range(3, 32):
        c.put('PREP_SEQUENTIAL', s.P + bit, s.match(tuple((lane, 1) for lane in state(s.PC)[2:bit])))
    c.move('VX_PC4_FINISH', s.NEXT_PC, state(s.PC))
    c.put('VX_PC4_FINISH', s.NEXT_PC + 2, s.buffer(s.WORK + s.PC + 2, -1))
    for bit in range(3, 32):
        c.put('VX_PC4_FINISH', s.NEXT_PC + bit, s.xor(s.WORK + s.PC + bit, s.WORK + s.P + bit))
    c.route('VX_PC4_FINISH', 'DECODE')
    # Add raw Custom-2 decoding and the third scalar register input x[rd].
    for funct, name in enumerate(VX_OPS):
        op_lane = 1600 + len(c.op_flag)
        c.op_flag[name] = op_lane
        c.reserve_state(op_lane)
        requirements = tuple((raw[bit], s.sign(0x5b | funct << 12, bit))
                             for bit in (*range(7), *range(12, 15), *range(25, 32)))
        c.put('DECODE', op_lane, s.match(requirements))
        phase = 'VX_PREP_' + name
        c.route('CLASSIFY', (phase, op_lane))
        c.address(phase, s.READ_ADDR, raw[15:20], reg=True)
        c.route(phase, 'READ_RS1')
    c.put('READ_RS2', flag, s.threshold(tuple((s.WORK + c.op_flag[name], 1) for name in VX_OPS), len(VX_OPS) - 1))
    c.route('READ_RS2', ('VX_PREP_RD', flag))
    c.address('VX_PREP_RD', s.READ_ADDR, raw[7:12], reg=True)
    c.route('VX_PREP_RD', 'VX_READ_RD')
    b.reads.add('VX_READ_RD')
    b.move('VX_READ_RD', cfg, s.bits(s.READBACK))
    c.route('VX_READ_RD', *((('VX_EXEC_' + name), lane) for name, lane in c.op_flag.items() if name in VX_OPS))
    # VGET reads the selected tracker and writes an ordinary scalar register.
    for sel, source in enumerate((b.state(vmx), b.state(vsum), b.state(varg))):
        select = b.alloc(f'get_select_{sel}', 1)[0]
        c.put('VX_EXEC_VGET', select, s.match(tuple((r1[bit], s.sign(sel, bit)) for bit in range(2))))
        c.route('VX_EXEC_VGET', (f'VX_GET_{sel}', select))
        c.move(f'VX_GET_{sel}', s.RD_WORD, source)
        c.route(f'VX_GET_{sel}', 'RD_CHECK')
    c.route('VX_EXEC_VGET', 'VX_GET_BITLEN_SIGN')
    c.branch('VX_GET_BITLEN_SIGN', vmx[31], 'VX_GET_BITLEN_ZERO', 'VX_GET_BITLEN')
    c.fix('VX_GET_BITLEN_ZERO', s.RD_WORD, 0)
    c.route('VX_GET_BITLEN_ZERO', 'RD_CHECK')
    # Binary bits of the 32-entry thermometer bit_length(vmx), sharing all predicates.
    c.fix('VX_GET_BITLEN', s.RD_WORD, 0)
    for bit in range(6):
        transitions = [(i, ((i >> bit) & 1) - (((i - 1) >> bit) & 1)) for i in range(1, 33)]
        expression = []
        for i, delta in transitions:
            if delta:
                expression.extend(s.match(tuple((lane, -1) for lane in b.state(vmx)[i - 1:]), -delta))
        expression.extend(s.constant(sum(delta for i, delta in transitions) - 1))
        c.put('VX_GET_BITLEN', s.RD_WORD + bit, tuple(expression))
    c.route('VX_GET_BITLEN', 'RD_CHECK')
    # VPUT stores a pair.  Only the low ten configuration bits address it.
    c.put('VX_EXEC_VPUT', flag, s.buffer(cfgs[9]))
    c.branch('VX_EXEC_VPUT', flag, 'VX_PUT_ROPE0', 'VX_PUT_VM0')
    for space, phase in (('vm', 'VX_PUT_VM'), ('rope', 'VX_PUT_ROPE')):
        index = (zero, *cfgs[:9]) if space == 'vm' else (zero, *cfgs[:2])
        b.write(phase + '0', space, index, r1, phase + '1')
        index1 = (one, *index[1:])
        b.write(phase + '1', space, index1, r2, s.COMMIT)
    # VMAT: address rowchunk = 3*base+start, then nine packed words per chunk.
    b.fix('VX_EXEC_VMAT', elem, 0)
    c.move('VX_EXEC_VMAT', m.M_DIVISOR, r1)
    c.add('VX_EXEC_VMAT', b.resize(r1[:13]), b.resize((zero, *r1[:13])), 'VX_WBASE')
    c.add('AFTER_VX_WBASE', state(s.SUM), b.resize(r1[19:21]), 'VX_WSTART')
    b.move('AFTER_VX_WSTART', wbase, state(s.SUM)[:16])
    c.add('AFTER_VX_WSTART', state(s.SUM), (zero, zero, zero, *state(s.SUM)[:29]), 'VX_ROW')
    b.move('AFTER_VX_ROW', wbase, state(s.SUM)[:16])
    c.route('AFTER_VX_ROW', 'VX_ROW_INIT')
    b.fix('VX_ROW_INIT', acc, 0)
    b.fix('VX_ROW_INIT', tile, 0)
    b.move('VX_ROW_INIT', chunk_current, cfgs[15:17])
    b.move('VX_ROW_INIT', waddr, b.state(wbase))
    c.route('VX_ROW_INIT', 'VX_TILE_QUERY')
    # head0/1 read the two packed VA words; head2 reads eight raw WROM nibbles.
    va_index = (zero, *ts, *b.state(chunk_current))
    b.addr('VX_TILE_QUERY', 'va', va_index)
    c.route('VX_TILE_QUERY', 'VX_TILE_READ')
    b.reads.add('VX_TILE_READ')
    b.move('VX_TILE_READ', act[:32], s.bits(s.READBACK))
    b.move('VX_TILE_READ', act[32:], s.bits(EXTRA_READBACK[0]))
    for base in EXTRA_READBACK:
        c.move('VX_TILE_READ', base, s.bits(base))
    # The stored nibble u represents weight u-8: flip its sign bit.
    for bit, lane in enumerate(weights):
        c.put('VX_TILE_READ', lane, s.buffer(EXTRA_READBACK[1] + bit, -1 if bit % 4 == 3 else 1))
    b.fix('VX_TILE_READ', prod, 0)
    c.route('VX_TILE_READ', 'VX_MUL_BIT0_INIT')
    # Parallel signed 8x4 products with six-bit local carry predicates.
    for wb in range(4):
        name = f'VX_MUL_BIT{wb}'
        for lane in range(8):
            activation = b.state(act[lane * 8:lane * 8 + 8])
            wbit = b.state(weights)[lane * 4 + wb]
            start, end = lane * 12, lane * 12 + 12
            for bit in range(12):
                av = zero if bit < wb else activation[min(bit - wb, 7)]
                c.put(name + '_MASK', px[start + bit], s.match(((av, 1), (wbit, 1)), -1 if wb == 3 else 1))
            fast_add(c, entry=name + '_INIT',
                finish=f'VX_MUL_BIT{wb + 1}_MASK' if wb < 3 else 'VX_REDUCE_0_INIT',
                left=b.state(prod[start:end]), right=b.state(px[start:end]),
                destination=prod[start:end], partial=px[start:end],
                carry0=pg[start:end], carry1=pp[start:end], block_carries=result[lane:lane + 1],
                state=b.state, logic=s, carry_in=one if wb == 3 else zero)
        c.route(name + '_MASK', name + '_INIT')
    c.routes['VX_TILE_READ'] = {'VX_MUL_BIT0_MASK': None}
    groups, width = 8, 12
    for level in range(3):
        out_width = width + 1
        phase = f'VX_REDUCE_{level}'
        for lane in range(groups // 2):
            left = b.resize(b.state(prod[lane * 2 * width:(lane * 2 + 1) * width]), out_width, True)
            right = b.resize(b.state(prod[(lane * 2 + 1) * width:(lane * 2 + 2) * width]), out_width, True)
            start, end = lane * out_width, (lane + 1) * out_width
            fast_add(c, entry=phase + '_INIT',
                finish=f'VX_REDUCE_{level + 1}_INIT' if level < 2 else 'VX_TILE_ACC',
                left=left, right=right, destination=prod[start:end], partial=px[start:end],
                carry0=pg[start:end], carry1=pp[start:end], block_carries=result[2 * lane:2 * lane + 2],
                state=b.state, logic=s, chunk=5)
        groups, width = groups // 2, out_width
    c.add('VX_TILE_ACC', b.state(acc), b.resize(b.state(prod[:15]), signed=True), 'VX_ACC')
    b.move('AFTER_VX_ACC', acc, state(s.SUM))
    b.inc('AFTER_VX_ACC', tile)
    b.inc('AFTER_VX_ACC', waddr)
    c.put('AFTER_VX_ACC', flag, s.match(tuple((lane, 1) for lane in ts)))
    c.branch('AFTER_VX_ACC', flag, 'VX_SCALE_QUERY', 'VX_TILE_QUERY')
    b.read('VX_SCALE_QUERY', 'wrom', b.state(waddr), 'VX_SCALE_READ')
    # Scale is exactly the low sixteen bits of the ninth packed WROM word.
    c.move('VX_SCALE_READ', s.RS1, b.state(acc))
    c.move('VX_SCALE_READ', s.RS2, (*s.bits(s.READBACK, 16), *((zero,) * 16)))
    # Reuse the scalar multiplier with a dedicated continuation, retaining operands in cfg fields.
    c.fix('VX_SCALE_READ', c.op_flag['MUL'], 1, 1)
    c.put('VX_SCALE_READ', flag, s.threshold(((cfgs[12], -1), (b.state(chunk_current)[1], 1)), 1))
    c.branch('VX_SCALE_READ', flag, 'VX_SCALE_INIT', 'VX_CHUNK_NEXT')
    b.inc('VX_CHUNK_NEXT', waddr)
    b.inc('VX_CHUNK_NEXT', chunk_current)
    b.fix('VX_CHUNK_NEXT', tile, 0)
    c.route('VX_CHUNK_NEXT', 'VX_TILE_QUERY')
    c.move('VX_SCALE_INIT', m.M_LOW, state(s.RS2))
    c.fix('VX_SCALE_INIT', m.M_HIGH, 0)
    c.fix('VX_SCALE_INIT', m.M_COUNT, 16, 5)
    c.route('VX_SCALE_INIT', 'M_MUL_PICK')
    # M_MUL_LOW is shared with scalar MUL; a live VX flag selects the scaling continuation.
    active = b.alloc('active', 1)[0]
    c.fix('VX_EXEC_VMAT', active, 1, 1)
    c.routes['M_MUL_LOW'] = {'VX_SCALE_RESULT': active, 'RD_CHECK': -active}
    b.move('VX_SCALE_RESULT', result, (*state(m.M_LOW)[16:], *state(m.M_HIGH)[:16]))
    c.route('VX_SCALE_RESULT', 'VX_RESULT_SIGN31')
    c.put('VX_RESULT_SIGN31', result[31], s.buffer(b.state(result)[30]))
    for bit in range(6):
        c.put('VX_RESULT_SIGN31', result_index[bit], s.threshold(((state(m.M_DIVISOR)[13 + bit], 1), (es[bit], 1)), 1))
    b.move('VX_RESULT_SIGN31', result_index[6:], cfgs[:4])
    b.move('VX_RESULT_SIGN31', result_keep, (cfgs[14],))
    b.fix('VX_RESULT_SIGN31', result_signed, 0)
    b.fix('VX_RESULT_SIGN31', result_next, 0)
    c.put('VX_RESULT_SIGN31', flag, s.buffer(cfgs[13]))
    c.branch('VX_RESULT_SIGN31', flag, 'VX_ARGMAX', 'VX_ABS_TEST')
    # Track max absolute value.  The generic adder handles INT31 negation exactly.
    c.put('VX_ABS_TEST', flag, s.match(((b.state(result)[31], 1), (b.state(result_signed)[0], -1))))
    c.branch('VX_ABS_TEST', flag, 'VX_ABS_NEG', 'VX_ABS_POS')
    c.add('VX_ABS_NEG', z32, b.state(result), 'VX_ABS', subtract=True)
    c.move('VX_ABS_POS', m.M_LOW, b.state(result))
    c.route('VX_ABS_POS', 'VX_MAX_COMPARE')
    c.move('AFTER_VX_ABS', m.M_LOW, state(s.SUM))
    c.route('AFTER_VX_ABS', 'VX_MAX_COMPARE')
    c.add('VX_MAX_COMPARE', b.state(vmx), state(m.M_LOW), 'VX_MAX', subtract=True)
    c.put('VX_MAX_COMPARE', auxflag, s.match(((b.state(result_keep)[0], -1), *((lane, -1) for lane in es))))
    c.put('VX_MAX_COMPARE', s.SIGN_DIFFER, s.xor(b.state(vmx)[31], state(m.M_LOW)[31]))
    # New maximum if current < magnitude, or first element and keep=0.
    # Every tracker candidate fits signed32, including absolute(INT31_MIN).
    # Signed comparison also handles a negative maximum retained from scores.
    c.put('AFTER_VX_MAX', s.PENDING, s.buffer(s.WORK + s.SIGN_DIFFER))
    c.route('AFTER_VX_MAX', 'VX_MAX_LESS')
    c.put('VX_MAX_LESS', flag, s.xor(s.WORK + s.G + 31, s.WORK + s.PENDING, -1))
    c.route('VX_MAX_LESS', 'VX_MAX_DECIDE')
    c.put('VX_MAX_DECIDE', flag, s.threshold(((s.WORK + flag, 1), (s.WORK + auxflag, 1)), 1))
    c.branch('VX_MAX_DECIDE', flag, 'VX_MAX_SET', 'VX_RESULT_WRITE')
    b.move('VX_MAX_SET', vmx, state(m.M_LOW))
    c.route('VX_MAX_SET', 'VX_RESULT_WRITE')
    c.route('VX_RESULT_WRITE', 'VX_RESULT_STORE')
    b.write('VX_RESULT_STORE', 'vm', b.state(result_index), b.state(result), 'VX_RESULT_NEXT')
    # VMAT argmax uses signed strict greater-than; ties retain the previous row.
    c.add('VX_ARGMAX', b.state(vbest), b.state(result), 'VX_ARGCOMPARE', subtract=True)
    c.put('VX_ARGMAX', s.SIGN_DIFFER, s.xor(b.state(vbest)[31], b.state(result)[31]))
    c.put('VX_ARGMAX', auxflag, s.match(((cfgs[14], -1), *((lane, -1) for lane in es))))
    c.put('AFTER_VX_ARGCOMPARE', flag, s.xor(s.WORK + s.G + 31, s.WORK + s.SIGN_DIFFER, -1))
    c.route('AFTER_VX_ARGCOMPARE', 'VX_ARG_DECIDE')
    c.put('VX_ARG_DECIDE', flag, s.threshold(((s.WORK + flag, 1), (s.WORK + auxflag, 1)), 1))
    c.branch('VX_ARG_DECIDE', flag, 'VX_ARG_SET', 'VX_ROW_NEXT')
    b.move('VX_ARG_SET', vbest, b.state(result))
    b.move('VX_ARG_SET', varg, (*es[:6], *state(m.M_DIVISOR)[:13], *((zero,) * 13)))
    b.fix('VX_ARG_SET', vbest_valid, 1)
    c.route('VX_ARG_SET', 'VX_ROW_NEXT')
    b.inc('VX_ROW_NEXT', elem)
    c.route('VX_ROW_NEXT', 'VX_ROW_LAST')
    last2 = b.eq('VX_ROW_LAST', flag, b.state(elem), cfgs[4:11])
    c.branch(last2, flag, 'VX_FINISH', 'VX_WBASE_NEXT')
    # k1 increments one rowchunk (9words); k3 will select a 27-word step.
    c.put('VX_WBASE_NEXT', flag, s.buffer(cfgs[12]))
    c.branch('VX_WBASE_NEXT', flag, 'VX_WBASE_STEP27', 'VX_WBASE_STEP9')
    c.add('VX_WBASE_STEP9', b.resize(b.state(wbase)), b.const(9), 'VX_WSTEP')
    c.add('VX_WBASE_STEP27', b.resize(b.state(wbase)), b.const(27), 'VX_WSTEP')
    b.move('AFTER_VX_WSTEP', wbase, state(s.SUM)[:16])
    c.route('AFTER_VX_WSTEP', 'VX_ROW_INIT')
    c.fix('VX_FINISH', active, 0, 1)
    c.route('VX_FINISH', s.COMMIT)
    # Quantization reads actual VM words and packs each output byte by a model
    # read/modify/write.  The right operand remains the raw fifteen-bit factor.
    c.route('VX_EXEC_VQ', 'VX_Q_QUERY')
    b.fix('VX_EXEC_VQ', elem, 0)
    c.move('VX_EXEC_VQ', m.M_DIVISOR, r1)
    c.fix('VX_EXEC_VQ', s.RS2 + 15, 0, 17)
    for bit in range(6):
        c.put('VX_Q_QUERY', s.A + bit, s.threshold(((cfgs[4 + bit], 1), (es[bit], 1)), 1))
    c.route('VX_Q_QUERY', 'VX_Q_VM_QUERY')
    b.read('VX_Q_VM_QUERY', 'vm', (*state(s.A, 6), *cfgs[:4]), 'VX_Q_VM_READ')
    c.move('VX_Q_VM_READ', s.A, s.bits(s.READBACK))
    c.move('VX_Q_VM_READ', s.B, cfgs[19:24])
    c.fix('VX_Q_VM_READ', s.SHIFT_LEFT, 0, 1)
    c.fix('VX_Q_VM_READ', s.SHIFT_ARITH, 1, 1)
    c.route('VX_Q_VM_READ', 'SHIFT_TEST_0')
    c.route('SHIFT_FINISH', ('VX_Q_CLAMP', c.op_flag['VQ']))
    mathunit.emit_clamp16(c, 'VX_Q_CLAMP', state(s.RD_WORD), s.bits(s.RS1, 16), s)
    c.move('VX_Q_CLAMP', s.RS1 + 16, (state(s.RD_WORD)[31],) * 16)
    c.fix('VX_Q_CLAMP', c.op_flag['MUL'], 1, 1)
    c.route('VX_Q_CLAMP', 'VX_SCALE_INIT')
    c.route('M_MUL_LOW', ('VX_Q_PRODUCT', c.op_flag['VQ']))
    b.move('VX_Q_PRODUCT', result, (*state(m.M_LOW)[16:], *state(m.M_HIGH)[:16]))
    c.route('VX_Q_PRODUCT', 'VX_Q_ROUND')
    mathunit.emit_vq_round(c, entry='VX_Q_ROUND', finish='VX_Q_ADDRESS', product=b.state(result),
        rounded=prod[8:24], dst=prod[:8], flag=auxflag, state=b.state, logic=s)
    # VA offsets are aligned to the power-of-two operation length, hence OR.
    for bit in range(9):
        c.put('VX_Q_ADDRESS', s.A + bit, s.threshold(((state(m.M_DIVISOR)[bit], 1),
            (es[bit] if bit < 7 else zero, 1)), 1))
    c.put('VX_Q_ADDRESS', flag, s.match(((cfgs[17], -1), (cfgs[18], -1))))
    c.branch('VX_Q_ADDRESS', flag, 'VX_Q_VA_ADDRESS', 'VX_Q_CACHE_ADDRESS')
    b.addr('VX_Q_VA_ADDRESS', 'va', state(s.A)[2:9])
    c.move('VX_Q_VA_ADDRESS', s.B, state(s.A, 2))
    c.route('VX_Q_VA_ADDRESS', 'VX_Q_WRITE_READ')
    b.reads.add('VX_Q_WRITE_READ')
    # Cache rows are off +32*(i//8), each with eight bytes.
    c.add('VX_Q_CACHE_ADDRESS', (zero, zero, zero, *state(m.M_DIVISOR)[:10], *((zero,) * 19)),
          (*es[:3], *((zero,) * 5), *es[3:6], *((zero,) * 21)), 'VX_GENERAL')
    c.move('AFTER_VX_GENERAL', s.B, state(s.SUM, 2))
    c.put('AFTER_VX_GENERAL', flag, s.buffer(cfgs[18]))
    c.branch('AFTER_VX_GENERAL', flag, 'VX_Q_VC_ADDRESS', 'VX_Q_KC_ADDRESS')
    for space in ('kc', 'vc'):
        b.read('VX_Q_' + space.upper() + '_ADDRESS', space, state(s.SUM)[2:13], 'VX_Q_WRITE_READ')
    c.move('VX_Q_WRITE_READ', s.EQUAL, s.bits(s.READBACK))
    c.move('VX_Q_WRITE_READ', s.READ_ADDR, state(s.READ_ADDR, 33))
    c.fix('VX_Q_WRITE_READ', s.READ_MARK, 1, 1)
    for offset in range(4):
        case = s.EXTRACT_FLAG + offset
        c.put('VX_Q_WRITE_READ', case, s.match(tuple((lane, s.sign(offset, bit))
            for bit, lane in enumerate(state(s.B, 2)))))
        phase = f'VX_Q_MERGE_{offset}'
        c.route('VX_Q_WRITE_READ', (phase, case))
        payload = (*state(s.EQUAL)[:offset * 8], *b.state(prod[:8]), *state(s.EQUAL)[offset * 8 + 8:])
        b.write_payload(phase, payload, 'VX_Q_NEXT', existing=True)
    b.inc('VX_Q_NEXT', elem)
    c.route('VX_Q_NEXT', 'VX_Q_LAST')
    last = b.eq('VX_Q_LAST', flag, es, cfgs[10:17])
    c.branch(last, flag, s.COMMIT, 'VX_Q_QUERY')
    # Reduction processes one actual packed activation byte at a time.  The
    # bounded eight-bit square table is a datapath LUT, shared by every workload.
    b.fix('VX_EXEC_VRED', elem, 0)
    b.fix('VX_EXEC_VRED', acc, 0)
    b.fix('VX_EXEC_VRED', vsum, 0)
    c.put('VX_EXEC_VRED', flag, s.match(tuple((lane, -1) for lane in cfgs[:7])))
    c.branch('VX_EXEC_VRED', flag, s.COMMIT, 'VX_RED_QUERY')
    b.read('VX_RED_QUERY', 'va', (*es[2:6], *cfgs[8:11]), 'VX_RED_READ')
    c.move('VX_RED_READ', s.EQUAL, s.bits(s.READBACK))
    for offset in range(4):
        case = s.EXTRACT_FLAG + offset
        c.put('VX_RED_READ', case, s.match(tuple((lane, s.sign(offset, bit)) for bit, lane in enumerate(es[:2]))))
        phase = f'VX_RED_BYTE_{offset}'
        c.route('VX_RED_READ', (phase, case))
        source = state(s.EQUAL + offset * 8, 8)
        c.move(phase, s.RS1, b.resize(source, signed=True))
        c.branch(phase, cfg[7], 'VX_RED_SUM', 'VX_RED_SQUARE')
    mathunit.emit_lut(c, 'VX_RED_SQUARE', state(s.RS1, 8), prod[:15],
        tuple(((value & 127) - (value & 128)) ** 2 for value in range(256)), s)
    c.route('VX_RED_SQUARE', 'VX_RED_SQUARE_ADD')
    c.add('VX_RED_SQUARE_ADD', b.state(acc), b.resize(b.state(prod[:15])), 'VX_VRED')
    c.add('VX_RED_SUM', b.state(acc), state(s.RS1), 'VX_VRED')
    b.move('AFTER_VX_VRED', acc, state(s.SUM))
    b.inc('AFTER_VX_VRED', elem)
    c.route('AFTER_VX_VRED', 'VX_RED_LAST')
    last = b.eq('VX_RED_LAST', flag, es, cfgs[:7])
    c.branch(last, flag, 'VX_RED_DONE', 'VX_RED_QUERY')
    b.move('VX_RED_DONE', vsum, b.state(acc))
    c.route('VX_RED_DONE', s.COMMIT)
    ports = dict(locals())
    ports['logic'], ports['m'] = s, m
    ops.add_ops(b, ports)
    # The default scalar continuation has no condition.  True VX conditions get
    # the same phase score plus a positive predicate margin and win uniquely.
    c.routes['M_MUL_LOW'] = {'RD_CHECK': None, 'VX_SCALE_RESULT': c.op_flag['VMAT'], 'VX_Q_PRODUCT': c.op_flag['VQ']}
    for target, code in b.result_targets.items():
        select = s.EXTRACT_FLAG + code
        c.put('VX_RESULT_NEXT', select, s.match(tuple((lane, s.sign(code, bit)) for bit, lane in enumerate(b.state(result_next)))))
        c.route('VX_RESULT_NEXT', (target, select))
    b.finish_stores()
    for name in RETURNS:
        phase = 'AFTER_' + name
        if phase not in c.routes:
            c.route(phase, 'SET_ILLEGAL')
    # Geometry is checked before any model is allocated or written.
    s.DISPATCH = s.ACTION + len(c.phases)
    validate_readback_contract(c, b)
    c.vx = b
    return c


def compile_program(c):
    normalize_expressions(c)
    raw = compile_circuit(c, s)
    compiled = deduplicate(raw)
    c.vx.unmerged_rows = raw[-1]
    return compiled


class Writer:
    """Patch only new MTP tensors while reusing the scalar source-byte audit."""
    def __init__(self, inner, circuit, compiled, packed, tiny_ff=None, tiny_bf16=False):
        self.inner, self.circuit, self.compiled, self.packed, self.tiny_ff = inner, circuit, compiled, packed, tiny_ff
        self.tiny_bf16 = tiny_bf16

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __setattr__(self, name, value):
        if name in ('inner', 'circuit', 'compiled', 'packed', 'tiny_ff', 'tiny_bf16'):
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    def add_uint32(self, name, value):
        if not self.packed and name == 'qwen35.attention.head_count':
            value = 3
        if not self.packed and self.tiny_ff and name == 'qwen35.feed_forward_length':
            value = self.tiny_ff
        return self.inner.add_uint32(name, value)

    def add_name(self, name):
        return self.inner.add_name('Bonsai with stock MTP RV32/VX' if self.packed else 'Tiny stock MTP RV32/VX')

    def add_tensor(self, name, data, **kwargs):
        if not name.startswith(f'blk.{64 if self.packed else 1}.'):
            return self.inner.add_tensor(name, data, **kwargs)
        ids = self.compiled[0]
        if name.endswith('.attn_q.weight'):
            if data.dtype != np.float32:
                raise ValueError('VX query tensor must use F32')
            data = data.copy()
            reads = self.circuit.vx.reads
            for phase in reads:
                for bit in range(57):
                    data[s.HEAD + bit, s.ACTION + ids[phase]] = 128
            # Two additional query heads share KV head zero.  Their gate is open at tile reads only.
            sentinel = s.ROTARY + 33 * s.REPEATS
            for head in (1, 2):
                start = 2 * s.HEAD * head
                for bit in range(32):
                    for phase, op in ids.items():
                        data[start + s.HEAD + bit, s.ACTION + op] = 128 if phase == 'VX_TILE_READ' else -128
                for bit in range(33):
                    for rep in range(s.REPEATS):
                        row = start + s.ROTARY + bit * s.REPEATS + rep
                        if head == 1:
                            source = s.WORK + (s.ONE if bit == 2 else s.READ_ADDR + bit)
                        else:
                            lanes = self.circuit.vx.fields['wrom_word']
                            source = (s.WORK + s.ZERO if bit < 2 else s.WORK + lanes[bit - 2] if bit < len(lanes) + 2
                                else s.WORK + (s.ONE if ADDRESS['wrom'] >> bit & 1 else s.ZERO))
                        data[row, source] = 1
                data[start + sentinel, s.WORK + s.READ_MARK] = 11.125
        elif name.endswith('.attn_output.weight'):
            data = data.copy()
            for head, base in enumerate(EXTRA_READBACK, 1):
                for bit in range(32):
                    row, col = base + bit, s.HEAD * head + bit
                    if kwargs.get('raw_dtype') == s.GGMLQuantizationType.Q4_0:
                        blocks = data.reshape(data.shape[0], -1, 18)
                        target = blocks[row, col // 32]
                        target[:2] = np.array([0.25], dtype='<f2').view(np.uint8)
                        target[2 + col % 16] ^= (8 ^ 12) << (4 if col % 32 >= 16 else 0)
                    else:
                        data[row, col] = 1
        elif name.endswith('.nextn.shared_head_norm.weight'):
            data = data.copy()
            for base in EXTRA_READBACK:
                data[base:base + 32] = 1
        elif self.tiny_bf16 and name.endswith(('.ffn_gate.weight', '.ffn_up.weight', '.ffn_down.weight')):
            if data.dtype != np.float32 or np.any(data.view(np.uint32) & 65535):
                raise ValueError(f'Inexact BF16 conversion in {name}')
            data = (data.view(np.uint32) >> 16).astype('<u2').view(np.uint8)
            kwargs['raw_dtype'] = s.GGMLQuantizationType.BF16
        return self.inner.add_tensor(name, data, **kwargs)


def export(output, source, circuit, compiled, *, tiny_ff=None, ffn_padding=None, tiny_bf16=False):
    factory, add_block = s.GGUFWriter, s.add_block
    s.GGUFWriter = lambda *a, **k: Writer(factory(*a, **k), circuit, compiled, bool(source), tiny_ff, tiny_bf16)
    s.add_block = lambda writer, index, width, ff, nh, nkv, vocab, packed, co, ci, **kwargs: add_block(
        writer, index, width, ff if source or not tiny_ff else tiny_ff, nh if source else 3,
        nkv, vocab, packed, co, ci, query_f32=True)
    try:
        s.export(output, source, circuit, compiled, query_f32=True, ffn_padding=ffn_padding)
    finally:
        s.GGUFWriter, s.add_block = factory, add_block
    manifest = output.with_suffix('.manifest.json')
    report = json.loads(manifest.read_text())
    if tiny_bf16:
        report['mtp_ffn_type'] = 'BF16'
    report['read_tokens'] = sorted(set(report['read_tokens']) | {compiled[0][phase] for phase in circuit.vx.reads})
    report.update(vx_memory_address_bases=ADDRESS, vx_feedback_lanes=circuit.vx.fields,
        vx_implemented_operations=VX_OPS,
        vx_read_tokens=[compiled[0][phase] for phase in sorted(circuit.vx.reads)],
        vx_attention_ports={'head0_readback_word':s.bits(s.READBACK), 'head0_query':s.bits(s.READ_ADDR,33),
            'head1_readback_word':s.bits(EXTRA_READBACK[0]), 'head1_query':'head0_query with address bit2 set',
            'head2_readback_word':s.bits(EXTRA_READBACK[1]), 'head2_query_word':circuit.vx.fields['wrom_word'],
            'head2_query_base':ADDRESS['wrom'], 'head2_query_word_shift':2,
            'extra_heads_active_phase':'VX_TILE_READ'},
        vx_parallel_lanes=8, vx_multiply_widths=[8, 4, 12],
        mtp_ffn_rows_before_exact_deduplication=circuit.vx.unmerged_rows)
    manifest.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path)
    parser.add_argument('--bonsai', type=Path)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--tiny-ff', type=int, help='explicit tiny FFN row allocation for the packed candidate geometry')
    parser.add_argument('--ffn-padding', type=int, help='zero-extend original packed language FFNs and allocate this MTP width')
    parser.add_argument('--tiny-bf16', action='store_true', help='use exactly representable BF16 FFN matrices in the tiny candidate')
    parser.add_argument('--scalar-regression', action='store_true',
                        help='export only the unchanged scalar circuit with exact FFN row deduplication')
    args = parser.parse_args()
    circuit = s.program(m) if args.scalar_regression else program()
    compiled = deduplicate(circuit.compile()) if args.scalar_regression else compile_program(circuit)
    if args.plan:
        if args.scalar_regression:
            print(json.dumps({'phases': compiled[0], 'ffn_rows': compiled[-1]}, indent=2))
            raise SystemExit
        print(json.dumps({'phases': compiled[0], 'ffn_rows': compiled[-1],
            'unmerged_rows': circuit.vx.unmerged_rows, 'feedback_lanes': len(circuit.persistent),
            'remaining_lanes': len(circuit.vx.available), 'dispatch_end': max(col for row, col in compiled[1]) + 1,
            'vx_fields': circuit.vx.fields}, indent=2))
    elif args.output:
        if args.scalar_regression:
            s.export(args.output, args.bonsai, circuit, compiled, query_f32=True)
        else:
            export(args.output, args.bonsai, circuit, compiled, tiny_ff=args.tiny_ff,
                   ffn_padding=args.ffn_padding, tiny_bf16=args.tiny_bf16)
    else:
        parser.error('output is required')
