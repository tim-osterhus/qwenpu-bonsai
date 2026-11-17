"""Export the stock Qwen3.5 MTP implementation of the pinned scalar RV32IM/Zba profile."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil

from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
import numpy as np

from export_stock_mtp_loop import BONSAI_HASH, digest_file


WORD, HANDLE_BITS = 32, 24
HEAD, ROTARY, REPEATS, CONTEXT = 256, 64, 4, 512
TINY_WIDTH, TINY_VOCAB = 4352, 384
WORK, ACTION, DISPATCH, READBACK = 1792, 3520, 3840, 928
WRITE_ADDR, WRITE_WORD, HANDLE_IN, ENTRY = 64, 128, 160, 192
READ_ADDR, READ_MARK, OLD_HANDLE, OLD_HIT = 256, 289, 320, 344
PC, RAW, CAUSE, NEXT_PC, RD, RD_WORD = 352, 384, 416, 424, 456, 464
RD_VALID, BUS_VALID, BUS_WRITE, BRANCH_VALID, TAKEN = range(496, 501)
BUS_WIDTH = 501
BUS_ADDR, BUS_WORD, EXIT_CODE = 512, 544, 576
ONE, CARRY, ALIGNED, IN_RANGE, PENDING = range(608, 613)
PUTCHAR_FLAG = 615
RS1, RS2, A, B, X, P, G, SUM, EQUAL = range(624, 912, WORD)
ZERO, SHIFT_LEFT, SHIFT_ARITH, SIGN_DIFFER, SIGNED, INVERT, CMP_BRANCH = range(1456, 1463)
MEM_SIGNED = 1463
EXTRACT_FLAG = 1472
COMPARE_BRANCHES = ('BLT', 'BGE', 'BLTU', 'BGEU')
COMPARE_VALUES = ('SLT', 'SLTU', 'SLTI', 'SLTIU')
LOADS = {'LW': 4, 'LH': 2, 'LHU': 2, 'LB': 1, 'LBU': 1}
STORES = {'SW': 4, 'SH': 2, 'SB': 1}
M_OPERATIONS = ('MUL', 'MULH', 'MULHSU', 'MULHU', 'DIV', 'DIVU', 'REM', 'REMU')
CSR_READS = {'CSRRS': 2, 'CSRRC': 3, 'CSRRSI': 6, 'CSRRCI': 7}
CSR_STUBS = {0xc00: 0, 0xc01: 0, 0xc02: 0, 0xc80: 0, 0xc81: 0, 0xc82: 0, 0xf14: 0, 0x301: 0x40001100}
OPERATIONS = (
    'LUI', 'AUIPC', 'ADDI', 'ADD', 'SUB', *LOADS, *STORES, 'BEQ', 'BNE', *COMPARE_BRANCHES,
    'JAL', 'JALR', 'EBREAK', 'ECALL', 'FENCE', 'AND', 'OR', 'XOR', 'ANDI', 'ORI', 'XORI',
    'SLL', 'SRL', 'SRA', 'SLLI', 'SRLI', 'SRAI', *COMPARE_VALUES, 'SH1ADD', 'SH2ADD', 'SH3ADD', *CSR_READS,
)
RETURN_MODES = ('SEQUENTIAL', 'RD', 'LW', 'LH', 'LB', 'SW', 'SH', 'SB', 'BRANCH', 'JAL', 'JALR', 'COMPARE')

PUBLIC = ('SEED_MISS', 'SEED_WORD', 'START', 'FETCH', 'HALT', 'RETIRE', 'STORE', 'REGISTER', 'FAULT', 'COMMIT')
SEED_MISS, SEED_WORD, START, FETCH, HALT, RETIRE, STORE, REGISTER, FAULT, COMMIT = PUBLIC


def bits(base, count=WORD):
    return tuple(range(base, base + count))


def sign(value, bit):
    return 1 if value & (1 << bit) else -1


def threshold(terms, bias=0, polarity=1):
    if bias:
        terms = (*terms, (WORK + ONE, bias))
    return ((polarity, tuple(terms)),)


def constant(value):
    return ((value, ()),)


def buffer(source, polarity=1):
    return threshold(((source, 1),), polarity=polarity)


def match(requirements, polarity=1):
    return threshold(tuple(requirements), 1 - len(requirements), polarity)


def xor(left, right, polarity=1):
    terms = ((left, 1), (right, 1))
    return (*threshold(terms, 1, polarity), *threshold(terms, -1, -polarity), *constant(-polarity))


class Circuit:
    def __init__(self, operations, returns):
        self.phases = dict.fromkeys(PUBLIC)
        self.effects = defaultdict(dict)
        self.routes = defaultdict(dict)
        self.op_flag = {name: 1600 + index for index, name in enumerate(operations)}
        self.returns = {name: 1536 + index for index, name in enumerate(returns)}
        self.persistent = (
            *bits(OLD_HANDLE, HANDLE_BITS), OLD_HIT, *bits(PC), *bits(RAW), *bits(CAUSE, 3),
            *bits(NEXT_PC), *bits(RD, 5), *bits(RD_WORD), *range(RD_VALID, BUS_WIDTH + 2),
            *bits(BUS_ADDR), *bits(BUS_WORD), *bits(EXIT_CODE), *range(ONE, PENDING + 1), PUTCHAR_FLAG,
            *(lane for base in (RS1, RS2, A, B, X, P, G, SUM, EQUAL) for lane in bits(base)),
            *range(ZERO, MEM_SIGNED + 1), *self.op_flag.values(), *self.returns.values(),
        )

    def reserve_state(self, *lanes):
        self.persistent += lanes
        if len(set(self.persistent)) != len(self.persistent):
            raise ValueError('Overlapping persistent feedback lanes')

    def put(self, phase, dest, expression):
        self.phases.setdefault(phase)
        self.effects[phase][dest] = expression

    def move(self, phase, dest, sources, polarity=1):
        for offset, source in enumerate(sources):
            self.put(phase, dest + offset, buffer(source, polarity))

    def fix(self, phase, dest, value, count=WORD):
        for offset in range(count):
            self.put(phase, dest + offset, constant(sign(value, offset)))

    def route(self, phase, *choices):
        self.phases.setdefault(phase)
        for choice in choices:
            target, flag = (choice, None) if isinstance(choice, str) else choice
            self.phases.setdefault(target)
            self.routes[phase][target] = flag

    def branch(self, phase, flag, if_true, if_false):
        self.route(phase, (if_true, flag), (if_false, -flag))

    def address(self, phase, dest, sources, reg=False):
        self.move(phase, dest, sources)
        for offset in range(len(sources), WORD):
            self.put(phase, dest + offset, constant(-1))
        self.put(phase, dest + WORD, constant(1 if reg else -1))
        if dest == READ_ADDR:
            self.put(phase, READ_MARK, constant(1))

    def add(self, phase, left, right, destination, subtract=False, carry_in=None):
        self.move(phase, A, left)
        self.move(phase, B, right, -1 if subtract else 1)
        if carry_in is None:
            self.fix(phase, CARRY, int(subtract), 1)
        else:
            self.move(phase, CARRY, (carry_in,))
        for mode, lane in self.returns.items():
            self.put(phase, lane, constant(1 if mode == destination else -1))
        self.route(phase, 'ADD_INIT')

    def compile(self):
        ids = {phase: index for index, phase in enumerate(self.phases)}
        if ACTION + len(ids) > DISPATCH:
            raise ValueError('Action and dispatch lanes overlap')
        if len(set(self.persistent)) != len(self.persistent) or max(self.persistent) >= WORK or WORK + max(self.persistent) >= ACTION:
            raise ValueError('Overlapping feedback lanes')
        for phase in self.phases:
            if phase not in self.routes:
                raise ValueError(f'Missing transition from {phase}')
        coefficients = {}
        extra_effects = defaultdict(dict)
        next_dispatch = DISPATCH
        for phase, routes in self.routes.items():
            for target, flag in routes.items():
                coefficients[ids[target], ACTION + ids[phase]] = 64
                if flag is not None:
                    expression = self.effects[phase].get(abs(flag), buffer(WORK + abs(flag)))
                    extra_effects[phase][next_dispatch] = expression
                    coefficients[ids[target], next_dispatch] = math.copysign(1 / 16, flag)
                    next_dispatch += 1
        if next_dispatch > TINY_WIDTH:
            raise ValueError('Dispatch lanes exceed embedding width')
        effects = defaultdict(list)
        for phase, op in ids.items():
            phase_effects = self.effects[phase] | extra_effects[phase]
            if phase not in (SEED_MISS, SEED_WORD):
                for lane in self.persistent:
                    phase_effects.setdefault(lane, constant(-1) if phase == START else buffer(WORK + lane))
            for dest, expression in phase_effects.items():
                for factor, terms in expression:
                    effects[dest, factor, terms].append(ACTION + op)
        groups = defaultdict(list)
        for (dest, factor, terms), op_lanes in effects.items():
            groups[factor, terms, tuple(op_lanes)].append(dest)
        gate, up, down = [], [], []
        row = 0
        for (factor, terms, op_lanes), destinations in groups.items():
            # A centered difference of shifted SiLUs approaches a signed plateau.
            # The up projection gates the predicate with the active micro-op.
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


def program(m_extension=None):
    operations = OPERATIONS + (M_OPERATIONS if m_extension else ())
    returns = RETURN_MODES + (m_extension.M_RETURN if m_extension else ())
    c = Circuit(operations, returns)
    op_flag = c.op_flag

    def state(base, count=WORD):
        return bits(WORK + base, count)

    def q(phase, sources, reg=False):
        c.address(phase, READ_ADDR, sources, reg)

    def k(phase, sources, payload, reg=False):
        c.address(phase, WRITE_ADDR, sources, reg)
        c.move(phase, WRITE_WORD, payload)

    def fault(phase, cause, keep_branch=False):
        c.fix(phase, CAUSE, cause, 3)
        c.fix(phase, RD_VALID, 0, 1)
        c.fix(phase, BUS_VALID, 0, 1)
        if not keep_branch:
            c.fix(phase, BRANCH_VALID, 0, 1)
        c.move(phase, NEXT_PC, state(PC))
        c.route(phase, FAULT)

    def lookback(phase):
        c.move(phase, OLD_HANDLE, bits(READBACK + WORD, HANDLE_BITS))
        c.move(phase, OLD_HIT, (READBACK + WORD + HANDLE_BITS,))

    raw = state(RAW)
    rd = state(RD, 5)
    rs1, rs2 = state(RS1), state(RS2)
    pc, next_pc = state(PC), state(NEXT_PC)
    imm_i = (*raw[20:32], *([raw[31]] * 20))
    imm_s = (*raw[7:12], *raw[25:32], *([raw[31]] * 20))
    zero, one = WORK + ZERO, WORK + ONE
    imm_b = (zero, *raw[8:12], *raw[25:31], raw[7], raw[31], *([raw[31]] * 19))
    imm_j = (zero, *raw[21:31], raw[20], *raw[12:20], raw[31], *([raw[31]] * 11))
    imm_u = (*([zero] * 12), *raw[12:32])
    c.fix(START, ONE, 1, 1)
    c.move(START, PC, state(ENTRY))
    q(START, state(ENTRY))
    c.route(START, FETCH)
    for phase in (SEED_MISS, SEED_WORD, HALT, FAULT):
        c.route(phase, phase)
    c.move(COMMIT, PC, next_pc)
    q(COMMIT, next_pc)
    c.route(COMMIT, FETCH)

    c.move(FETCH, RAW, bits(READBACK))
    for lane in (RD_VALID, BUS_VALID, BUS_WRITE, BRANCH_VALID, TAKEN, PENDING, PUTCHAR_FLAG):
        c.fix(FETCH, lane, 0, 1)
    c.fix(FETCH, BUS_WIDTH, 0, 2)
    c.route(FETCH, 'PREP_SEQUENTIAL')
    c.add('PREP_SEQUENTIAL', pc, (zero, zero, one, *((zero,) * 29)), 'SEQUENTIAL')

    def encoding(opcode, funct3=None, funct7=None):
        value, mask = opcode, 0x7f
        if funct3 is not None:
            value |= funct3 << 12
            mask |= 7 << 12
        if funct7 is not None:
            value |= funct7 << 25
            mask |= 0x7f << 25
        return value, mask

    encodings = {
        'LUI': encoding(0x37), 'AUIPC': encoding(0x17), 'ADDI': encoding(0x13, 0),
        'ADD': encoding(0x33, 0, 0), 'SUB': encoding(0x33, 0, 0x20),
        'LW': encoding(0x03, 2), 'SW': encoding(0x23, 2),
        'BEQ': encoding(0x63, 0), 'BNE': encoding(0x63, 1), 'JAL': encoding(0x6f), 'JALR': encoding(0x67, 0),
        'EBREAK': (0x00100073, 0xffffffff), 'ECALL': (0x00000073, 0xffffffff),
        'FENCE': encoding(0x0f),
    }
    encodings.update((name, encoding(0x03, funct)) for name, funct in (('LB', 0), ('LH', 1), ('LBU', 4), ('LHU', 5)))
    encodings.update((name, encoding(0x23, funct)) for name, funct in (('SB', 0), ('SH', 1)))
    encodings.update((name, encoding(0x63, funct)) for name, funct in zip(COMPARE_BRANCHES, (4, 5, 6, 7)))
    for name, funct in (('SLL', 1), ('SLT', 2), ('SLTU', 3), ('XOR', 4), ('SRL', 5), ('OR', 6), ('AND', 7)):
        encodings[name] = encoding(0x33, funct, 0)
        suffix = 'I' if name not in ('SLT', 'SLTU') else ('I' if name == 'SLT' else '')
        immediate_name = 'SLTIU' if name == 'SLTU' else name + suffix
        encodings[immediate_name] = encoding(0x13, funct, 0 if name in ('SLL', 'SRL') else None)
    encodings['SRA'] = encoding(0x33, 5, 0x20)
    encodings['SRAI'] = encoding(0x13, 5, 0x20)
    encodings.update((f'SH{shift}ADD', encoding(0x33, 2 * shift, 0x10)) for shift in (1, 2, 3))
    for name, funct in CSR_READS.items():
        value, mask = encoding(0x73, funct)
        encodings[name] = value, mask | (31 << 15)
    if m_extension:
        encodings.update((name, encoding(0x33, funct, 1)) for funct, name in enumerate(M_OPERATIONS))
    if set(encodings) != set(operations):
        raise ValueError('Decode inventory and execution phases disagree')
    c.move('AFTER_SEQUENTIAL', NEXT_PC, state(SUM))
    c.route('AFTER_SEQUENTIAL', 'DECODE')
    for name, (value, mask) in encodings.items():
        c.put('DECODE', op_flag[name], match(tuple((lane, sign(value, bit)) for bit, lane in enumerate(raw) if mask & (1 << bit))))
    c.move('DECODE', RD, raw[7:12])
    # Each logit tests a single flag, and the selected prep phase emits the same raw register query.
    c.route('DECODE', 'CLASSIFY')
    regular = tuple(name for name in operations if name not in ('EBREAK', 'ECALL', *CSR_READS))
    c.route('CLASSIFY', *((f'PREP_RS1_{name}', op_flag[name]) for name in regular),
            *((f'CHECK_{name}', op_flag[name]) for name in CSR_READS),
            ('SET_EBREAK', op_flag['EBREAK']), ('PREP_ECALL', op_flag['ECALL']), 'SET_ILLEGAL')
    fault('SET_EBREAK', 1)
    fault('SET_ILLEGAL', 0)

    for name in regular:
        phase = f'PREP_RS1_{name}'
        q(phase, raw[15:20], reg=True)
        c.route(phase, 'READ_RS1')
    c.move('READ_RS1', RS1, bits(READBACK))
    q('READ_RS1', raw[20:25], reg=True)
    c.route('READ_RS1', 'READ_RS2')
    c.move('READ_RS2', RS2, bits(READBACK))
    c.route('READ_RS2', *((f'EXEC_{name}', op_flag[name]) for name in regular))

    def const_register(value):
        return tuple(one if sign(value, bit) > 0 else zero for bit in range(5))

    q('PREP_ECALL', const_register(17), reg=True)
    c.route('PREP_ECALL', 'READ_A7')
    c.move('READ_A7', RS1, bits(READBACK))
    q('READ_A7', const_register(10), reg=True)
    c.route('READ_A7', 'READ_A0')
    c.move('READ_A0', EXIT_CODE, bits(READBACK))
    c.put('READ_A0', IN_RANGE, match(tuple((lane, sign(93, bit)) for bit, lane in enumerate(rs1))))
    c.put('READ_A0', PUTCHAR_FLAG, match(tuple((lane, sign(1, bit)) for bit, lane in enumerate(rs1))))
    c.route('READ_A0', (HALT, IN_RANGE), (COMMIT, PUTCHAR_FLAG), 'SET_ECALL_FAULT')
    fault('SET_ECALL_FAULT', 7)

    for name in CSR_READS:
        phase = f'CHECK_{name}'
        choices = []
        for index, (address, value) in enumerate(CSR_STUBS.items()):
            flag = EXTRACT_FLAG + index
            c.put(phase, flag, match(tuple((lane, sign(address, bit)) for bit, lane in enumerate(raw[20:]))))
            target = f'CSR_VALUE_{address:03X}'
            choices.append((target, flag))
            c.fix(target, RD_WORD, value)
            c.route(target, 'RD_CHECK')
        c.route(phase, *choices, 'SET_ILLEGAL')

    c.move('EXEC_LUI', RD_WORD, imm_u)
    c.route('EXEC_LUI', 'RD_CHECK')
    c.add('EXEC_AUIPC', pc, imm_u, 'RD')
    c.add('EXEC_ADDI', rs1, imm_i, 'RD')
    c.add('EXEC_ADD', rs1, rs2, 'RD')
    c.add('EXEC_SUB', rs1, rs2, 'RD', subtract=True)
    for name, size in LOADS.items():
        c.add(f'EXEC_{name}', rs1, imm_i, {1: 'LB', 2: 'LH', 4: 'LW'}[size])
        c.fix(f'EXEC_{name}', MEM_SIGNED, int(name in ('LB', 'LH')), 1)
    for name in STORES:
        c.add(f'EXEC_{name}', rs1, imm_s, name)
    c.add('EXEC_JAL', pc, imm_j, 'JAL')
    c.add('EXEC_JALR', rs1, imm_i, 'JALR')
    c.route('EXEC_FENCE', COMMIT)
    for kind in ('AND', 'OR', 'XOR'):
        action = f'BIT_{kind}'
        for name, right in ((kind, rs2), (kind + 'I', imm_i)):
            phase = f'EXEC_{name}'
            c.move(phase, A, rs1)
            c.move(phase, B, right)
            c.route(phase, action)
        for bit, (left, right) in enumerate(zip(state(A), state(B))):
            expression = xor(left, right) if kind == 'XOR' else threshold(((left, 1), (right, 1)), -1 if kind == 'AND' else 1)
            c.put(action, RD_WORD + bit, expression)
        c.route(action, 'RD_CHECK')
    for shift in (1, 2, 3):
        c.add(f'EXEC_SH{shift}ADD', (zero,) * shift + rs1[:-shift], rs2, 'RD')

    for kind in ('SLL', 'SRL', 'SRA'):
        for name, amount in ((kind, rs2[:5]), (kind + 'I', raw[20:25])):
            phase = f'EXEC_{name}'
            c.move(phase, A, rs1)
            c.move(phase, B, amount)
            c.fix(phase, SHIFT_LEFT, int(kind == 'SLL'), 1)
            c.fix(phase, SHIFT_ARITH, int(kind == 'SRA'), 1)
            c.route(phase, 'SHIFT_TEST_0')
    shift_data = state(A)
    for bit, amount in enumerate(state(B, 5)):
        step = 1 << bit
        next_step = 'SHIFT_FINISH' if bit == 4 else f'SHIFT_TEST_{bit + 1}'
        c.branch(f'SHIFT_TEST_{bit}', amount - WORK, f'SHIFT_SELECT_{bit}', next_step)
        c.route(f'SHIFT_SELECT_{bit}', (f'SHIFT_LEFT_{bit}', SHIFT_LEFT), (f'SHIFT_ARITH_{bit}', SHIFT_ARITH), f'SHIFT_RIGHT_{bit}')
        for direction, sources in (('LEFT', (zero,) * step + shift_data[:-step]),
                                   ('RIGHT', shift_data[step:] + (zero,) * step),
                                   ('ARITH', shift_data[step:] + (shift_data[-1],) * step)):
            phase = f'SHIFT_{direction}_{bit}'
            c.move(phase, A, sources)
            c.route(phase, next_step)
    c.move('SHIFT_FINISH', RD_WORD, shift_data)
    c.route('SHIFT_FINISH', 'RD_CHECK')

    for name in (*COMPARE_BRANCHES, *COMPARE_VALUES):
        phase = f'EXEC_{name}'
        branch = name in COMPARE_BRANCHES
        right = imm_i if name in ('SLTI', 'SLTIU') else rs2
        c.put(phase, SIGN_DIFFER, xor(rs1[-1], right[-1]))
        c.fix(phase, SIGNED, int(not name.endswith('U')), 1)
        c.fix(phase, INVERT, int(name.startswith('BGE')), 1)
        c.fix(phase, CMP_BRANCH, int(branch), 1)
        c.fix(phase, BRANCH_VALID, int(branch), 1)
        c.add(phase, rs1, right, 'COMPARE', subtract=True)
    c.put('AFTER_COMPARE', PENDING, threshold(((WORK + SIGNED, 1), (WORK + SIGN_DIFFER, 1)), -1))
    c.route('AFTER_COMPARE', 'COMPARE_LESS')
    c.put('COMPARE_LESS', ALIGNED, xor(WORK + G + WORD - 1, WORK + PENDING, -1))
    c.route('COMPARE_LESS', 'COMPARE_SELECT')
    c.put('COMPARE_SELECT', TAKEN, xor(WORK + ALIGNED, WORK + INVERT))
    c.branch('COMPARE_SELECT', CMP_BRANCH, 'COMPARE_BRANCH', 'COMPARE_VALUE')
    c.branch('COMPARE_BRANCH', TAKEN, 'ADD_BRANCH', COMMIT)
    c.move('COMPARE_VALUE', RD_WORD, (WORK + TAKEN, *((zero,) * 31)))
    c.route('COMPARE_VALUE', 'RD_CHECK')

    for name in ('BEQ', 'BNE'):
        phase = f'EXEC_{name}'
        for bit, (left, right) in enumerate(zip(rs1, rs2)):
            c.put(phase, EQUAL + bit, xor(left, right, -1))
        c.fix(phase, BRANCH_VALID, 1, 1)
        c.route(phase, f'DECIDE_{name}')
        c.put(f'DECIDE_{name}', TAKEN, match(tuple((lane, 1) for lane in state(EQUAL)), 1 if name == 'BEQ' else -1))
        c.branch(f'DECIDE_{name}', TAKEN, 'ADD_BRANCH', COMMIT)
    c.add('ADD_BRANCH', pc, imm_b, 'BRANCH')

    carry = WORK + CARRY
    for bit, (left, right) in enumerate(zip(state(A), state(B))):
        c.put('ADD_INIT', X + bit, xor(left, right))
        c.put('ADD_INIT', P + bit, threshold(((left, 1), (right, 1)), 1))
        generator = ((left, 1), (right, 1), (carry, 1)) if bit == 0 else ((left, 1), (right, 1))
        c.put('ADD_INIT', G + bit, threshold(generator, 0 if bit == 0 else -1))
    c.route('ADD_INIT', 'PREFIX_1')
    for shift in (1, 2, 4, 8, 16):
        phase = f'PREFIX_{shift}'
        for bit in range(shift, WORD):
            propagate = WORK + P + bit
            c.put(phase, P + bit, threshold(((propagate, 1), (WORK + P + bit - shift, 1)), -1))
            c.put(phase, G + bit, threshold(((WORK + G + bit, 2), (propagate, 1), (WORK + G + bit - shift, 1)), 1))
        c.route(phase, 'ADD_FINISH' if shift == 16 else f'PREFIX_{2 * shift}')
    for bit, partial in enumerate(state(X)):
        c.put('ADD_FINISH', SUM + bit, xor(partial, carry if bit == 0 else WORK + G + bit - 1))
    c.route('ADD_FINISH', *((f'AFTER_{mode}', flag) for mode, flag in c.returns.items()))
    c.move('AFTER_RD', RD_WORD, state(SUM))
    c.route('AFTER_RD', 'RD_CHECK')

    for name in ('JAL', 'JALR', 'BRANCH'):
        phase = f'AFTER_{name}'
        c.move(phase, NEXT_PC, state(SUM))
        if name == 'JALR':
            c.fix(phase, NEXT_PC, 0, 1)
        if name != 'BRANCH':
            c.move(phase, RD_WORD, next_pc)
        c.route(phase, 'CHECK_BRANCH' if name == 'BRANCH' else 'CHECK_JUMP')
    jump_bounds = tuple((next_pc[bit], 1 if bit == 18 else -1) for bit in (0, 1, *range(18, 32)))
    for phase, target in (('CHECK_BRANCH', COMMIT), ('CHECK_JUMP', 'RD_CHECK')):
        c.put(phase, IN_RANGE, match(jump_bounds))
        c.branch(phase, IN_RANGE, target, 'SET_BAD_JUMP')
    fault('SET_BAD_JUMP', 6, keep_branch=True)

    for name, write in (('LW', False), ('SW', True)):
        width_code = {4: 0, 1: 1, 2: 2}
        for size, suffix in ((1, 'B'), (2, 'H'), (4, 'W')):
            phase = f'AFTER_{name[0]}{suffix}'
            address = state(SUM)
            c.move(phase, BUS_ADDR, address)
            c.fix(phase, BUS_VALID, 1, 1)
            c.fix(phase, BUS_WRITE, int(write), 1)
            c.fix(phase, BUS_WIDTH, width_code[size], 2)
            align_bits = size.bit_length() - 1
            if align_bits:
                c.put(phase, ALIGNED, match(tuple((lane, -1) for lane in address[:align_bits])))
            else:
                c.fix(phase, ALIGNED, 1, 1)
            c.put(phase, IN_RANGE, match(tuple((lane, -1) for lane in address[15:])))
            if not write:
                c.put(phase, PENDING, match(tuple((address[bit], 1 if bit == 17 else -1) for bit in range(17, WORD))))
            else:
                count = 8 * size
                c.move(phase, BUS_WORD, rs2[:count] + (zero,) * (WORD - count))
            c.branch(phase, ALIGNED, f'CHECK_{name}', f'SET_{name}_ALIGN')
        fault(f'SET_{name}_ALIGN', 3 if write else 2)
        fault(f'SET_{name}_RANGE', 5 if write else 4)
        if not write:
            c.put(f'CHECK_{name}', IN_RANGE, threshold(((WORK + IN_RANGE, 1), (WORK + PENDING, 1)), 1))
        c.branch(f'CHECK_{name}', IN_RANGE, f'PREP_{name}', f'SET_{name}_RANGE')
        q(f'PREP_{name}', (zero, zero, *state(BUS_ADDR)[2:]))
        c.route(f'PREP_{name}', 'STORE_LOOKUP' if write else 'READ_LW')

    aligned_bus = (zero, zero, *state(BUS_ADDR)[2:])
    for phase in ('READ_LW', 'STORE_LOOKUP'):
        writing = phase == 'STORE_LOOKUP'
        if writing:
            lookback(phase)
        c.move(phase, EQUAL, bits(READBACK))
        choices = []
        for size in (4, 2, 1):
            count = 8 * size
            for offset in range(0, 4, size):
                start, end = offset * 8, offset * 8 + count
                for signed in ((False,) if writing or size == 4 else (False, True)):
                    action = f'{"MERGE" if writing else "EXTRACT"}_{size}_{offset}_{int(signed)}'
                    conditions = tuple((lane, sign(width_code[size], bit)) for bit, lane in enumerate(state(BUS_WIDTH, 2)))
                    conditions += tuple((WORK + BUS_ADDR + bit, sign(offset, bit)) for bit in range(size.bit_length() - 1, 2))
                    if not writing and size < 4:
                        conditions += ((WORK + MEM_SIGNED, 1 if signed else -1),)
                    flag = EXTRACT_FLAG + len(choices)
                    c.put(phase, flag, match(conditions))
                    choices.append((action, flag))
                    data = state(EQUAL)
                    if writing:
                        k(action, aligned_bus, (*data[:start], *state(BUS_WORD)[:count], *data[end:]))
                        c.route(action, STORE)
                    else:
                        payload = data[start:end]
                        c.move(action, BUS_WORD, payload + (zero,) * (WORD - count))
                        c.move(action, RD_WORD, payload + ((payload[-1] if signed else zero),) * (WORD - count))
                        c.route(action, 'RD_CHECK')
        c.route(phase, *choices)
    c.put('RD_CHECK', RD_VALID, match(tuple((lane, -1) for lane in rd), -1))
    c.branch('RD_CHECK', RD_VALID, 'PREP_RD', COMMIT)
    q('PREP_RD', rd, reg=True)
    c.route('PREP_RD', 'REG_LOOKUP')
    lookback('REG_LOOKUP')
    k('REG_LOOKUP', rd, state(RD_WORD), reg=True)
    c.route('REG_LOOKUP', REGISTER)
    c.route(STORE, RETIRE)
    c.route(REGISTER, RETIRE)
    c.route(RETIRE, COMMIT)
    if m_extension:
        m_extension.add_m_program(c, state=state, rs1=rs1, rs2=rs2, zero=zero, carry_out=WORK + G + WORD - 1,
                                  result=state(SUM), finish='RD_CHECK', op_flags=op_flag)
    return c


def add_block(writer, index, width, ff, n_head, n_kv, vocab, packed, compiled, circuit, *, query_f32=False):
    ids, scores, gate, up, down, used = compiled
    if used > ff or n_head % n_kv or len(ids) > vocab:
        raise ValueError(f'Unexpected MTP geometry or FFN too small ({ff} rows for {used})')
    prefix = f'blk.{index}.'

    def add(name, shape, entries, matrix=False, ffn=False):
        if packed and ffn:
            data = np.zeros(shape, dtype='<u2')
            for (row, col), value in entries:
                data[row, col] = np.float32(value).view(np.uint32) >> 16
            writer.add_tensor(prefix + name, data.view(np.uint8), raw_dtype=GGMLQuantizationType.BF16)
        elif packed and matrix:
            rows, cols = shape
            data = np.zeros((rows, cols // 32, 18), dtype=np.uint8)
            data[..., 2:] = 0x88
            groups = defaultdict(dict)
            for (row, col), value in entries:
                groups[row, col // 32][col % 32] = value
            for position, values in groups.items():
                scale = max(abs(value) for value in values.values()) / 4
                target = data[position]
                target[:2] = np.array([scale], dtype='<f2').view(np.uint8)
                for col, value in values.items():
                    quant = value / scale
                    if quant != int(quant) or np.float16(scale) != scale:
                        raise ValueError(f'Inexact Q4_0 coefficient in {name} at {position}')
                    target[2 + col % 16] ^= (8 ^ (int(quant) + 8)) << (4 if col >= 16 else 0)
            writer.add_tensor(prefix + name, data.reshape(rows, -1), raw_dtype=GGMLQuantizationType.Q4_0)
        else:
            data = np.zeros(shape, dtype='f4')
            for position, value in entries:
                data[position] = value
            writer.add_tensor(prefix + name, data)

    action_lanes = tuple(ACTION + index for index in ids.values())
    external = (*bits(WRITE_ADDR, WORD + 1), *bits(WRITE_WORD), *bits(HANDLE_IN, HANDLE_BITS), *bits(ENTRY),
                *bits(READ_ADDR, WORD + 1), READ_MARK)
    ingress = (*external, *circuit.persistent)
    rms_weight = 1 / math.sqrt(width)
    norm = tuple((i, rms_weight) for i in range(width))
    add('nextn.embed_tokens.weight', (vocab, width), (((op, lane), 1) for op, lane in enumerate(action_lanes)), matrix=True)
    add('nextn.shared_head_head.weight', (vocab, width), scores.items(), matrix=True)
    add('nextn.enorm.weight', (width,), norm)
    add('nextn.hnorm.weight', (width,), ((i, rms_weight) for i in ingress))
    # F32 preserves signed predicate margins across sparse feedback projection rows.
    add('nextn.eh_proj.weight', (width, width * 2), [((i, i), 1) for i in action_lanes]
        + [((WORK + i, width + i), 1) for i in ingress])
    output_lanes = {*circuit.persistent, *bits(WRITE_ADDR, WORD + 1), *bits(WRITE_WORD), *bits(READ_ADDR, WORD + 1),
                    READ_MARK, *bits(READBACK, WORD + HANDLE_BITS + 1), *(lane for _, lane in scores)}
    add('nextn.shared_head_norm.weight', (width,), ((i, 1) for i in sorted(output_lanes)))
    add('attn_norm.weight', (width,), norm)

    # Each read compares the 33 typed address signs outside rotary. A sentinel key is closer than
    # every one-bit mismatch and farther than an exact match, yielding zero for an absent word.
    sentinel_lane = ROTARY + (WORD + 1) * REPEATS
    query, key, value, output = [], [], [], []
    for bit in range(WORD + 1):
        for rep in range(REPEATS):
            lane = ROTARY + bit * REPEATS + rep
            query.append(((lane, WORK + READ_ADDR + bit), 1))
            key.append(((lane, WORK + WRITE_ADDR + bit), 1))
    query.append(((sentinel_lane, WORK + READ_MARK), 11.125))
    key.append(((sentinel_lane, ACTION + ids[SEED_MISS]), 1))
    reads = (FETCH, 'READ_RS1', 'READ_RS2', 'READ_A7', 'READ_A0', 'READ_LW', 'STORE_LOOKUP', 'REG_LOOKUP')
    for bit in range(WORD + HANDLE_BITS + 1):
        for phase, op in ids.items():
            query.append(((HEAD + bit, ACTION + op), 128 if phase in reads else -128))
        output.append(((READBACK + bit, bit), 1))
        value.append(((bit, ACTION + ids[SEED_MISS]), -1 / 16))
    value.extend(((bit, WORK + WRITE_WORD + bit), 1) for bit in range(WORD))
    value.extend(((WORD + bit, WORK + HANDLE_IN + bit), 1) for bit in range(HANDLE_BITS))
    value.extend(((WORD + HANDLE_BITS, ACTION + ids[phase]), 1 / 16) for phase in (SEED_WORD, STORE, REGISTER))
    add('attn_q.weight', (2 * HEAD * n_head, width), query, matrix=not query_f32)
    # F32 preserves exact zero in the sentinel key lane of ordinary write rows.
    add('attn_k.weight', (HEAD * n_kv, width), key)
    add('attn_v.weight', (HEAD * n_kv, width), value, matrix=True)
    add('attn_output.weight', (width, HEAD * n_head), output, matrix=True)
    for name in ('attn_q_norm.weight', 'attn_k_norm.weight'):
        add(name, (HEAD,), ((i, 8) for i in range(ROTARY, sentinel_lane + 1)))
    add('post_attention_norm.weight', (width,), norm)
    for name, shape, entries in (('ffn_gate', (ff, width), gate), ('ffn_up', (ff, width), up),
                                  ('ffn_down', (width, ff), down)):
        add(name + '.weight', shape, entries, ffn=True)
    return {
        'context_request': CONTEXT, 'head_width': HEAD, 'query_heads': n_head, 'kv_heads': n_kv,
        'rotary_dimensions': ROTARY, 'tokens': ids, 'read_tokens': [ids[phase] for phase in reads],
        'operations': tuple(circuit.op_flag), 'mtp_ffn_rows_used': used, 'mtp_ffn_rows': ff,
        'mtp_ffn_type': 'BF16' if packed else 'F32', 'mtp_eh_proj_type': 'F32',
        'mtp_query_type': 'Q4_0' if packed and not query_f32 else 'F32',
        'mtp_key_type': 'F32', 'key_repeats': REPEATS,
        'key_address_lanes': bits(ROTARY, (WORD + 1) * REPEATS), 'sentinel_key_lane': sentinel_lane,
        'value_word_lanes': bits(0), 'value_handle_lanes': bits(WORD, HANDLE_BITS), 'value_hit_lane': WORD + HANDLE_BITS,
        'attention_readback_lanes': bits(READBACK, WORD + HANDLE_BITS + 1),
        'input_lanes': {'typed_seed_address': bits(WRITE_ADDR, WORD + 1), 'seed_word': bits(WRITE_WORD),
                        'fresh_handle': bits(HANDLE_IN, HANDLE_BITS), 'entry': bits(ENTRY)},
        'feedback_lanes': {'read_address': bits(READ_ADDR, WORD + 1), 'write_address': bits(WRITE_ADDR, WORD + 1),
                           'write_word': bits(WRITE_WORD), 'old_handle': bits(OLD_HANDLE, HANDLE_BITS), 'old_hit': OLD_HIT,
                           'pc': bits(PC), 'word': bits(RAW), 'fault': bits(CAUSE, 3), 'next_pc': bits(NEXT_PC),
                           'rd': bits(RD, 5), 'rd_word': bits(RD_WORD), 'rd_valid': RD_VALID, 'bus_valid': BUS_VALID,
                           'bus_write': BUS_WRITE, 'branch_valid': BRANCH_VALID, 'branch_taken': TAKEN,
                           'bus_address': bits(BUS_ADDR), 'bus_word': bits(BUS_WORD), 'bus_width': bits(BUS_WIDTH, 2),
                           'exit_code': bits(EXIT_CODE), 'putchar_flag': PUTCHAR_FLAG},
    }


def export(output, source, circuit, compiled, *, query_f32=False, ffn_padding=None):
    if output.exists():
        raise FileExistsError(output)
    minimum_space = 4 * 1024**3 if source else 1024**3
    if shutil.disk_usage(output.parent).free < minimum_space + (source.stat().st_size if source else 0):
        raise OSError('Insufficient free space for MTP checkpoint and verification')
    if source and digest_file(source) != BONSAI_HASH:
        raise ValueError('Unexpected Bonsai input hash')
    original = GGUFReader(source) if source else None
    changed = {'general.name', 'qwen35.block_count'}
    padded_names = set()
    if ffn_padding is not None:
        if not original or ffn_padding % 128:
            raise ValueError('FFN padding needs packed Bonsai and a block-aligned size')
        from export_stock_mtp_ffn_padding import pad_ffn, check_padded
        changed.add('qwen35.feed_forward_length')
    if original:
        writer = GGUFWriter(output, 'qwen35', endianess=original.endianess)
        fields = original.fields
        index, width, ff, head, n_head, n_kv, n_rot = (fields['qwen35.' + name].contents() for name in (
            'block_count', 'embedding_length', 'feed_forward_length', 'attention.key_length', 'attention.head_count',
            'attention.head_count_kv', 'rope.dimension_count'))
        vocab = len(fields['tokenizer.ggml.tokens'].contents())
        if (index, width, head, n_head, n_kv, n_rot, vocab, len(original.tensors)) != (64, 5120, HEAD, 24, 4, ROTARY, 248320, 851):
            raise ValueError('Unexpected Bonsai layout')
        if ffn_padding is not None:
            if ffn_padding < ff or ffn_padding < compiled[-1]:
                raise ValueError('FFN padding is smaller than the original or the circuit')
            ff = ffn_padding
        for name, field in fields.items():
            if name.startswith('GGUF.') or name in changed or name == 'general.architecture':
                continue
            if name == 'general.alignment':
                writer.data_alignment = int(field.contents())
            writer.add_key_value(name, field.contents(), field.types[0], field.types[1] if len(field.types) > 1 else None)
        if ffn_padding is not None:
            writer.add_uint32('qwen35.feed_forward_length', ff)
        for tensor in original.tensors:
            data = tensor.data
            if ffn_padding is not None and tensor.name.endswith(('.ffn_gate.weight', '.ffn_up.weight', '.ffn_down.weight')):
                data = pad_ffn(tensor, ff)
                padded_names.add(tensor.name)
            writer.add_tensor(tensor.name, data, raw_dtype=tensor.tensor_type, tensor_endianess=original.endianess)
    else:
        writer = GGUFWriter(output, 'qwen35')
        index, width, n_head, n_kv, vocab = 1, TINY_WIDTH, 1, 1, TINY_VOCAB
        ff = (compiled[-1] + 31) // 32 * 32
        for name, value in {
            'context_length': CONTEXT, 'embedding_length': width, 'vocab_size': vocab, 'feed_forward_length': ff,
            'attention.head_count': n_head, 'attention.head_count_kv': n_kv, 'attention.key_length': HEAD,
            'attention.value_length': HEAD, 'rope.dimension_count': ROTARY, 'ssm.conv_kernel': 4,
            'ssm.state_size': HEAD, 'ssm.group_count': 1, 'ssm.time_step_rank': 1,
            'ssm.inner_size': HEAD, 'full_attention_interval': 4,
        }.items():
            writer.add_uint32('qwen35.' + name, value)
        writer.add_array('qwen35.rope.dimension_sections', [16, 8, 8, 0])
        writer.add_float32('qwen35.attention.layer_norm_rms_epsilon', 1e-6)
        writer.add_float32('qwen35.rope.freq_base', 10000.)
        writer.add_string('tokenizer.ggml.model', 'none')
        writer.add_tensor('token_embd.weight', np.zeros((vocab, width), dtype=np.float32))
        writer.add_tensor('output_norm.weight', np.ones(width, dtype=np.float32))
    writer.add_name('Bonsai-27B with stock MTP scalar RV32' if source else 'Tiny stock MTP scalar RV32')
    writer.add_block_count(index + 1)
    writer.add_uint32('qwen35.nextn_predict_layers', 1)
    report = add_block(writer, index, width, ff, n_head, n_kv, vocab, bool(source), compiled, circuit, query_f32=query_f32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    report.update(bytes=output.stat().st_size, width=width, mtp_layer=index, sha256=digest_file(output))
    if original:
        reread = GGUFReader(output)
        source_fields = {name: field for name, field in original.fields.items()
                         if not name.startswith('GGUF.') and name not in changed}
        for name, field in source_fields.items():
            clone = reread.fields[name]
            if field.types != clone.types or field.contents() != clone.contents():
                raise ValueError(f'Packed language metadata changed: {name}')
        clones = {tensor.name: tensor for tensor in reread.tensors}
        digest = hashlib.sha256()
        for tensor in original.tensors:
            clone = clones[tensor.name]
            if tensor.name in padded_names:
                check_padded(tensor, clone)
            elif (tensor.tensor_type != clone.tensor_type or not np.array_equal(tensor.shape, clone.shape)
                    or not np.array_equal(tensor.data.view(np.uint8), clone.data.view(np.uint8))):
                raise ValueError(f'Packed language tensor changed: {tensor.name}')
            digest.update(memoryview(tensor.data).cast('B'))
        report.update(language_tensors_byte_equal=len(original.tensors) - len(padded_names),
                      language_tensors_zero_extended=len(padded_names),
                      language_original_packed_blocks_equal=True, ordered_language_payload_sha256=digest.hexdigest(),
                      source_metadata_fields_equal=len(source_fields), intentionally_changed_fields=sorted(changed),
                      added_metadata_fields=['qwen35.nextn_predict_layers'], source_bonsai_sha256=BONSAI_HASH,
                      source_bonsai_bytes=source.stat().st_size)
    output.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    import export_stock_mtp_scalar_m as m_extension

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path)
    parser.add_argument('--bonsai', type=Path)
    parser.add_argument('--q-f32', action='store_true', help='use F32 for the MTP query projection in a packed export')
    parser.add_argument('--plan', action='store_true', help='print the compiled phase IDs and required FFN rows')
    args = parser.parse_args()
    circuit = program(m_extension)
    compiled = circuit.compile()
    if args.plan:
        print(json.dumps({'phases': compiled[0], 'ffn_rows': compiled[-1]}, indent=2))
    elif args.output:
        export(args.output, args.bonsai, circuit, compiled, query_f32=args.q_f32)
    else:
        parser.error('output is required for export')
