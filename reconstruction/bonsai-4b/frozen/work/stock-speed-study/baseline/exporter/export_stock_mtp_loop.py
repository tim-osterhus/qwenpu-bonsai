"""Export a stock MTP byte checksum VM with indirect reads and runtime jumps."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
import numpy as np

from export_stock_mtp_ram import BONSAI_HASH, digest_file


START, RETIRE, HALT, CONTINUE, ADD_INDIRECT, ADVANCE = 0, 25, 26, 27, 28, 29
SEEDS, FETCHES, LOADS, LOOKUPS, COMMITS, ADDS = (
    tuple(range(i, i + n)) for i, n in ((1, 4), (5, 8), (13, 4), (17, 4), (21, 4), (30, 4)))
EVAL = tuple(range(34, 42))
CASES = tuple(tuple(range(42 + 8 * i, 50 + 8 * i)) for i in range(8))
JZS, JMPS, LENGTHS, TRAPS = (tuple(range(i, i + n)) for i, n in ((106, 8), (114, 8), (122, 5), (127, 6)))
N_OPCODES = 133
CONTEXT = 256
BYTE = range(8)
ACC, READOUT, PAYLOAD, HANDLE_OUT, ADDEND, CARRY = 160, 168, 176, 184, 192, 200
ACC_WORK, PAYLOAD_WORK, HANDLE_COPY, ADDEND_WORK, CARRY_WORK = 208, 216, 224, 232, 240
HANDLE_IN, SEED_IN, WORD_IN, LENGTH_IN = 248, 256, 264, 269
DISPATCH = (288, 320, 352)
WORD_WORK, LENGTH_WORK = 384, 392
PC, PC_WORK, PC_DISPATCH = 400, 408, 416
PTR, PTR_WORK, PTR_DISPATCH = 424, 432, 440
REMAIN, REMAIN_WORK, ZERO_DISPATCH = 448, 456, 464
WORD_BITS, PC_BITS, LENGTH_BITS, REGISTER_COUNT = 5, 3, 3, 5
TINY_WIDTH, TINY_VOCAB, TINY_FF = 480, 160, 416


def lanes(start, count=8):
    return tuple(range(start, start + count))


def signs(value, bits):
    return tuple(2 * ((value >> i) & 1) - 1 for i in range(bits))


def block(writer, index, width, ff, head, n_head, n_kv, n_rot, vocab, packed):
    prefix = f'blk.{index}.'

    def add(name, shape, entries=(), bf16=False):
        data = np.zeros(shape, dtype='f4')
        for position, value in entries:
            data[position] = value
        if bf16:
            data = (data.view(np.uint32) >> 16).astype('<u2').view(np.uint8)
        writer.add_tensor(prefix + name, data, raw_dtype=GGMLQuantizationType.BF16 if bf16 else None)

    def packed_rows(name, shape, entries):
        rows, cols = shape
        data = np.zeros((rows, cols // 32, 18), dtype=np.uint8)
        data[..., 2:] = 0x88
        groups = {}
        for (row, col), value in entries:
            groups.setdefault((row, col // 32), []).append((col % 32, value))
        for position, items in groups.items():
            scale = max(abs(value) for _, value in items) / 4
            target = data[position]
            target[:2] = np.array([scale], dtype='<f2').view(np.uint8)
            for col, value in items:
                quant = value / scale
                if quant != int(quant):
                    raise ValueError(f'Inexact Q4_0 coefficient in {name}')
                shift = 4 if col >= 16 else 0
                target[2 + col % 16] ^= (8 ^ (int(quant) + 8)) << shift
        writer.add_tensor(prefix + name, data.reshape(rows, -1), raw_dtype=GGMLQuantizationType.Q4_0)

    if n_rot + 3 > head or ZERO_DISPATCH >= width or head < 16 or N_OPCODES >= vocab:
        raise ValueError('Unexpected key, feedback, or vocabulary geometry')
    key_lanes = tuple(range(n_rot, n_rot + 3))
    addresses = tuple((*signs(i, 2), 1) for i in range(4))
    norm = [(i, 1 / math.sqrt(width)) for i in range(width)]
    cases = tuple(op for group in CASES for op in group)
    actions = tuple(range(TRAPS[0]))  # HALT is selected but never decoded.
    returns = (*LOADS, RETIRE, CONTINUE, ADVANCE, *CASES[-1], SEEDS[-1])

    transitions = [(op, SEEDS[0]) for op in LENGTHS]
    transitions += list(zip(SEEDS, SEEDS[1:]))
    transitions += list(zip(LOOKUPS, COMMITS))
    transitions += [(op, RETIRE) for op in COMMITS]
    transitions += [(op, EVAL[0]) for op in ADDS]
    transitions += [(op, target) for group, target in zip(CASES, EVAL[1:]) for op in group]
    scores = [((nxt, op), 8) for op, nxt in transitions]

    # Every FETCH applies the same word decoder. Low three bits of JZ/JMP select
    # the immediate-bearing micro-action; no row depends on a program fixture.
    words = (*LOADS, *LOOKUPS, ADD_INDIRECT, ADVANCE, HALT, *TRAPS[:5], *JZS, *JMPS)
    for word, op in enumerate(words):
        scores.extend(((op, fetch), 8) for fetch in FETCHES)
        scores.extend(((op, WORD_WORK + bit), 4 * sign) for bit, sign in enumerate(signs(word, WORD_BITS)))
    for length, op in enumerate(LENGTHS):
        scores.append(((op, START), 8))
        scores.extend(((op, LENGTH_WORK + bit), 4 * sign) for bit, sign in enumerate(signs(length, LENGTH_BITS)))
    for target, fetch in enumerate(FETCHES):
        scores.extend(((fetch, PC_DISPATCH + bit), 4 * sign) for bit, sign in enumerate(signs(target, PC_BITS)))
    for jz, jmp in zip(JZS, JMPS):
        scores.extend((((jmp, jz), 8), ((jmp, ZERO_DISPATCH), 4), ((CONTINUE, jz), 8)))
    scores.append(((CONTINUE, ZERO_DISPATCH), -4))
    for address, op in enumerate((*ADDS, TRAPS[-1])):
        scores.extend((((op, ADD_INDIRECT), 8), ((op, PTR_DISPATCH + address), 4)))
    for evaluate, group in zip(EVAL, CASES):
        for case, op in enumerate(group):
            scores.append(((op, evaluate), 8))
            scores.extend(((op, lane), 4 * sign) for lane, sign in zip(DISPATCH, signs(case, 3)))

    table = packed_rows if packed else add
    table('nextn.embed_tokens.weight', (vocab, width), [((i, i), 1) for i in actions])
    table('nextn.shared_head_head.weight', (vocab, width), scores)
    add('nextn.enorm.weight', (width,), norm)
    feedback = (*lanes(ACC), *lanes(PAYLOAD), *lanes(HANDLE_OUT), *lanes(ADDEND), CARRY,
                *lanes(PC, PC_BITS), *lanes(PTR, REGISTER_COUNT), *lanes(REMAIN, REGISTER_COUNT))
    ingress = (*lanes(HANDLE_IN), *lanes(SEED_IN), *lanes(WORD_IN, WORD_BITS), *lanes(LENGTH_IN, LENGTH_BITS))
    add('nextn.hnorm.weight', (width,), [(i, norm[i][1]) for i in (*feedback, *ingress)])
    moves = [(src + bit, dest + bit) for src, dest in ((ACC, ACC_WORK), (PAYLOAD, PAYLOAD_WORK),
             (HANDLE_OUT, HANDLE_COPY), (ADDEND, ADDEND_WORK)) for bit in BYTE]
    moves.extend(((CARRY, CARRY_WORK), *zip(lanes(WORD_IN, WORD_BITS), lanes(WORD_WORK, WORD_BITS)),
                  *zip(lanes(LENGTH_IN, LENGTH_BITS), lanes(LENGTH_WORK, LENGTH_BITS)),
                  *zip(lanes(PC, PC_BITS), lanes(PC_WORK, PC_BITS)),
                  *zip(lanes(PTR, REGISTER_COUNT), lanes(PTR_WORK, REGISTER_COUNT)),
                  *zip(lanes(REMAIN, REGISTER_COUNT), lanes(REMAIN_WORK, REGISTER_COUNT))))
    moves.extend((i, i) for i in (*lanes(HANDLE_IN), *lanes(SEED_IN)))
    add('nextn.eh_proj.weight', (width, 2 * width), [((i, i), 1) for i in actions]
        + [((dst, width + src), 1) for src, dst in moves])
    add('nextn.shared_head_norm.weight', (width,), [(i, 1) for i in
        (*actions, *feedback, *lanes(READOUT), *lanes(WORD_WORK, WORD_BITS), *lanes(LENGTH_WORK, LENGTH_BITS),
         *DISPATCH, *lanes(PC_DISPATCH, PC_BITS), *lanes(PTR_DISPATCH, REGISTER_COUNT), ZERO_DISPATCH)])

    reads = dict(zip(LOADS, addresses)) | dict(zip(LOOKUPS, addresses))
    reads.update(zip(ADDS, addresses))
    writes = dict(zip(SEEDS, addresses)) | dict(zip(COMMITS, addresses))
    q = [((head + lane, op), 512 if op in reads else -512) for lane in range(16) for op in actions]
    q += [((lane, op), sign) for op, address in reads.items() for lane, sign in zip(key_lanes, address)]
    k = [((lane, op), sign) for op, address in writes.items() for lane, sign in zip(key_lanes, address)]
    add('attn_norm.weight', (width,), norm)
    add('attn_q.weight', (2 * head * n_head, width), q)
    add('attn_k.weight', (head * n_kv, width), k)
    add('attn_v.weight', (head * n_kv, width), [((bit, src + bit), 8) for src in (PAYLOAD_WORK, SEED_IN) for bit in BYTE]
        + [((8 + bit, HANDLE_IN + bit), 8) for bit in BYTE])
    add('attn_output.weight', (width, head * n_head), [((READOUT + bit, bit), 1) for bit in BYTE]
        + [((HANDLE_OUT + bit, 8 + bit), 1) for bit in BYTE])
    for name in ('attn_q_norm.weight', 'attn_k_norm.weight'):
        add(name, (head,), [(i, 16) for i in key_lanes])
    add('post_attention_norm.weight', (width,), norm)

    gate, up, down = [], [], []
    row = 0

    def unit(g, u, d):
        nonlocal row
        gate.extend(((row, lane), weight) for lane, weight in g)
        up.extend(((row, lane), weight) for lane, weight in u)
        down.extend(((lane, row), weight) for lane, weight in d)
        row += 1

    def refresh(opcodes, src, *destinations):
        # e is the selected opcode activation. This clips each signed bit using
        # e * [SiLU(128x+4e) - SiLU(128x-4e) - SiLU(4e) + SiLU(-4e)].
        # The plateau equals +/-4e², the same scale used for fixed bits below.
        u = [(op, 1) for op in opcodes]
        for source, bias, sign in ((True, 4, 1), (True, -4, -1), (False, 4, -1), (False, -4, 1)):
            g = [(op, bias) for op in opcodes]
            if source:
                g.append((src, 128))
            unit(g, u, [(dest, sign) for dest in destinations])

    def fixed(opcode_signs, *destinations):
        for sign in (1, -1):
            unit([(op, 4 * sign) for op, _ in opcode_signs], opcode_signs, [(dest, sign) for dest in destinations])

    def move(opcodes, src, dest):
        for sign in (1, -1):
            unit([(op, 4 * sign) for op in opcodes], [(src, 1)], [(dest, sign)])

    keep_acc = tuple(op for op in actions if op not in (START, *LOADS))
    for bit in BYTE:
        replace = CASES[bit]
        refresh(tuple(op for op in keep_acc if op not in replace), ACC_WORK + bit, ACC + bit)
        refresh(LOADS, READOUT + bit, ACC + bit)
        fixed([(START, -1)] + [(op, 2 * (case.bit_count() % 2) - 1)
                               for case, op in enumerate(replace)], ACC + bit)
        refresh(LOOKUPS, ACC_WORK + bit, PAYLOAD + bit)
        move(COMMITS, HANDLE_COPY + bit, HANDLE_OUT + bit)
    keep_addend = (*EVAL, *cases)
    for bit in BYTE:
        refresh(ADDS, READOUT + bit, ADDEND + bit)
        refresh(keep_addend, ADDEND_WORK + bit, ADDEND + bit)
        move((EVAL[bit],), ACC_WORK + bit, DISPATCH[0])
        move((EVAL[bit],), ADDEND_WORK + bit, DISPATCH[1])
    refresh(EVAL, CARRY_WORK, CARRY, DISPATCH[2])
    fixed([(op, -1) for op in ADDS] + [(op, 1 if case.bit_count() >= 2 else -1)
          for group in CASES for case, op in enumerate(group)], CARRY)

    # FETCH advances the model's PC. A jump instead replaces the PC and
    # dispatches it immediately. Other instruction completions use the PC
    # already computed by FETCH, including conditional fallthrough.
    keep_pc = tuple(op for op in actions if op not in (START, *FETCHES, *JMPS, *returns))
    for bit in range(PC_BITS):
        refresh(keep_pc, PC_WORK + bit, PC + bit)
        refresh(returns, PC_WORK + bit, PC + bit, PC_DISPATCH + bit)
        fixed([(START, -1)] + [(op, signs((target + 1) % len(FETCHES), PC_BITS)[bit])
                               for target, op in enumerate(FETCHES)], PC + bit)
        fixed([(op, signs(target, PC_BITS)[bit]) for target, op in enumerate(JMPS)],
              PC + bit, PC_DISPATCH + bit)

    keep_ptr = tuple(op for op in actions if op not in (START, ADVANCE))
    keep_remain = tuple(op for op in actions if op not in (START, *LENGTHS, ADVANCE))
    for index in range(REGISTER_COUNT):
        refresh(keep_ptr, PTR_WORK + index, PTR + index)
        refresh((ADD_INDIRECT,), PTR_WORK + index, PTR_DISPATCH + index)
        refresh(keep_remain, REMAIN_WORK + index, REMAIN + index)
        ptr_init = [(START, 1 if index == 0 else -1)]
        remain_init = [(START, 1 if index == 0 else -1)] + [
            (op, 1 if length == index else -1) for length, op in enumerate(LENGTHS)]
        if index == 0:
            ptr_init.append((ADVANCE, -1))
        else:
            refresh((ADVANCE,), PTR_WORK + index - 1, PTR + index)
        if index == REGISTER_COUNT - 1:
            remain_init.append((ADVANCE, -1))
        else:
            refresh((ADVANCE,), REMAIN_WORK + index + 1, REMAIN + index)
        fixed(ptr_init, PTR + index)
        fixed(remain_init, REMAIN + index)
    refresh(JZS, REMAIN_WORK, ZERO_DISPATCH)
    if row > ff:
        raise ValueError(f'FFN has {ff} rows, needs {row}')
    add('ffn_gate.weight', (ff, width), gate, bf16=packed)
    add('ffn_up.weight', (ff, width), up, bf16=packed)
    add('ffn_down.weight', (width, ff), down, bf16=packed)
    return {'context_request': CONTEXT, 'head_width': head, 'query_heads': n_head, 'kv_heads': n_kv,
            'rotary_dimensions': n_rot, 'address_key_lanes': key_lanes,
            'accumulator_lanes': lanes(ACC), 'payload_lanes': lanes(PAYLOAD), 'readout_lanes': lanes(READOUT),
            'addend_lanes': lanes(ADDEND), 'carry_lane': CARRY, 'adder_dispatch_lanes': DISPATCH,
            'word_ingress_lanes': lanes(WORD_IN, WORD_BITS), 'word_dispatch_lanes': lanes(WORD_WORK, WORD_BITS),
            'seed_ingress_lanes': lanes(SEED_IN), 'handle_ingress_lanes': lanes(HANDLE_IN),
            'length_ingress_lanes': lanes(LENGTH_IN, LENGTH_BITS), 'length_dispatch_lanes': lanes(LENGTH_WORK, LENGTH_BITS),
            'pc_lanes': lanes(PC, PC_BITS), 'pc_dispatch_lanes': lanes(PC_DISPATCH, PC_BITS),
            'pointer_lanes': lanes(PTR, REGISTER_COUNT), 'pointer_dispatch_lanes': lanes(PTR_DISPATCH, REGISTER_COUNT),
            'remaining_lanes': lanes(REMAIN, REGISTER_COUNT), 'zero_dispatch_lane': ZERO_DISPATCH,
            'handle_feedback_lanes': lanes(HANDLE_OUT), 'mtp_ffn_rows_used': row,
            'mtp_ffn_type': 'BF16' if packed else 'F32'}


def export(output, source):
    if output.exists():
        raise FileExistsError(output)
    if source and digest_file(source) != BONSAI_HASH:
        raise ValueError('Unexpected Bonsai input hash')
    original = GGUFReader(source) if source else None
    if original:
        writer = GGUFWriter(output, 'qwen35', endianess=original.endianess)
        fields = original.fields

        def config(name):
            return fields['qwen35.' + name].contents()

        index, width, ff = (config(name) for name in ('block_count', 'embedding_length', 'feed_forward_length'))
        head, n_head, n_kv = (config(name) for name in ('attention.key_length', 'attention.head_count', 'attention.head_count_kv'))
        n_rot = config('rope.dimension_count')
        vocab = len(fields['tokenizer.ggml.tokens'].contents())
        if (index, width, vocab, len(original.tensors)) != (64, 5120, 248320, 851):
            raise ValueError('Unexpected Bonsai layout')
        for name, field in fields.items():
            if name.startswith('GGUF.') or name in ('general.architecture', 'general.name', 'qwen35.block_count'):
                continue
            if name == 'general.alignment':
                writer.data_alignment = int(field.contents())
            writer.add_key_value(name, field.contents(), field.types[0], field.types[1] if len(field.types) > 1 else None)
        for tensor in original.tensors:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type, tensor_endianess=original.endianess)
    else:
        writer = GGUFWriter(output, 'qwen35')
        index, width, ff, head, n_head, n_kv, n_rot, vocab = 1, TINY_WIDTH, TINY_FF, 64, 1, 1, 32, TINY_VOCAB
        for name, value in {
            'context_length': CONTEXT, 'embedding_length': width, 'vocab_size': vocab, 'feed_forward_length': ff,
            'attention.head_count': n_head, 'attention.head_count_kv': n_kv, 'attention.key_length': head,
            'attention.value_length': head, 'rope.dimension_count': n_rot, 'ssm.conv_kernel': 4,
            'ssm.state_size': head, 'ssm.group_count': 1, 'ssm.time_step_rank': 1,
            'ssm.inner_size': head, 'full_attention_interval': 4,
        }.items():
            writer.add_uint32('qwen35.' + name, value)
        writer.add_array('qwen35.rope.dimension_sections', [8, 4, 4, 0])
        writer.add_float32('qwen35.attention.layer_norm_rms_epsilon', 1e-6)
        writer.add_float32('qwen35.rope.freq_base', 10000.)
        writer.add_string('tokenizer.ggml.model', 'none')
        writer.add_tensor('token_embd.weight', np.zeros((vocab, width), dtype=np.float32))
        writer.add_tensor('output_norm.weight', np.ones(width, dtype=np.float32))

    writer.add_name('Bonsai-27B with stock MTP looped checksum' if original else 'Tiny stock MTP looped checksum')
    writer.add_block_count(index + 1)
    writer.add_uint32('qwen35.nextn_predict_layers', 1)
    report = block(writer, index, width, ff, head, n_head, n_kv, n_rot, vocab, packed=bool(original))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    report.update(bytes=output.stat().st_size, width=width, mtp_layer=index, sha256=digest_file(output))
    if original:
        reread = GGUFReader(output)
        changed_fields = {'general.name', 'qwen35.block_count'}
        source_fields = {name: field for name, field in original.fields.items()
                         if not name.startswith('GGUF.') and name not in changed_fields}
        for name, field in source_fields.items():
            clone = reread.fields[name]
            if field.types != clone.types or field.contents() != clone.contents():
                raise ValueError(f'Packed language metadata changed: {name}')
        combined = {tensor.name: tensor for tensor in reread.tensors}
        digest = hashlib.sha256()
        for tensor in original.tensors:
            clone = combined[tensor.name]
            if (tensor.tensor_type != clone.tensor_type or not np.array_equal(tensor.shape, clone.shape)
                    or not np.array_equal(tensor.data.view(np.uint8), clone.data.view(np.uint8))):
                raise ValueError(f'Packed language tensor changed: {tensor.name}')
            digest.update(memoryview(tensor.data).cast('B'))
        report.update(language_tensors_byte_equal=len(original.tensors), ordered_language_payload_sha256=digest.hexdigest(),
                      source_metadata_fields_equal=len(source_fields), intentionally_changed_fields=sorted(changed_fields),
                      added_metadata_fields=['qwen35.nextn_predict_layers'], source_bonsai_sha256=BONSAI_HASH,
                      source_bonsai_bytes=source.stat().st_size)
    output.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--bonsai', type=Path)
    args = parser.parse_args()
    export(args.output, args.bonsai)
