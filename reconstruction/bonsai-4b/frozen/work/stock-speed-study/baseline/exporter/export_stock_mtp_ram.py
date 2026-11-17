"""Export a bounded, two-address RAM program in a stock Qwen3.5 MTP block."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
import numpy as np


START, HALT, EMPTY = 0, 1, 2
WRITES = tuple(range(3, 9))
READS = tuple(range(9, 15))
SCRUBS = tuple(range(15, 21))
DELAYS = tuple(range(21, 24))
BRANCHES = tuple((24 + 4 * i, 25 + 4 * i) for i in range(6))
COMMITS = tuple((26 + 4 * i, 27 + 4 * i) for i in range(6))
PROGRAM = (START, WRITES[0], WRITES[1], DELAYS[0], SCRUBS[0], READS[0], SCRUBS[1], READS[1],
           WRITES[2], DELAYS[1], WRITES[3], SCRUBS[2], READS[2], DELAYS[2], SCRUBS[3], READS[3],
           WRITES[4], SCRUBS[4], READS[4], WRITES[5], SCRUBS[5], READS[5], HALT)
H, R, M = 48, 49, 50  # External data input, RAM readout, previous readout for commit.
CAP = 64
PIN = '9fcaed763ccda38ea81068ad9d7f991aaddca451'
BONSAI_HASH = '17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0'
INPUTS = {'a': '100110', 'b': '011001', 'c': '010110', 'cleared': '100110'}


def block(writer, index, width, ff, head, n_head, n_kv, n_rot, base, vocab, packed):
    prefix = f'blk.{index}.'

    def add(name, shape, entries=(), dtype='f4'):
        data = np.zeros(shape, dtype=dtype)
        for position, value in entries:
            data[position] = value
        writer.add_tensor(prefix + name, data)

    def packed_rows(name, shape, entries):
        # All nonzero weights in each block share an exact Q4_0 scale.
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

    one = list(enumerate([1] * width))
    unit = 1 / math.sqrt(width)
    rotations = n_rot // 2
    rotary = min(range(rotations), key=lambda i: abs(base ** (-i / rotations) - 0.020))
    theta = base ** (-rotary / rotations)
    max_age = len(PROGRAM) + 2 * len(READS) - 1
    if not 0 < theta * max_age < math.pi / 2 or n_rot >= head:
        raise ValueError('The configured RoPE pair cannot order this bounded program')

    emb = [((token, token), 1) for token in range(48)]
    scores = [((successor, token), 64) for token, successor in zip(PROGRAM, PROGRAM[1:]) if token not in READS]
    for read, branch, commit in zip(READS, BRANCHES, COMMITS):
        scores.append(((EMPTY, read), 9))
        for polarity, branch_token, commit_token in zip((1, -1), branch, commit):
            scores.extend([((branch_token, read), 8), ((branch_token, R), 4 * polarity),
                           ((commit_token, branch[0]), 8), ((commit_token, branch[1]), 8),
                           ((commit_token, M), 4 * polarity)])
            scores.append(((PROGRAM[PROGRAM.index(read) + 1], commit_token), 64))
    scores.append(((HALT, EMPTY), 64))
    write_table = packed_rows if packed else add
    write_table('nextn.embed_tokens.weight', (vocab, width), emb)
    write_table('nextn.shared_head_head.weight', (vocab, width), scores)
    add('nextn.enorm.weight', (width,), list(enumerate([unit] * width)))
    add('nextn.hnorm.weight', (width,), [(H, unit), (R, unit)])
    add('nextn.eh_proj.weight', (width, 2 * width),
        [((i, i), 1) for i in range(48)] + [((H, width + H), 1), ((M, width + R), 1)])
    add('nextn.shared_head_norm.weight', (width,), [(i, 1) for i in range(48)] + [(R, 1), (M, 1)])

    q = [((head, token), 256 if token in READS else -256) for token in range(48)]
    k = []
    for i, token in enumerate(READS):
        q.extend([((n_rot, token), (-1) ** i), ((rotary + rotations, token), 1)])
    for i, token in enumerate(WRITES):
        k.extend([((n_rot, token), (-1) ** i), ((rotary, token), 1)])
    add('attn_norm.weight', (width,), list(enumerate([unit] * width)))
    add('attn_q.weight', (2 * head * n_head, width), q)
    add('attn_k.weight', (head * n_kv, width), k)
    add('attn_v.weight', (head * n_kv, width), [((0, H), 4)])
    add('attn_output.weight', (width, head * n_head), [((R, 0), 1)])
    for name in ('attn_q_norm.weight', 'attn_k_norm.weight'):
        add(name, (head,), [(n_rot, 128), (rotary, 64), (rotary + rotations, 64)])
    add('post_attention_norm.weight', (width,), one)
    for name, shape in (('ffn_gate.weight', (ff, width)), ('ffn_up.weight', (ff, width)),
                        ('ffn_down.weight', (width, ff))):
        write_table(name, shape, [])
    return {'rotary_pair': rotary, 'radians_per_position': theta, 'maximum_fixture_age': max_age}


def export(output, source):
    if output.exists():
        raise FileExistsError(output)
    original = GGUFReader(source) if source else None
    if original:
        writer = GGUFWriter(output, 'qwen35', endianess=original.endianess)
        fields = original.fields

        def config(name):
            return fields['qwen35.' + name].contents()

        index, width, ff = (config(name) for name in ('block_count', 'embedding_length', 'feed_forward_length'))
        head, n_head, n_kv = (config(name) for name in ('attention.key_length', 'attention.head_count', 'attention.head_count_kv'))
        n_rot, base = (config(name) for name in ('rope.dimension_count', 'rope.freq_base'))
        vocab = len(fields['tokenizer.ggml.tokens'].contents())
        if (index, width, vocab, len(original.tensors)) != (64, 5120, 248320, 851):
            raise ValueError('Expected the unmodified Bonsai-27B Binary model')
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
        index, width, ff, head, n_head, n_kv, n_rot, base, vocab = 1, CAP, 64, 64, 1, 1, 32, 10000., CAP
        for name, value in {
            'context_length': 64, 'embedding_length': width, 'vocab_size': vocab, 'feed_forward_length': ff,
            'attention.head_count': n_head, 'attention.head_count_kv': n_kv, 'attention.key_length': head,
            'attention.value_length': head, 'rope.dimension_count': n_rot, 'ssm.conv_kernel': 4,
            'ssm.state_size': head, 'ssm.group_count': 1, 'ssm.time_step_rank': 1,
            'ssm.inner_size': head, 'full_attention_interval': 4,
        }.items():
            writer.add_uint32('qwen35.' + name, value)
        writer.add_array('qwen35.rope.dimension_sections', [8, 4, 4, 0])
        writer.add_float32('qwen35.attention.layer_norm_rms_epsilon', 1e-6)
        writer.add_float32('qwen35.rope.freq_base', base)
        writer.add_string('tokenizer.ggml.model', 'none')
        writer.add_tensor('token_embd.weight', np.zeros((vocab, width), dtype=np.float32))
        writer.add_tensor('output_norm.weight', np.ones(width, dtype=np.float32))

    writer.add_name('Bonsai-27B with stock MTP RAM' if original else 'Tiny stock Qwen3.5 MTP RAM')
    writer.add_block_count(index + 1)
    writer.add_uint32('qwen35.nextn_predict_layers', 1)
    report = block(writer, index, width, ff, head, n_head, n_kv, n_rot, base, vocab, packed=bool(original))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    result = GGUFReader(output)
    active = next(t for t in result.tensors if t.name == f'blk.{index}.attn_v.weight')
    report.update(bytes=output.stat().st_size, width=width, mtp_layer=index, mutation_tensor=active.name,
                  mutation_file_offset=active.data_offset + 4 * H, mutation_before=4, mutation_after=-4)
    if original:
        combined = {tensor.name: tensor for tensor in result.tensors}
        digest = hashlib.sha256()
        for tensor in original.tensors:
            clone = combined[tensor.name]
            if (tensor.tensor_type != clone.tensor_type or not np.array_equal(tensor.shape, clone.shape)
                    or not np.array_equal(tensor.data.view(np.uint8), clone.data.view(np.uint8))):
                raise ValueError(f'Packed language tensor changed: {tensor.name}')
            digest.update(memoryview(tensor.data).cast('B'))
        report.update(language_tensors_byte_equal=len(original.tensors), ordered_language_payload_sha256=digest.hexdigest())
    report['sha256'] = digest_file(output)
    output.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def digest_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def mutate(model, restore):
    manifest = json.loads(model.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    before, after = manifest['mutation_before'], manifest['mutation_after']
    if restore:
        before, after = after, before
    elif digest_file(model) != manifest['sha256']:
        raise ValueError('The input checkpoint does not match its original hash')
    with model.open('r+b') as stream:
        offset = manifest['mutation_file_offset']
        stream.seek(offset)
        if stream.read(4) != struct.pack('<f', before):
            raise ValueError('The input coefficient does not match the mutation manifest')
        stream.seek(offset)
        stream.write(struct.pack('<f', after))
    sha = digest_file(model)
    if restore and sha != manifest['sha256']:
        raise ValueError('The restored checkpoint does not match its original hash')
    model.with_name(model.stem + ('-restored' if restore else '-mutated') + '.sha256').write_text(sha + '\n', encoding='ascii')


def expect(condition, message):
    if not condition:
        raise ValueError(message)


def verify_trace(trace, payloads, *, cleared, mutated):
    # Independent address/value reference and the externally specified program.
    # Wn and Rn are address A for even n and B for odd n.
    program = (0, 3, 4, 21, 15, 9, 16, 10, 5, 22, 6, 17, 11, 23, 18, 12,
               7, 19, 13, 8, 20, 14, 1)
    memory = [None, None]
    expected = []
    values = ''
    for token in program:
        expected.append(token)
        if 3 <= token <= 8:
            i = token - 3
            memory[i % 2] = int(payloads[i]) ^ mutated
        if 9 <= token <= 14:
            i = token - 9
            if cleared and i == 2:
                memory = [None, None]
            value = memory[i % 2]
            if value is None:
                expected.extend((2, 1))
                break
            branch = 24 + 4 * i + (1 - value)
            expected.extend((branch, branch + 2))
            values += str(value)

    actual = [(row['position'], row['input'], row['selected']) for row in trace]
    expect(actual == [(i, token, next_token) for i, (token, next_token) in enumerate(zip(expected, expected[1:]))],
           f'Selected trace differs for payloads {payloads}, cleared={cleared}, mutated={mutated}')
    expect(all(math.isfinite(row['margin']) and row['margin'] > 0 for row in trace), 'Nonpositive selection margin')
    reads = [row for row in trace if 'readout' in row]
    commits = [row['commit_input'] for row in trace if 'commit_input' in row]
    expect(len(reads) == len(values) + int(cleared) and len(commits) == len(values), 'Missing numeric witness')
    for read, commit, value in zip(reads, commits, values):
        sign = 1 if value == '1' else -1
        expect(read['readout'] * sign > 1 and commit * sign > 1, 'Read or commit lost the expected polarity')
    if cleared:
        expect(reads[-1]['readout'] == 0 and reads[-1]['selected'] == 2, 'Cache clear did not produce EMPTY')
    return {'payloads': payloads, 'branches_and_commits': values, 'selected_trace': expected,
            'minimum_logit_margin': min(row['margin'] for row in trace), 'reads': reads}


def verify(directory, runtime, output, source):
    def load(name):
        return json.loads((directory / (name + '.json')).read_text(encoding='utf-8-sig'))

    def experiment(label, mutated=False):
        data = load(label)
        expect(all(data[key] for key in ('byte_identical_read_feedback', 'stale_invariance', 'aba_reset')), label)
        traces = {name: verify_trace(data[name], payloads, cleared=name == 'cleared', mutated=mutated)
                  for name, payloads in INPUTS.items()}
        for name in ('a', 'b', 'c'):
            expect(data[name + '_branches'] == data[name + '_commits'] == traces[name]['branches_and_commits'], label)
        for i, read in enumerate(traces['a']['reads']):
            for name in ('b', 'c'):
                other = traces[name]['reads'][i]
                expect((other['position'], other['input'], other['h_input_sha256']) ==
                       (read['position'], read['input'], read['h_input_sha256']), 'Matched read input differs')
        clear = traces['cleared']['reads'][-1]
        baseline = traces['a']['reads'][2]
        expect((clear['position'], clear['input'], clear['h_input_sha256']) ==
               (baseline['position'], baseline['input'], baseline['h_input_sha256']), 'Cache-clear input differs')
        return {'traces': traces, 'width': data['width'], 'vocab': data['vocab'],
                'peak_working_set_bytes': data['peak_working_set_bytes']}

    report = {
        'status': 'pass', 'runtime_revision': PIN, 'runtime': 'Official Prism Windows x64 CUDA 12.4',
        'device': 'NVIDIA GeForce GTX 1080, compute capability 6.1', 'scope': 'Two addressed one-bit cells, three writes and three reads per cell',
        'memory': 'Append-only F16 attention KV, with nonrotary address match and monotonic rotary recency over the fixture',
        'routing': 'Only START is host-selected. Greedy stock logits select every subsequent opcode, including scrub, read, branch, commit and halt. The host supplies six signed payloads in one designated feedback lane only on model-selected write opcodes.',
        'feedback': 'The probe copies all h_nextn lanes byte-for-byte into the next call. Writes overlay the single payload lane. The model-selected scrub gates off attention and produces identical complete read input across different histories. Readout also feeds back into commit selection.',
        'limits': 'The fixed schedule demonstrates latest-write-wins for two addresses over a 34-position envelope. The append-only KV grows with calls, and rotary recency has a finite monotonic window.',
        'reset_order': 'A, B, C, A with the same MTP context cleared before each run. All A trace records matched.',
        'runtime_sha256': {name: digest_file(runtime / name) for name in ('llama.dll', 'ggml-cuda.dll')},
        'tiny_cpu': experiment('tiny-cpu'),
    }
    for model in ('tiny', 'combined') if source else ('tiny',):
        base, changed, restored = (model + '-' + kind for kind in ('cuda', 'mutated', 'restored'))
        manifest = load(model + '.manifest')
        original_trace, restored_trace = load(base), load(restored)
        expect({k: v for k, v in original_trace.items() if k != 'peak_working_set_bytes'} ==
               {k: v for k, v in restored_trace.items() if k != 'peak_working_set_bytes'}, 'Restored trace differs')
        restored_sha = (directory / (restored + '.sha256')).read_text(encoding='ascii').strip()
        changed_sha = (directory / (changed + '.sha256')).read_text(encoding='ascii').strip()
        expect(restored_sha == manifest['sha256'] == digest_file(directory / (model + '.gguf')) != changed_sha,
               'Missing checkpoint mutation or restore')
        row = {'manifest': manifest, 'restored_sha256': restored_sha, 'mutated_sha256': changed_sha,
               'cuda': experiment(base), 'mutated_cuda': experiment(changed, mutated=True),
               'restored_trace_exact': True, 'cuda_measurement': load(base + '.measurement')}
        log = (directory / (base + '.stderr.log')).read_text(encoding='utf-8-sig')
        row['stock_log_excerpt'] = [line for line in log.splitlines() if any(marker in line for marker in (
            'compute capability 6.1', 'offloaded ', 'model buffer size', 'llama_kv_cache: size =', 'llama_memory_recurrent: size ='))]
        if model == 'combined':
            expect(digest_file(source) == BONSAI_HASH and manifest['language_tensors_byte_equal'] == 851, 'Bonsai source mismatch')
            row['source_bonsai_sha256'] = BONSAI_HASH
            row['added_gguf_bytes'] = manifest['bytes'] - source.stat().st_size
            fields = ('visible_result', 'text_before', 'text_after_result', 'text_next_token', 'text_after_next')
            for run, reference in ((base, 'reference-cuda'), (changed, 'reference-mutated')):
                combined, original = load(run), load(reference)
                expect(combined['vocab'] == original['vocab'] == 248320, 'Incomplete text logit comparison')
                expect(combined['visible_result'] == combined['a_commits'], 'Text did not receive model-selected result')
                text = {key: combined[key] for key in fields}
                expect(text == {key: original[key] for key in fields}, 'Stock Bonsai text boundary differs')
                row[reference] = {'parked_unchanged': True, 'all_248320_logits_exact': True, 'boundaries': text,
                                  'measurement': load(reference + '.measurement'),
                                  'peak_working_set_bytes': original['peak_working_set_bytes']}
        report[model] = row

    report['reproduction'] = {
        'workdir': 'work/stock-mtp-ram',
        'prerequisites': ['Official extracted Prism b9596 Windows CUDA 12.4 archive and runtime DLLs',
                          'MSVC x64 developer shell, Python with NumPy and the runtime gguf-py in PYTHONPATH',
                          'Pinned include and ggml/include headers from the stated revision',
                          'Local original Bonsai checkpoint with the stated SHA-256 for the combined test'],
        'compile': 'cl.exe /nologo /std:c++17 /EHsc /W4 /WX /O2 /utf-8 /I"$headers/include" /I"$headers/ggml/include" "$probe" /link bcrypt.lib psapi.lib',
        'probe': 'probe_stock_mtp_ram.exe RUNTIME MODEL cpu|cuda tiny|combined|reference [actual guest result for reference]',
        'sequence': ['Export tiny.gguf and optionally combined.gguf with the commands below.',
                     'Run the tiny probe with CUDA_VISIBLE_DEVICES=-1 and cpu tiny, capturing stdout in tiny-cpu.json.',
                     'Set CUDA_VISIBLE_DEVICES=0 and run the probe with cuda tiny or cuda combined via measure-combined.py under the work directory. Use label tiny-cuda or combined-cuda.',
                     'For each model, use the mutate subcommand, run CUDA again with label tiny-mutated or combined-mutated, then run mutate --restore and rerun with label tiny-restored or combined-restored.',
                     'For combined-cuda and combined-mutated, use the original Bonsai with cuda reference and the corresponding visible_result. Measure using label reference-cuda or reference-mutated.',
                     'Run verify with the commands below, then remove the temporary tiny.gguf and combined.gguf.'],
        'export_tiny': 'python ../../overlay/experimental/standard-layer/export_stock_mtp_ram.py export tiny.gguf',
        'export_combined': 'python ../../overlay/experimental/standard-layer/export_stock_mtp_ram.py export combined.gguf --bonsai BONSAI',
        'measurement_example': 'python ../stock-prefix-io/measure-combined.py tiny-cuda .\\probe_stock_mtp_ram.exe RUNTIME tiny.gguf cuda tiny',
        'verify': 'python ../../overlay/experimental/standard-layer/export_stock_mtp_ram.py verify . RUNTIME REPORT --bonsai BONSAI',
    }
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('export')
    build.add_argument('output', type=Path)
    build.add_argument('--bonsai', type=Path)
    edit = sub.add_parser('mutate')
    edit.add_argument('model', type=Path)
    edit.add_argument('--restore', action='store_true')
    inspect = sub.add_parser('verify')
    inspect.add_argument('directory', type=Path)
    inspect.add_argument('runtime', type=Path)
    inspect.add_argument('output', type=Path)
    inspect.add_argument('--bonsai', type=Path)
    args = parser.parse_args()
    if args.command == 'export':
        export(args.output, args.bonsai)
    elif args.command == 'mutate':
        mutate(args.model, args.restore)
    else:
        verify(args.directory, args.runtime, args.output, args.bonsai)
