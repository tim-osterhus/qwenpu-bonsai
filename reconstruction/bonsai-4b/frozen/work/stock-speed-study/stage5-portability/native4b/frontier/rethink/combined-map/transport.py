"""Static launcher ABI and CPU checks for the production seed/feedback boundary."""

from functools import lru_cache
from hashlib import sha256
from math import exp, isqrt, sqrt
import struct

import numpy as np


AMPLITUDE = 16
SIGN_ENERGY = 89
UNIT_ENERGY = 512
VALUE_REFERENCE = 1 / 32
EPS = float(np.float32(1e-6))
WROM_FILE_ROW_BYTES = 72


@lru_cache(maxsize=1)
def square_pairs():
    pairs = {}
    for first in range(isqrt(64 * (UNIT_ENERGY - SIGN_ENERGY - 1)) + 1):
        for second in range(first + 1):
            pairs.setdefault(first * first + second * second, (first, second))
    return pairs


@lru_cache(maxsize=4097)
def seed_anchors(nibble_energy):
    """Return four scores in eighths of the common raw seed scale."""
    needed = 64 * (UNIT_ENERGY - SIGN_ENERGY - 1) - nibble_energy
    pairs = square_pairs()
    # Index one side of the partition. The fixed iteration order is part of
    # the reproducible immutable payload encoding.
    for subtotal, left in pairs.items():
        if (right := pairs.get(needed - subtotal)) is not None:
            return (*left, *right)
    raise ValueError(f"No four-square seed completion for {needed}")


def put_bits(row, positions, value):
    for bit, coordinate in enumerate(positions):
        row[coordinate] = AMPLITUDE if value & (1 << bit) else -AMPLITUDE


def seed_row(abi, address, word, handle, nibbles=None):
    """Construct one stock h_embd input from immutable file seed data."""
    row = np.zeros(abi["width"], dtype=np.float32)
    ingress = abi["raw_ingress"]
    for name, value in (("write_address", address), ("write_word", word), ("fresh_handle", handle)):
        lanes = ingress[name]
        if not 0 <= value < 1 << len(lanes):
            raise ValueError(f"Seed {name} does not fit {len(lanes)} bits")
        put_bits(row, lanes, value)
    if nibbles is not None:
        if len(nibbles) != 64 or any(not -8 <= n <= 7 or n != int(n) for n in nibbles):
            raise ValueError("Expected 64 signed four-bit values")
        row[ingress["wrom_signed_nibbles"]] = np.asarray(nibbles) * AMPLITUDE / 8
        row[ingress["wrom_reference"][0]] = AMPLITUDE
        squares = seed_anchors(sum(int(n) ** 2 for n in nibbles))
        row[ingress["wrom_norm_anchors"]] = np.asarray(squares) * AMPLITUDE / 8
    return row


def elf_wrom_row(raw_weights, index):
    """Decode the file's first 64 low nibbles, using the existing offset-binary ABI."""
    start = index * WROM_FILE_ROW_BYTES
    if index < 0 or start + WROM_FILE_ROW_BYTES > len(raw_weights):
        raise ValueError("ELF WROM row is out of range")
    return tuple((byte & 15) - 8 for byte in raw_weights[start:start+64])


def launcher_abi(details, place, side, anchors, fields, read_tokens, tile, range_guard, s, dot):
    _, ids, _, codes, _, selector, _, _, _, _, _, _, _, _, ingress, _, _, _, _, _ = details
    raw = place["fixed_raw"] | place["raw"]

    def physical(lanes, map_):
        return [map_[lane] for lane in lanes]

    inactive_field_ids = {"tile_extra_0": [ids[tile]], "tile_extra_1": [ids[tile]]}
    assert ids[tile] not in codes
    output_fields = {name: physical(lanes, raw) for name, lanes in fields.items()
                     if name not in inactive_field_ids and name not in ("fresh_handle", "entry")}
    output_fields["output_byte"] = output_fields["exit_code"][:9]
    raw_ingress = {name: [raw.get(lane, lane) for lane in fields[name]] for name in
                   ("write_address", "write_word", "fresh_handle", "entry", "read_address", "read_marker")}
    raw_ingress["wrom_signed_nibbles"] = [col for lane, col in side.items() if lane != dot.SIDE_PAYLOAD + 256]
    raw_ingress["wrom_reference"] = [side[dot.SIDE_PAYLOAD + 256]]
    raw_ingress["wrom_norm_anchors"] = list(anchors.values())
    ordinary_ingress = {col for name, cols in raw_ingress.items() if not name.startswith("wrom_") for col in cols}
    side_columns = set(side.values()) | set(anchors.values())
    assert len(ordinary_ingress) == 155 and ordinary_ingress.isdisjoint(side_columns) and len(side_columns) == 69
    assert not side_columns.intersection(place["head_feedback"])
    assert set(raw_ingress["fresh_handle"] + raw_ingress["entry"]).isdisjoint(place["head_feedback"])
    assert all(len(word) == len(set(word)) for word in output_fields.values())
    assert set().union(*(set(word) for word in output_fields.values())) <= set(place["head_feedback"])
    assert set(raw_ingress["write_address"] + raw_ingress["write_word"]) <= set(place["fixed_raw"].values())
    projected = {name: [place["work"][i] for i in fields[name]] for name in
                 ("write_address", "write_word", "fresh_handle", "entry", "read_address", "read_marker")}
    for name, raw_lanes in raw_ingress.items():
        if name.startswith("wrom_"):
            projected[name] = [place["side"][lane] for lane, col in (side | anchors).items() if col in raw_lanes]
        assert set(raw_lanes).isdisjoint(projected[name])
    assert len(ingress) >= len(ordinary_ingress) and selector.input_roles == 67
    return dict(width=place["width"], token_ids=ids,
                packed_code_by_action_token=codes, inactive_tokens=sorted(set(ids.values()) - codes.keys()),
                raw_ingress=raw_ingress, projected_ingress=projected, feedback_fields=output_fields,
                unreachable_feedback_fields=inactive_field_ids,
                read_tokens=[ids[name] for name in read_tokens], wrom_bound_tokens=[ids[name] for name in range_guard],
                legacy_tile_token=ids[tile], retire_tokens=[ids[name] for name in (s.COMMIT, s.HALT, s.FAULT)],
                write_tokens=[ids[name] for name in (s.STORE, s.REGISTER)], dot_va2_token=ids["DOT64_UNUSED_4"],
                seed_input=dict(amplitude=AMPLITUDE, typed_address_order="32 numeric bits, then register namespace",
                                seed_v_reference=VALUE_REFERENCE,
                                wrom_address_base=dot.SIDE_BASE, wrom_word_stride=9,
                                nibble_encoding="offset binary: (ELF byte & 15) - 8",
                                formula="raw = 16 * nibble / 8; four loader anchors complete raw and EH energy"))


def f32(x):
    return np.float32(x)


def rms_normalizer(energy, width):
    # CPU arithmetic model of the stock F32 RMS order. Energy is computed
    # over each projected row separately, including the four e_norm values.
    return f32(1 / f32(sqrt(f32(energy / width) + f32(EPS))))


def normalize_inputs(hidden, token_embedding):
    width = hidden.shape[-1]
    weight = f32(1 / sqrt(width))
    hn = (hidden * rms_normalizer(norm_energy(hidden), width)) * weight
    en = (token_embedding * rms_normalizer(1, width)) * weight
    return np.concatenate((en, hn))


def norm_energy(values):
    # The CPU kernel multiplies in F32 before adding to a wider sum.
    return float(np.sum(values * values, dtype=np.float64))


def energy_check():
    energies = {0}
    for _ in range(64):
        energies = {older + n * n for older in energies for n in range(-8, 8)}
    largest = max(max(seed_anchors(energy)) for energy in energies)
    assert len(energies) == 4038
    assert all(sum(n*n for n in seed_anchors(energy)) + energy == 64 * (UNIT_ENERGY - SIGN_ENERGY - 1)
               for energy in energies)
    return dict(reachable_nibble_sum_square_energies=len(energies), largest_raw_anchor=AMPLITUDE * largest / 8)


def file_encoding_check():
    values = bytes(range(256))
    rows = b"".join(values[start:start+64] + bytes([255]) * 8 for start in range(0, len(values), 64))
    nibbles = tuple(n for row in range(4) for n in elf_wrom_row(rows, row))
    for raw, signed in zip(values, nibbles, strict=True):
        planes = (raw & 15) ^ 8
        wide = sum(((planes >> bit) & 1) * weight for bit, weight in enumerate((1, 2, 4, -8)))
        assert wide == signed
    return dict(equivalent_file_byte_values=256, encoded_rows=4, unmodified_tail_bytes_per_row=8)


def check_pinned_wrom_elf(path, abi):
    data = path.read_bytes()
    digest = sha256(data).hexdigest()
    assert digest == "5c50c2ac365a5e342f4abf236848751c69e46219712a76ec7f33ea6525a72ed2"
    assert data[:7] == b"\x7fELF\x01\x01\x01"
    first = struct.unpack_from("<I", data, 28)[0]
    ph_size, ph_count = struct.unpack_from("<HH", data, 42)
    assert ph_size == 32
    wrom_address, count = 0x100000, 4608
    weights = bytearray(WROM_FILE_ROW_BYTES * count)
    for i in range(ph_count):
        ptype, offset, address, _, filesz, memsz, flags, _ = struct.unpack_from("<8I", data, first + i * ph_size)
        if ptype == 1 and address >= wrom_address:
            start = address - wrom_address
            assert 0 <= filesz <= memsz <= len(weights) - start and flags == 4
            weights[start:start+filesz] = data[offset:offset+filesz]
    encoded = sha256()
    all_nibbles = set()
    base = abi["seed_input"]["wrom_address_base"]
    for i in range(count):
        nibbles = elf_wrom_row(weights, i)
        all_nibbles.update(nibbles)
        row = seed_row(abi, base + 36*i, 0, i + 1, nibbles)
        assert norm_energy(row) == UNIT_ENERGY * AMPLITUDE**2
        encoded.update(row.astype("<f4", copy=False).tobytes())
    assert all_nibbles == set(range(-8, 8))
    return dict(elf_sha256=digest, raw_wrom_sha256=sha256(weights).hexdigest(),
                all_signed_nibbles_seen=True, raw_rows_of_pinned_labels_checked=count,
                synthetic_handle_base=1, raw_input_witness_sha256=encoded.hexdigest())


def apply_coefficients(matrix, inputs, length):
    result = np.zeros(length, dtype=np.float32)
    for (row, col), coeff in matrix.items():
        result[row] += f32(coeff) * inputs[col]
    return result


def seed_and_transport_checks(details, place, abi, projections, s, full, ports, gates_dot):
    """Evaluate real Q/K/V sparse columns at each distinct launcher ingress."""
    _, ids, _, codes, _, selector, *_ = details
    width, head, groups = abi["width"], full.HEAD, full.GROUPS
    raw, projected = abi["raw_ingress"], abi["projected_ingress"]
    eh = projections["EH"]
    expected_side = {row: col for label in ("wrom_signed_nibbles", "wrom_reference", "wrom_norm_anchors")
                     for row, col in zip(projected[label], raw[label], strict=True)}
    assert all(eh[row, width + col] == 1 for row, col in expected_side.items())
    assert not set(projected["wrom_norm_anchors"]) & {col for name in ("Q", "K", "V") for _, col in projections[name]}

    def inputs_for(action, row):
        embed = np.zeros(width, dtype=np.float32)
        for col, value in selector.embedding(codes[ids[action]]):
            embed[place["input"][col]] = value
        projected_input = apply_coefficients(eh, normalize_inputs(row, embed), width)
        energy = norm_energy(projected_input)
        if np.any(row):
            assert 1.99 < energy < 2
        return (projected_input * rms_normalizer(energy, width)) * f32(1 / sqrt(width))

    def cache(action, row):
        inp = inputs_for(action, row)
        k = apply_coefficients(projections["K"], inp, groups * head).reshape(groups, head)
        for i, key in enumerate(k):
            key_energy = norm_energy(key)
            assert not np.any(key[:s.ROTARY])
            k[i] = (key * rms_normalizer(key_energy, head)) * 8
        v = apply_coefficients(projections["V"], inp, groups * head).reshape(groups, head)
        return k.astype(np.float16), v.astype(np.float16)

    witness = sha256()
    profiles = ([0] * 64, [-8] * 64, [7] * 64,
                [((3 * i + i // 16) & 15) - 8 for i in range(64)])
    dot_base = full.dot.SIDE_BASE
    signed_channels = slice(64, 80)
    sentinel = s.ROTARY + 33 * s.REPEATS
    reference_channels = slice(128, 136)
    checked = 0
    worst_readback = 1.
    # All-zero ordinary payloads are encoded as 32 negative signs. Only
    # immutable Dot64 seeds carry W/reference lanes. Test both in one
    # source batch as well as a short final batch reusing its staging array.
    rows = [(s.SEED_MISS, np.zeros(width, dtype=np.float32), None, 0, 0, 0)]
    for i, nibbles in enumerate(profiles):
        for typ in (0, 1):
            address, word, handle = dot_base + i * 36 | (typ << 32), 0xFA5A5731 ^ i, 1 + 2 * i + typ
            rows.append((s.SEED_WORD, seed_row(abi, address, word, handle, nibbles), nibbles, address, word, handle))
    address, word, handle = 0x0007_FFFC, 0xAAA55335, 9
    rows.append((s.SEED_WORD, seed_row(abi, address, word, handle), None, address, word, handle))
    storage = np.empty((7, width), dtype=np.float32)
    for first in range(0, len(rows), 7):
        batch = rows[first:first+7]
        storage[:] = 0
        for i, (_, row, *_) in enumerate(batch):
            storage[i] = row
        assert not np.any(storage[len(batch):])
        for pos, (action, row, nibbles, address, word, handle) in enumerate(batch):
            assert np.array_equal(row, storage[pos])
            keys, v = cache(action, storage[pos])
            expected_w = np.zeros((groups, 16), dtype=np.float16)
            expected_ref = np.zeros((groups, 8), dtype=np.float16)
            if action == s.SEED_MISS:
                assert not np.any(keys[:, :sentinel]) and np.all(keys[:, sentinel] > 0)
                assert np.all(v[0, :57] < 0)
            else:
                keysigns = np.array([s.sign(address, bit) for bit in range(33)])
                keybits = keys[:, s.ROTARY:sentinel].reshape(groups, 33, s.REPEATS)
                assert np.all(np.sign(keybits) == keysigns[None, :, None]) and not np.any(keys[:, sentinel:])
                assert np.all(np.sign(v[:, :32]) == [s.sign(word, bit) for bit in range(32)])
                assert np.all(np.sign(v[0, 32:56]) == [s.sign(handle, bit) for bit in range(24)])
                assert v[0, 56] > 0
                worst_readback = min(worst_readback, float(np.min(np.abs(v[0, :57]))))
                if nibbles is not None:
                    assert float(row @ row) == UNIT_ENERGY * AMPLITUDE**2
                    expected_w[:] = np.asarray(nibbles).reshape(groups, 16) / 256
                    expected_ref[0] = VALUE_REFERENCE
                    assert np.all(np.abs(v[:, :32]) == VALUE_REFERENCE)
                    assert np.all(np.abs(v[0, 32:56]) == VALUE_REFERENCE)
            assert np.array_equal(v[:, signed_channels], expected_w)
            assert not np.any(v[:, 80:128]) and not np.any(v[:, 136:])
            assert not np.any(np.signbit(v[:, 64:128][v[:, 64:128] == 0]))
            assert np.array_equal(v[:, reference_channels], expected_ref)
            witness.update(v.tobytes())
            witness.update(keys.tobytes())
            checked += 1

    ordinary = {}
    for action, assignments in (
            (s.START, {"entry": 0x00044000}),
            (s.FETCH, {"read_address": 0x16543ABCD, "read_marker": 1}),
            (s.STORE, {"write_address": 0x00007FFC, "write_word": 0xAAF55A01, "fresh_handle": 17}),
            (s.REGISTER, {"write_address": 0x10000001F, "write_word": 0x8000000B, "fresh_handle": 42}),
            (full.dot.CP, {}), (full.dot.FN, {})):
        row = np.zeros(width, dtype=np.float32)
        for field, value in assignments.items():
            put_bits(row, raw[field], value)
        inp = inputs_for(action, row)
        assert not np.any(inp[list(expected_side)])
        assert all(np.array_equal(np.sign(inp[projected[field]]),
                                  [s.sign(value, bit) for bit in range(len(raw[field]))])
                   for field, value in assignments.items())
        keys, v = cache(action, row)
        assert not np.any(v[:, 64:136])
        if action in (s.STORE, s.REGISTER):
            assert np.all(np.sign(v[:, :32]) == [s.sign(assignments["write_word"], bit) for bit in range(32)])
            assert np.all(np.sign(v[0, 32:56]) == [s.sign(assignments["fresh_handle"], bit) for bit in range(24)])
            assert v[0, 56] > 0
        ordinary[action] = dict(raw_nonzero=int(np.count_nonzero(row)), wrom_channels_zero=True)
    attention = attention_checks(details, place, abi, projections, s, full, ports, gates_dot,
                                 inputs_for, cache)
    attention["va_and_scalar_readback"] = va_and_readback_checks(
        details, place, abi, projections, s, full, inputs_for, cache)
    return dict(energy=energy_check(), file_encoding=file_encoding_check(),
                mixed_batch_sizes=[7, 3], f32_equation_profiles=checked,
                cache_f16_profile_sha256=witness.hexdigest(), minimum_profile_handle_word_or_hit=worst_readback,
                ordinary_ingress_controls=ordinary, expected_wrom_v="n/256", expected_reference_v="1/32",
                seed_raw_squared_norm=UNIT_ENERGY * AMPLITUDE**2, projected_hnorm_squared_norm="approximately 1",
                projected_action_squared_norm="approximately 1", typed_attention=attention)


def attention_checks(details, place, abi, projections, s, full, ports, gates_dot, inputs_for, cache):
    dot, head, groups = full.dot, full.HEAD, full.GROUPS
    width, qheads = place["width"], full.QUERY
    wrom_word = abi["feedback_fields"]["wrom_index"]
    raw = place["fixed_raw"] | place["raw"]
    scratch = place["scratch"]
    wrom_channels = range(64, 128)
    nibble_bits = tuple(range(-8, 8)) * groups
    zero = np.zeros(width, dtype=np.float32)
    correct_address = dot.SIDE_BASE + 4 * 99
    profiles = {
        "matching": cache(s.SEED_WORD, seed_row(abi, correct_address, 0, 1, nibble_bits)),
        "same_address_wrong_type": cache(s.SEED_WORD, seed_row(
            abi, correct_address | 1 << 32, 0xffffffff, 2, tuple(reversed(nibble_bits)))),
        "one_bit_wrong_address": cache(s.SEED_WORD, seed_row(
            abi, correct_address ^ 4, 0xaaaa5555, 3, tuple(reversed(nibble_bits)))),
        "scalar_same_low_address": cache(s.SEED_WORD, seed_row(abi, correct_address & 0xffff, 0x55AA55AA, 4)),
        "miss_sentinel": cache(s.SEED_MISS, zero),
        "current_scratch": cache(dot.CP, zero),
    }
    active_profiles = [key for key in profiles if key != "matching"]
    sign_profiles = ((dot.CP, 1, 0), (dot.CN, -1, 0),
                     (dot.FP, 1, 23), (dot.FN, -1, 17))
    min_margins = {name: float("inf") for name in active_profiles}
    basis_checks = 0
    for action, sign, quotient in sign_profiles:
        for selected in range(64):
            row = zero.copy()
            put_bits(row, wrom_word, 99)
            put_bits(row, [raw[s.ZERO]], 0)
            put_bits(row, [raw[s.ONE]], 1)
            put_bits(row, [raw[lane] for lane in gates_dot], 1 << selected)
            put_bits(row, [raw[lane] for lane in ports["Q"]], quotient)
            inputs = inputs_for(action, row)
            q = apply_coefficients(projections["Q"], inputs, qheads * 2 * head).reshape(qheads, 2, head)
            value_in = np.zeros((qheads, head), dtype=np.float32)
            fine = action in (dot.FP, dot.FN)
            for group in range(groups):
                pos, neg = 4 * group + 2, 4 * group + 3
                for hi in (pos, neg):
                    query, gate = q[hi]
                    assert not np.any(query[:s.ROTARY])  # no RoPE on the typed address lanes
                    query_energy = norm_energy(query)
                    normalized = (query * rms_normalizer(query_energy, head)) * 8
                    scores = {name: float(normalized @ pairs[0][group].astype(np.float32)) / sqrt(head)
                              for name, pairs in profiles.items()}
                    for profile in active_profiles:
                        min_margins[profile] = min(min_margins[profile], scores["matching"] - scores[profile])
                    # Present only stored V. This tests the equation in the
                    # near-one attention limit, with wrong-row bounds below.
                    actual_gates = gate[list(wrom_channels)]
                    assert np.all(np.abs(actual_gates) > 100)
                    expected_live = selected % 16 if group == selected // 16 and hi == (pos if sign == 1 else neg) else -1
                    assert all((coeff > 0) == (n % 16 == expected_live) for n, coeff in enumerate(actual_gates))
                    value_in[hi] = np.where(gate > 100, profiles["matching"][1][group], 0)
            outputs = apply_coefficients(projections["O"], value_in.ravel(), width)
            reference = VALUE_REFERENCE
            correction = (16 - quotient) if fine else 0
            assert outputs[scratch[dot.PLANE]] == reference * (sign * nibble_bits[selected] + 32 * correction) / 512
            assert outputs[scratch[dot.REF]] == outputs[scratch[dot.REF2]] == reference
            basis_checks += 1
    assert all(margin > 61 for margin in min_margins.values())
    assert min_margins["miss_sentinel"] > 1000 and min_margins["current_scratch"] > 1000
    # Conservative N*exp(-margin) bound with one matching row. This is a
    # score envelope for unit-sign keys at the tested F16 scales.
    worst_mass = 131071 * exp(-min(min_margins.values()))
    assert worst_mass < 1e-21
    return dict(typed_bits=33, key_repeats=s.REPEATS,
                wrom_gate_and_reference_controls=basis_checks, tested_profiles=active_profiles,
                minimum_score_margins=min_margins, max_wrong_mass_at_131072_cells=worst_mass,
                dot_ref="1/32", fine_quotient_reference_rule="plane/reference = (signed nibble sum + 512 - 32*q)/512")


def va_and_readback_checks(details, place, abi, projections, s, full, inputs_for, cache):
    c = details[0]
    dot, head = full.dot, full.HEAD
    width, qheads = place["width"], full.QUERY
    fields = abi["raw_ingress"]
    scratch = place["scratch"]
    raw = place["fixed_raw"] | place["raw"]
    marker = c.vx.fields["tile_weights"][:8]
    words = [(0x9A7EC811 + i * 0x2413969) & 0xffffffff for i in range(16)]
    base = 0x4200
    zero = np.zeros(width, dtype=np.float32)
    block = {base + 4*i: cache(s.SEED_WORD, seed_row(abi, base + 4*i, word, i + 1))
             for i, word in enumerate(words)}
    sentinel = cache(s.SEED_MISS, zero)
    rmw = cache(s.SEED_WORD, seed_row(abi, base | 1 << 32, ~words[0] & 0xffffffff, 34))
    queries = 0
    match_to_sentinel = []
    sentinel_to_mismatch = []

    def q_and_gate(action, address, active_bit=None):
        row = zero.copy()
        put_bits(row, fields["read_address"], address)
        put_bits(row, fields["read_marker"], 1)
        if active_bit is not None:
            put_bits(row, [raw[i] for i in marker], 1 << active_bit)
        inputs = inputs_for(action, row)
        return apply_coefficients(projections["Q"], inputs, qheads * 2 * head).reshape(qheads, 2, head)

    def logits(query, group, wanted, wrong):
        normed = (query * rms_normalizer(norm_energy(query), head)) * 8
        scale = 1 / sqrt(head)
        assert not np.any(normed[:s.ROTARY])
        against = []
        for keys, _ in (wanted, sentinel, wrong):
            against.append(float(normed @ keys[group].astype(np.float32)) * scale)
        best, miss, bad = against
        assert best > miss > bad
        match_to_sentinel.append(best - miss)
        sentinel_to_mismatch.append(miss - bad)

    for bit, va1 in enumerate(dot.VA):
        for index in range(33):
            assert c.effects[va1][s.READ_ADDR + index] == s.buffer(s.WORK + s.READ_ADDR + index,
                                                                  -1 if index == 3 else 1)
        for step, action in enumerate((va1, full.SECOND)):
            q = q_and_gate(action, base + 8 * step, bit)
            values = np.zeros((qheads, head), dtype=np.float32)
            expected_signs = {}
            for group in range(full.GROUPS):
                for sub in (0, 1):
                    hi = 4*group + sub
                    chosen = 4*group + sub + 2*step
                    addr = base + 4*chosen
                    query, gate = q[hi]
                    logits(query, group, block[addr], block[addr ^ 4])
                    assert np.all(np.abs(gate[:32]) > 100)
                    assert all((coeff > 0) == (i % 8 == bit) for i, coeff in enumerate(gate[:32]))
                    values[hi] = np.where(gate > 100, block[addr][1][group], 0)
                    expected_signs.update({scratch[dot.VA_OUT + group*16 + sub*4 + byte]:
                                           s.sign(words[chosen], 8*byte + bit) for byte in range(4)})
                    queries += 1
            out = apply_coefficients(projections["O"], values.ravel(), width)
            assert all(np.sign(out[lane]) == sign for lane, sign in expected_signs.items())
    for address, row, opposite, word, handle in (
            (base, block[base], rmw, words[0], 1),
            (base | 1 << 32, rmw, block[base], ~words[0] & 0xffffffff, 34)):
        q = q_and_gate(s.FETCH, address)[0]
        logits(q[0], 0, row, opposite)
        assert np.all(q[1, :57] > 32)
        for match in (True, False):
            values = np.zeros((qheads, head), dtype=np.float32)
            chosen = row if match else sentinel
            values[0] = np.where(q[1] > 32, chosen[1][0], 0)
            out = apply_coefficients(projections["O"], values.ravel(), width)
            assert np.all(np.sign(out[abi["feedback_fields"]["readback_value"]]) ==
                          [s.sign(word if match else 0, bit) for bit in range(32)])
            assert np.all(np.sign(out[abi["feedback_fields"]["readback_handle"]]) ==
                          [s.sign(handle if match else 0, bit) for bit in range(24)])
            assert np.sign(out[abi["feedback_fields"]["readback_hit"][0]]) == (1 if match else -1)
    return dict(va_heads=8, va_queries=queries, va_offset_words=[0, 2],
                scalar_typed_fetch_and_miss_controls=4,
                minimum_matched_over_miss_score=min(match_to_sentinel),
                minimum_miss_over_adjacent_or_wrong_type_score=min(sentinel_to_mismatch))
