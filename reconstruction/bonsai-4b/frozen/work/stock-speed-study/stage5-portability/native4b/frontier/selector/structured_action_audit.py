"""Audit a structured action code from sparse 9B coefficients without model I/O.

From the qwenpu-bonsai repository root in PowerShell:
    $env:PYTHONPATH = (Resolve-Path 'work/prism-upstream/build-source/gguf-py').Path
    $dir = 'work/stock-speed-study/stage5-portability/native4b/frontier/selector'
    python -B "$dir/structured_action_audit.py" > "$dir/structured-action-audit.json"
"""

from collections import defaultdict, deque
from fractions import Fraction
from hashlib import sha256
from itertools import chain, product
import json
import math
from pathlib import Path
import random
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import selector_audit as selector
import plane_dot64 as dot_plane


def integer_rows(entries, action_start, actions):
    rows = defaultdict(dict)
    for (row, col), value in entries:
        phase = col - action_start
        if value and phase in actions:
            scaled = int(16 * value)
            assert scaled == 16 * value and phase not in rows[row]
            rows[row][phase] = scaled
    return rows


def classes(matrices, phases):
    signatures = defaultdict(list)
    for matrix_id, matrix in enumerate(matrices):
        for row, entries in matrix.items():
            for phase, value in entries.items():
                signatures[phase].append((matrix_id, row, value))
    groups = defaultdict(list)
    for phase in phases:
        groups[tuple(signatures[phase])].append(phase)
    return list(groups.values())


def exact_rank(rows, columns):
    pivots = {}
    for row in rows:
        remaining = {phase: Fraction(row[phase]) for phase in columns if phase in row}
        while remaining:
            col = max(remaining)
            lead = remaining[col]
            if col not in pivots:
                pivots[col] = {i: value / lead for i, value in remaining.items()}
                break
            for i, term in pivots[col].items():
                value = remaining.get(i, 0) - lead * term
                if value:
                    remaining[i] = value
                else:
                    remaining.pop(i, None)
    return len(pivots)


def pq2_tiles(group, phase_code):
    # 18 potential one-hot class positions and the largest bit occupy tile zero.
    # Eight smaller bits occupy tile one. The active squares sum exactly to 1.
    first = bytearray(struct.pack("<e", .5) + bytes([0x55]) * 32)
    second = bytearray(struct.pack("<e", .25) + bytes([0x55]) * 32)
    symbols = {group: 1, 18: 1 if phase_code & 1 else -1}
    for index, value in symbols.items():
        first[2 + index // 4] ^= (2 + value) << (2 * (index % 4))
    for index in range(8):
        value = 1 if phase_code & (2 << index) else -1
        second[2 + index // 4] ^= (2 + value) << (2 * (index % 4))
    # Canonical digit 1 -> digit 2 for +1 and digit 0 for -1.
    # XOR with 3 implements +1, XOR with 1 implements -1.
    return first + second


def decode_tiles(data):
    result = []
    for base in (0, 34):
        scale = struct.unpack_from("<e", data, base)[0]
        result.extend((((byte >> shift) & 3) - 1) * scale
                      for byte in data[base + 2:base + 34] for shift in (0, 2, 4, 6))
    return result


def sample_proof(circuit, ids, ordinary_up, protected_groups):
    s = selector.native.s
    trials = {"ADD_INIT": s.X, "PREFIX_1": s.G + 1, "PREFIX_2": s.P + 2, "ADD_FINISH": s.SUM}
    trial_ids = {ids[name] for name in trials}
    assert any(trial_ids <= set(group) for group in protected_groups)
    direct_rank = exact_rank(ordinary_up.values(), trial_ids)
    codes = dict(zip(trials, product((-1, 1), repeat=2), strict=True))
    terms = {name: circuit.effects[name][dest] for name, dest in trials.items()}
    input_lanes = sorted({lane for expr in terms.values() for _, coefficients in expr for lane, _ in coefficients})
    penalty = 4096 + max(512 * sum(abs(coef) for _, coef in coefficients)
                         for expr in terms.values() for _, coefficients in expr)

    def silu(x):
        if x >= 0:
            return x / (1 + math.exp(-x))
        return x * math.exp(x) / (1 + math.exp(x))

    def effect(expr, inputs, act, shift):
        plus, minus = silu(4 * act - shift), silu(-4 * act - shift)
        out = 0.0
        for factor, coefficients in expr:
            if coefficients:
                predicate = 512 * sum(coef * inputs[lane] for lane, coef in coefficients)
                diff = silu(predicate + 4 * act - shift) - silu(predicate - 4 * act - shift) - plus + minus
            else:
                diff = plus - minus
            out += 16 * act * factor * diff
        return out

    rng = random.Random(1322)
    amplitudes = (.03, .075, .15, .35, .75)
    matching = nonmatching = 0
    max_match_error = max_leakage = 0.0
    for act in amplitudes:
        for active, code in codes.items():
            for _ in range(256):
                inputs = {lane: act * rng.choice((-1, 1)) for lane in input_lanes}
                inputs[s.WORK + s.ONE] = act
                for candidate, want in codes.items():
                    distance = sum(a != b for a, b in zip(code, want, strict=True))
                    actual = effect(terms[candidate], inputs, act, act * penalty * distance)
                    if active == candidate:
                        expected = effect(terms[candidate], inputs, act, 0)
                        matching += 1
                        max_match_error = max(max_match_error, abs(actual - expected))
                    else:
                        nonmatching += 1
                        max_leakage = max(max_leakage, abs(actual))
            logits = {candidate: act * sum(a * b for a, b in zip(code, want, strict=True))
                      for candidate, want in codes.items()}
            logits["GUARD"] = 0.0
            winner, runner = sorted(logits, key=logits.__getitem__, reverse=True)[:2]
            assert winner == active and logits[winner] - logits[runner] == 2 * act
    assert direct_rank == 4 and max_match_error == 0 and max_leakage < 1e-30
    return dict(input_seed=1322, normalized_action_amplitudes=amplitudes,
                phases={phase: {"token_id": ids[phase], "destination": trials[phase], "code": code,
                                "effects": terms[phase]} for phase, code in codes.items()},
                existing_direct_up_exact_rank=direct_rank, trial_code_dimensions_including_reference=3,
                hamming_penalty_coefficient=penalty, matched_samples=matching, inactive_samples=nonmatching,
                matched_max_abs_error=max_match_error, inactive_max_abs_output=max_leakage,
                ideal_two_bit_score_margin_min=2 * min(amplitudes),
                formula="Use shared UP=16*a. Subtract a*penalty*HammingDistance from each SiLU input. The baseline pair receives the same shift.")


def audit():
    native = selector.native
    circuit, (ids, scores, gate, up, _, rows), ports, gates, _ = native.build()
    s = native.s
    names = tuple(ids)
    reachable = {s.SEED_MISS, s.SEED_WORD, s.START, s.FETCH}
    queue = deque(sorted(reachable))
    while queue:
        for next_phase in circuit.routes[queue.popleft()]:
            if next_phase not in reachable:
                reachable.add(next_phase)
                queue.append(next_phase)
    actions = sorted(ids[phase] for phase in reachable)
    q, k, v, _ = native.attention(circuit, ids, ports, gates)
    dot_start = rows - len(dot_plane.compile_ff(False, True).g) - 4 * len(gates)
    analog = {row for (row, col), value in up if value and col < s.ACTION}
    assert all(row >= dot_start for row in analog)
    entries = {"q": q.items(), "k": k.items(), "v": v.items(), "score": scores.items()}
    entries.update({"ordinary_gate": [entry for entry in gate if entry[0][0] < dot_start],
                    "ordinary_up": [entry for entry in up if entry[0][0] < dot_start],
                    "dot_analog_gate": [entry for entry in gate if entry[0][0] in analog],
                    "dot_analog_up": [entry for entry in up if entry[0][0] in analog],
                    "dot_gate": [entry for entry in gate if entry[0][0] >= dot_start],
                    "dot_up": [entry for entry in up if entry[0][0] >= dot_start]})
    active_ids = set(actions)
    matrices = {key: integer_rows(values, s.ACTION, active_ids) for key, values in entries.items()}
    protected_keys = ("q", "k", "v", "dot_gate", "dot_up")
    bundles = {key: (key,) for key in matrices}
    bundles.update({"qkv": ("q", "k", "v"), "protected": protected_keys,
                    "ordinary_up_and_score": ("ordinary_up", "score"), "existing_all": protected_keys +
                    ("ordinary_gate", "ordinary_up", "score")})
    census = {}
    for key, keys in bundles.items():
        group_matrices = [matrices[tensor] for tensor in keys]
        groups = classes(group_matrices, actions)
        action_rows = list(chain.from_iterable(matrix.values() for matrix in group_matrices))
        result = dict(nonzero_rows=len(action_rows), column_classes=len(groups),
                      real_rank_lower_bound_parity=selector.parity_rank(action_rows))
        if len(groups) <= 18:
            result["exact_real_rank"] = exact_rank(action_rows, [group[0] for group in groups])
            result["classes"] = [{"representative": names[group[0]], "size": len(group)} for group in groups]
        census[key] = result

    protected_groups = classes([matrices[key] for key in protected_keys], actions)
    assert len(protected_groups) == 18 and len(actions) == 510 and len(actions) <= 512
    groups_by_phase = {phase: group for group, phases in enumerate(protected_groups) for phase in phases}
    embedded = sha256()
    for code, phase in enumerate(actions):
        group = groups_by_phase[phase]
        tiles = pq2_tiles(group, code)
        decoded = decode_tiles(tiles)
        assert decoded[:18] == [.5 * (i == group) for i in range(18)]
        assert decoded[18] == (.5 if code & 1 else -.5)
        assert decoded[128:136] == [(.25 if code & (2 << j) else -.25) for j in range(8)]
        assert sum(x * x for x in decoded) == 1 and sum(x != 0 for x in decoded) == 10
        embedded.update(tiles)

    checked = 0
    for kind in protected_keys:
        for row in matrices[kind].values():
            for group in protected_groups:
                coefficient = row.get(group[0], 0) / 16
                doubled = 2 * coefficient
                expected = struct.unpack("<f", struct.pack("<f", doubled))[0]
                if kind in ("k", "v"):
                    assert expected == struct.unpack("<e", struct.pack("<e", doubled))[0]
                if kind == "dot_up":
                    assert struct.unpack("<I", struct.pack("<f", expected))[0] & 65535 == 0
                assert expected == doubled
                for phase in group:
                    assert .5 * expected == row.get(phase, 0) / 16
                    checked += 1

    return dict(source_sha256=sha256((selector.COMPILER / "compile_full9b.py").read_bytes()).hexdigest(),
                original_actions=len(ids), reachable_actions=len(actions),
                unreachable_actions=[phase for phase in names if phase not in reachable], original_ffn_rows=rows,
                ordinary_ffn_rows=dot_start, appended_dot_rows=rows - dot_start, dot_analog_up_rows=len(analog),
                census=census, protected_column_checks=checked,
                code=dict(protected_class_lanes=18, signed_id_lanes=9, active_code_lanes_per_action=10,
                          squared_embedding_norm=1, encoded_tile_sha256=embedded.hexdigest(),
                          existing_action_id_map="Assign dense 0..509 codes in original token ID order. The three unreachable actions have no code."),
                sampled_formula=sample_proof(circuit, ids, matrices["ordinary_up"], protected_groups),
                limits=["Counts use the pinned sparse compiler and conservative reachability. Every protected action-column contribution matches before runtime rounding.",
                        "The sampled plateau holds the action amplitude fixed and models each stock SiLU formula in Python. A stock fixture, native precision, cancellation, full state transport, multi-action FFN selections and next-action-code production still require validation.",
                        "Replacing scored eligibility with a code lookup requires model-owned generation of the next code, distinct from old input residual. Keep exact legacy score coefficients only if their dimension bound is acceptable."])


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
