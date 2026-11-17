"""Measure simple action-code savings before changing the Bonsai 4B circuit."""

from collections import defaultdict, deque
from hashlib import sha256
import json
from math import gcd
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
COMPILER = HERE.parent / "full9b/compiler"
sys.path.insert(0, str(COMPILER))
import compile_full9b as native


def parity_rank(rows):
    """Lower bound on real rank, reducing integer rows modulo two."""
    pivots = {}
    for row in rows:
        divisor = gcd(*row.values())
        parity = sum(1 << col for col, value in row.items() if value // divisor & 1)
        while parity:
            pivot = parity.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = parity
                break
            parity ^= pivots[pivot]
    return len(pivots)


def main():
    circuit, compiled, ports, gates, _ = native.build()
    ids, scores, gate, up, _, rows = compiled
    s = native.s
    # Include explicit boot and seed phases and the diagnostic fetch entry.
    # Ignore conditions here to include both outcomes of every branch.
    roots = {s.SEED_MISS, s.SEED_WORD, s.START, s.FETCH}
    reachable = set(roots)
    frontier = deque(sorted(roots))
    while frontier:
        phase = frontier.popleft()
        for target in circuit.routes[phase]:
            if target not in reachable:
                reachable.add(target)
                frontier.append(target)
    dead = set(ids) - reachable
    assert all(target in reachable for phase in reachable for target in circuit.routes[phase])

    q, k, v, _ = native.attention(circuit, ids, ports, gates)
    logical = {s.ACTION + v: k for k, v in ids.items()}
    tensors = {"gate": gate, "up": up, "q": q.items(), "k": k.items(), "v": v.items(),
               "score": scores.items()}
    signatures = defaultdict(list)
    matrix = defaultdict(dict)
    for tensor, entries in tensors.items():
        for (row, lane), coefficient in entries:
            if coefficient and lane in logical:
                signatures[logical[lane]].append((tensor, row, float(coefficient)))
                if tensor in ("up", "score"):
                    col = lane - s.ACTION
                    assert coefficient * 16 == int(coefficient * 16)
                    assert col not in matrix[tensor, row]
                    matrix[tensor, row][col] = int(coefficient * 16)
    groups = defaultdict(list)
    for phase in sorted(reachable, key=ids.__getitem__):
        groups[tuple(sorted(signatures[phase]))].append(phase)
    shareable = [bucket for bucket in groups.values() if len(bucket) > 1]
    live_up_rows = {row for (row, lane), value in up
                    if value and (lane not in logical or logical[lane] in reachable)}
    report = {
        "scope": "Conservative symbolic reachability and exact column equality with the existing tensor consumers.",
        "compiler_sha256": sha256((COMPILER / "compile_full9b.py").read_bytes()).hexdigest(),
        "roots": sorted(roots), "action_ids": len(ids), "reachable": len(reachable),
        "unreachable_actions": sorted(dead, key=ids.__getitem__),
        "exactly_identical_reachable_action_columns": shareable,
        "reachable_action_column_classes": len(groups),
        "maximum_direct_action_lane_saving": len(ids) - len(groups),
        "exact_action_rank_lower_bounds": {
            "up": parity_rank(values for (tensor, _), values in matrix.items() if tensor == "up"),
            "score": parity_rank(values for (tensor, _), values in matrix.items() if tensor == "score"),
            "up_and_score": parity_rank(matrix.values()),
        },
        "ffn_rows_baseline": rows,
        "ffn_rows_disabled_by_reachability": rows - len(live_up_rows),
    }
    encoded = json.dumps(report, indent=2) + "\n"
    (HERE / "selector-audit.json").write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
