"""Identify a load-bearing coefficient in the emitted BYTE record path."""

import argparse
import json
from pathlib import Path
import struct
from unittest.mock import patch

import numpy as np

from export_forward_fixture import (
    FIRMWARE, OUTPUT, TOKENIZER, assemble, bridge, core,
    make_circuit, ref, spec,
)
from chipformer.q35 import ir


def offset_of_output_coefficient(data, target_level, target_dim, target_unit):
    cursor = 48  # Magic and the eleven u32 header fields.

    def take():
        nonlocal cursor
        result, = struct.unpack_from("<I", data, cursor)
        cursor += 4
        return result

    def skip_matrix():
        nonlocal cursor
        for _ in range(take()):
            entries = take()
            cursor += entries * 8

    for level in range(target_level + 1):
        if take() == 0:
            skip_matrix()
            skip_matrix()
            if level != target_level:
                skip_matrix()
                continue
            for dim in range(take()):
                entries = take()
                if dim == target_dim:
                    for _ in range(entries):
                        if take() == target_unit:
                            return cursor
                        cursor += 4
                    break
                cursor += entries * 8
            raise RuntimeError("Selected FFN output coefficient was not exported.")
        _, values, _ = take(), take(), take()
        cursor += 8 * values
        for _ in range(7):
            skip_matrix()
    raise RuntimeError("Selected sublayer is not an FFN.")


def main(source, tokenizer, output):
    _, vocabulary, circuit, _, _, _, levels = make_circuit(source, tokenizer)
    machine = ref.Machine.from_program(assemble(FIRMWARE))
    bridge.park(machine)
    core.snapshot(circuit, machine)
    reference = json.loads((output / "reference.json").read_text(encoding="utf-8"))[0]
    ids = reference["tokens"]
    as_ir = lambda seq: [bridge.text_token(token) if token < vocabulary else token - vocabulary for token in seq]
    stream = as_ir(ids)
    byte = ids.index(vocabulary + spec.tok("BYTE", ord("8")))
    target_dim = circuit.keep[core.NFAM + 3]
    candidates = []
    evaluate_ffn = ir.eval_ffn

    def inspect_node(node, residual, chip, check):
        if node.out == target_dim:
            for index, (coef, a, b) in enumerate(node.terms):
                if coef <= 0:
                    continue
                active = float(coef) * ir.ev(a, residual) * np.maximum(ir.ev(b, residual), 0.0)
                if active[byte - 1] == 2 * core.KAPPA and np.array_equal(np.flatnonzero(active), [byte - 1]):
                    candidates.append((node, index))
        evaluate_ffn(node, residual, chip, check)

    def verify(seq):
        residual = ir.evaluate(circuit, seq, check=True)
        predictions = core.predictions(circuit, residual)
        observed = [predictions[i] for i, owns in enumerate(reference["ownership"][:-1]) if owns]
        expected = [seq[i + 1] for i, owns in enumerate(reference["ownership"][:-1]) if owns]
        if observed != expected:
            mismatch = next(i for i, (actual, wanted) in enumerate(zip(observed, expected)) if actual != wanted)
            raise RuntimeError(f"IR disagreed with expected trace at owned position {mismatch}.")

    with patch.object(ir, "eval_ffn", inspect_node):
        verify(stream)
    if not candidates:
        raise RuntimeError("No coefficient was active exclusively on the target BYTE record.")

    alternate = list(ids)
    alternate[byte] = vocabulary + spec.tok("BYTE", ord("0"))
    alternate[byte + 1] = circuit.text.byte_token[ord("0")]
    data = (output / "forward.qpf").read_bytes()
    for node, index in candidates:
        original = node.terms[index]
        node.terms[index] = (0, *original[1:])
        try:
            verify(as_ir(alternate))
        except (RuntimeError, ir.ExactnessError):
            continue
        finally:
            node.terms[index] = original
        sublayer = next(i for i, nodes in enumerate(levels) if any(n is node for n in nodes))
        units = 0
        for n in levels[sublayer]:
            if n is node:
                break
            units += len(n.terms)
        offset = offset_of_output_coefficient(data, sublayer, target_dim, units + index)
        original_bytes = struct.pack("<f", original[0])
        if data[offset:offset + 4] != original_bytes:
            raise RuntimeError("Serialized coefficient differs from the IR.")
        result = {
            "graph_offset": offset, "original_hex": original_bytes.hex(), "replacement_hex": "00000000",
            "node": node.name, "sublayer": sublayer, "unit": units + index,
            "coefficient": float(original[0]), "expected_first_divergence_stream_index": byte,
            "original_selected_id": ids[byte], "mutated_selected_id": alternate[byte],
            "original_echo_id": ids[byte + 1], "mutated_echo_id": alternate[byte + 1],
            "expected_original_output": "8", "expected_mutated_output": "0",
            "ir_invariants_checked": True,
        }
        (output / "mutation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return
    raise RuntimeError("Candidate mutations did not preserve the alternate byte and halt.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    main(args.source, args.tokenizer, args.output)
