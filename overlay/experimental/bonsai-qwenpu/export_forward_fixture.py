"""Export the text-aware RV32/VX circuit and a parked fixture for packed Bonsai."""

import argparse
import hashlib
import json
from pathlib import Path

from pack_checkpoint import GGUFReader, build
from export_fixture import (
    FFN, HD, ONE, REF_COLS, VECTOR_SUM, Writer, assemble, core,
    export_embeddings, export_head, export_state, ref, spec,
)
from chipformer.q35.emit import allocate
from chipformer.q35.ir import Target
from chipformer.rv import bridge
from chipformer.rv.merge import pick_hidden

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "dist/local-bench/bonsai-forward"
TOKENIZER = ROOT / "dist/qwen3.5-0.8b-base/tokenizer.json"
GRAPH_NAME = "qwenpu.forward.graph"
PROGRAM_NAME = "qwenpu.forward.program.fixture"
VERSION_KEY = "qwenpu.forward_version"
VECTOR_BODY = VECTOR_SUM[VECTOR_SUM.index("    vput"):]

# Boot once to the initial park. On GO, read the first captured text byte,
# add five using the VX unit, then print the digit and park again.
FIRMWARE = f"""_start:
    li a7, 93
    ecall
    li t3, {spec.TEXT_ADDR}
    lbu t1, 16(t3)
    addi t1, t1, -48
    li t2, 5
{VECTOR_BODY}
    addi a0, a0, 48
    li a7, 1
    ecall
    li a7, 93
    ecall
"""


def checked_tokenizer(base, path):
    text = bridge.Text(path)
    field = base.fields["tokenizer.ggml.tokens"]
    vocabulary = text.tok.get_vocab(with_added_tokens=True)
    for token, index in vocabulary.items():
        if index >= len(field.data) or field.parts[field.data[index]].tobytes().decode() != token:
            raise ValueError(f"Bridge tokenizer differs from Bonsai at token ID {index}.")
    if text.n > len(field.data):
        raise ValueError("Bridge tokenizer exceeds Bonsai's vocabulary.")
    return text, len(field.data)


def feature_row(circuit, token):
    return {ONE: 1.0, **{d: circuit.feature_value(d, token) for d in circuit.token_fns}}


def write_text_embeddings(writer, circuit, text, vocabulary, width):
    writer.u32(vocabulary, width, core.FAM_ID["Q"], len(text.printable))
    writer.u32(*text.printable)
    # Ordinary input tokens differ only in their 16 captured bytes. The
    # printable tokens and tool delimiters also have compiled code/control bits.
    special_ids = sorted({text.TCO, text.TCC, *text.printable})
    ordinary = next(index for index in range(text.n) if index not in special_ids)
    generic = feature_row(circuit, bridge.text_token(ordinary))
    byte_features = [next(iter(feature.terms)) for byte in circuit.tb for feature in byte]
    for dim in byte_features:
        generic[dim] = 0
    writer.rows([generic, *({dim: 1} for dim in byte_features)])
    writer.file.write(b"".join(text.row_bytes(index) for index in range(vocabulary)))
    writer.u32(len(special_ids), *special_ids)
    writer.rows([feature_row(circuit, bridge.text_token(index)) for index in special_ids])


def write_graph(path, circuit, levels, text, vocabulary, width):
    dims = len(circuit.dims)
    with path.open("wb") as stream:
        stream.write(b"QPF1")
        writer = Writer(stream)
        writer.u32(dims, HD, REF_COLS[0], REF_COLS[-1] + 1, core.NFAM, core.NPAY,
                   spec.tok("GO"), spec.tok("HALT"), *spec.FAMILIES["BYTE"], len(levels))
        for nodes in levels:
            if isinstance(nodes[0], FFN):
                writer.u32(0)
                units = [(coef, a, b, node.out) for node in nodes for coef, a, b in node.terms]
                writer.rows([a.terms for _, a, _, _ in units])
                writer.rows([b.terms for _, _, b, _ in units])
                outputs = [{} for _ in range(dims)]
                for i, (coef, _, _, out) in enumerate(units):
                    outputs[out][i] = coef
                writer.rows(outputs)
            else:
                writer.u32(1)
                export_head(writer, nodes, dims)
        writer.rows([{dim: 1.0} for dim in circuit.keep])
        writer.u32(len(core.OUT_NAMES))
        for name in core.OUT_NAMES:
            base, size = spec.FAMILIES.get(name, (spec.CONTROLS.get(name, 0), 0))
            writer.u32(base, size, name in spec.CONTROLS)
            if name == "Q":
                writer.u32(0)
                writer.rows([])
            else:
                export_embeddings(writer, circuit, name, base, size)
        write_text_embeddings(writer, circuit, text, vocabulary, width)


def make_circuit(source, tokenizer):
    base = GGUFReader(source)
    text, vocabulary = checked_tokenizer(base, tokenizer)
    width = int(base.fields["qwen35.embedding_length"].contents())
    intermediate = int(base.fields["qwen35.feed_forward_length"].contents())
    keys = int(base.fields["qwen35.ssm.group_count"].contents())
    values = int(base.fields["qwen35.ssm.inner_size"].contents()) // HD
    circuit = core.build(Target(keys, values, intermediate), text, vx=True)
    layout = allocate(circuit, None, set(circuit.keep), intermediate)
    widened, ballast = pick_hidden(width, 1 + layout.n_slots)
    levels = [[node for node in circuit.nodes if node.level == level]
              for level in sorted({node.level for node in circuit.nodes})]
    return text, vocabulary, circuit, layout, widened, ballast, levels


def main(source, tokenizer, output):
    output.mkdir(parents=True, exist_ok=True)
    text, vocabulary, circuit, layout, widened, ballast, levels = make_circuit(source, tokenizer)
    graph = output / "forward.qpf"
    write_graph(graph, circuit, levels, text, vocabulary, widened)

    firmware = assemble(FIRMWARE)
    machine = ref.Machine.from_program(firmware)
    bridge.park(machine)
    core.snapshot(circuit, machine)
    snapshot = output / "fixture.qps"
    export_state(snapshot, levels)

    inputs = []
    references = []
    for digit, answer in (("3", "8"), ("1", "6")):
        prefix = text.encode("Compute") + [text.TCO] + text.encode(digit) + [text.TCC]
        session = bridge.Session(text, ref.Machine.from_program(firmware))
        bridge.park(session.m)
        session.qwen(prefix, max_records=128)
        if session.m.fault or session.T.decode_bytes(session.runs[0]) != (answer + "\n").encode():
            raise RuntimeError("Unexpected independent reference output.")
        ids = [t - spec.TEXT_TOKEN if spec.is_text(t) else vocabulary + t for t in session.toks]
        inputs.append({"digit": digit, "prefix": prefix})
        references.append({"digit": digit, "tokens": ids, "ownership": session.own, "answer": answer})

    (output / "inputs.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    (output / "reference.json").write_text(json.dumps(references, indent=2), encoding="utf-8")
    checkpoint = output / "Bonsai-27B-Q1_0-QwenPU-forward.gguf"
    verification = build(source, checkpoint, payloads={GRAPH_NAME: graph, PROGRAM_NAME: snapshot},
                         version_key=VERSION_KEY)
    verification.update(text_vocab=vocabulary, chip_dims=len(circuit.dims), allocated_slots=layout.n_slots,
                        widened_width=widened, ballast_slots=ballast, chip_layers=len(levels),
                        reference_sha256=hashlib.sha256((output / "reference.json").read_bytes()).hexdigest())
    (output / "checkpoint-verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    main(args.source, args.tokenizer, args.output)
