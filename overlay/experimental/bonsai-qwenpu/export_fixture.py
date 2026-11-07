"""Export the pinned QwenPU vector fixture as sparse graph weights."""
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "dist/local-bench"
UPSTREAM = BENCH / "qwenpu-trace/upstream"
sys.path.insert(0, str(UPSTREAM))

import chipformer
if str(UPSTREAM / "chipformer") not in chipformer.__path__:
    chipformer.__path__.insert(0, str(UPSTREAM / "chipformer"))

from chipformer.dsl.graph import ONE
from chipformer.q35.ir import FFN, HD, REF_COLS
from chipformer.rv import core, ref, spec
from chipformer.rv.asm import assemble

VECTOR_SUM = """    li t1, {left}
    li t2, {right}
    vput zero, t1, t2
    li t0, 2048
    li t1, 0
    li t2, 32767
    vq t0, t1, t2
    li t0, 130
    vred t0, zero, zero
    li t1, 1
    vget a0, t1, zero
"""
PRINT_DIGIT = """\
    addi a0, a0, 48
    li a7, 1
    ecall
    li a7, 93
    ecall
"""
CANARY_ADDRESS = 256
PROGRAMS = (
    ("_start:\n    li s0, 3\n    sw s0, 256(zero)\n"
     + VECTOR_SUM.format(left=3, right=5) + PRINT_DIGIT),
    ("_start:\n    lw s0, 256(zero)\n"
     + VECTOR_SUM.format(left=2, right=4) + "    add a0, a0, s0\n" + PRINT_DIGIT),
)


class Writer:
    def __init__(self, file):
        self.file = file

    def u32(self, *values):
        self.file.write(struct.pack(f"<{len(values)}I", *values))

    def rows(self, rows):
        self.u32(len(rows))
        for row in rows:
            entries = [(index, float(value)) for index, value in sorted(row.items()) if value]
            self.u32(len(entries))
            self.file.write(b"".join(struct.pack("<If", index, value) for index, value in entries))


def export_head(writer, heads, dims):
    pairs = [(i, vd) for i, head in enumerate(heads) for vd in head.vheads()]
    writer.u32(len(heads), len(pairs), max(getattr(head, "qhot", 2) for head in heads))
    writer.u32(*(i for i, _ in pairs))
    writer.u32(*(vd.beta is None for _, vd in pairs))

    def taps(ch):
        return {lag * dims + d: float(value) * tap
                for lag, tap in enumerate(ch.taps) if tap
                for d, value in ch.expr.terms.items()}

    for attr in ("q", "k"):
        writer.rows([taps(channels[col]) if col in channels else {}
                     for head in heads for channels in [getattr(head, attr)] for col in range(HD)])
    writer.rows([taps(vd.v[col]) if col in vd.v else {} for _, vd in pairs for col in range(HD)])
    writer.rows([vd.a.terms for _, vd in pairs])
    writer.rows([vd.beta.terms if vd.beta is not None else {} for _, vd in pairs])
    writer.rows([vd.z[col].terms if col in vd.z else {} for _, vd in pairs for col in range(HD)])
    outputs = [{} for _ in range(dims)]
    for j, (i, vd) in enumerate(pairs):
        for col, d, value in vd.out:
            index = j * HD + col
            outputs[d][index] = outputs[d].get(index, 0) + value
    writer.rows(outputs)


def export_state(path, levels):
    with path.open("wb") as file:
        file.write(b"QPS1")
        writer = Writer(file)
        writer.u32(len(levels))
        for nodes in levels:
            state = {}
            if not isinstance(nodes[0], FFN):
                head_offset = 0
                for head in nodes:
                    for local, initial in enumerate(head.init or []):
                        for row, vec in initial.items():
                            start = ((head_offset + local) * HD + row) * HD
                            state.update((start + col, value) for col, value in enumerate(vec) if value)
                    head_offset += len(head.vheads())
            writer.rows([state])


def export_embeddings(writer, circuit, name, base, size):
    # The compiler's family, payload and readout features are affine in payload
    # bits. Quotient-digit features also test payload equality in QS and QD.
    def features(token):
        return {ONE: 1.0, **{d: circuit.feature_value(d, token) for d in circuit.token_fns}}

    if name in ("QS", "QD"):
        writer.u32(1)
        writer.rows([features(base + payload) for payload in range(size)])
    else:
        writer.u32(0)
        zero = features(base)
        deltas = [{d: value - zero[d] for d, value in features(base + (1 << bit)).items()}
                  for bit in range((max(size, 1) - 1).bit_length())]
        writer.rows([zero, *deltas])


def main(output):
    circuit = core.build(vx=True)
    dims = len(circuit.dims)
    levels = [[node for node in circuit.nodes if node.level == level]
              for level in sorted({node.level for node in circuit.nodes})]
    with (output / "vector.qpc").open("wb") as file:
        file.write(b"QPC2")
        writer = Writer(file)
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
        writer.rows([expr.terms for expr in [*circuit.out_fam, *circuit.out_pay]])
        writer.u32(len(core.OUT_NAMES))
        for name in core.OUT_NAMES:
            base, size = spec.FAMILIES.get(name, (spec.CONTROLS.get(name, 0), 0))
            writer.u32(base, size, name in spec.CONTROLS)
            if name != "Q":
                export_embeddings(writer, circuit, name, base, size)
            else:
                writer.u32(0)
                writer.rows([])

    expected = {"programs": []}
    for number, source in enumerate(PROGRAMS, 1):
        machine = ref.Machine.from_program(assemble(source))
        core.snapshot(circuit, machine)
        export_state(output / f"program-{number}.qps", levels)
        trace = machine.run(128)
        if not machine.halted or machine.fault or trace[-1] != spec.tok("HALT"):
            raise RuntimeError(f"Program {number} did not halt cleanly in the reference.")
        expected["programs"].append({"trace": trace[1:], "output": machine.out.decode("ascii")})
        if number == 1:
            canary = machine.ram_byte(CANARY_ADDRESS)
            if canary != 3:
                raise RuntimeError("First program did not leave the RAM canary.")

    contaminated = ref.Machine.from_program(assemble(PROGRAMS[1]))
    contaminated.ram[CANARY_ADDRESS // 4] = canary
    dirty_trace = contaminated.run(128)
    if [p["output"] for p in expected["programs"]] != ["8", "6"] or contaminated.out != b"9":
        raise RuntimeError("Unexpected reference outputs for clean or contaminated RAM.")
    expected["contaminated_second"] = {"trace": dirty_trace[1:], "output": "9"}
    (output / "expected.json").write_text(json.dumps(expected), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())

