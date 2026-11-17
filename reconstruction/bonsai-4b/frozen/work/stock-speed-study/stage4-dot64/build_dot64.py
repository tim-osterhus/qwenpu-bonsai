"""Build the staged stock VX controller with a neural 64-lane VMAT dot."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

import gguf
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
WORK = STUDY.parent
sys.path[:0] = [str(STUDY / "baseline/exporter"), str(WORK / "stock-mtp-vx/plane-dot64")]
import export_stock_mtp_vx as vx
import export_stock_mtp_scalar as s
import export_stock_mtp_scalar_m as m
from plane_dot64 import action_ff_rows


WIDTH, HEAD, HEADS, GROUPS = 5120, 256, 24, 4
SIDE_BASE, SIDE_ROWS = 0x70000000, 16
PAYLOAD = 1663
SIDE_PAYLOAD = 4710
PAYLOAD_INGRESS = 2048
REF, REF2, PLANE = 5000, 5001, 5002
VA_OUT = 5003
VA = tuple(f"DOT64_VA_{bit}" for bit in range(8))
CN, CP, FN, FP, SUM = "DOT64_CN", "DOT64_CP", "DOT64_FN", "DOT64_FP", "DOT64_SUM"
DISPATCH = "DOT64_NEXT_PLANE"
CORE = (*VA, CN, CP, FN, FP, SUM)


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def circuit(transform=None):
    c = vx.program()
    b = c.vx
    fields = b.fields
    state = b.state
    zero = s.WORK + s.ZERO
    acc, waddr = fields["accumulator"], fields["wrom_word"]
    ports = dict(ONE=fields["tile_products"][0], ZERO=fields["tile_products"][1], UC=fields["tile_products"][2],
                 ACC=fields["tile_products"][5:23], Q=fields["tile_products"][23:29], G=fields["tile_generate"][:18])
    gates = fields["tile_activations"]
    plane_bits = fields["tile_weights"][:8]
    # Reuse IDs from the replaced VMAT region. Keep every other action ID and the trimmed guard stable.
    obsolete = [phase for phase in c.phases if phase.startswith(("VX_MUL_BIT", "VX_REDUCE_"))]
    replacement = dict(zip(obsolete, (*CORE, DISPATCH, *[f"DOT64_UNUSED_{i}" for i in range(len(obsolete)-14)]), strict=True))
    c.phases = dict.fromkeys(replacement.get(phase, phase) for phase in c.phases)
    for phase in obsolete:
        c.routes.pop(phase, None)
        c.effects.pop(phase, None)
    for phase in replacement.values():
        if phase.startswith("DOT64_UNUSED_"):
            c.route(phase, "SET_ILLEGAL")

    for phase in ("VX_TILE_QUERY", "VX_TILE_READ", "VX_TILE_ACC", "AFTER_VX_ACC", "VX_CHUNK_NEXT"):
        c.routes[phase] = {}
        c.effects[phase] = {}
    used = set(gates) | set(plane_bits) | {ports["ONE"], ports["ZERO"], ports["UC"]}
    for key in ("ACC", "Q", "G"):
        used.update(ports[key])
    for effects in c.effects.values():
        used.update(effects)
        for expression in effects.values():
            used.update(lane - s.WORK for _, terms in expression for lane, _ in terms)
    for routes in c.routes.values():
        used.update(abs(flag) for flag in routes.values() if flag is not None)
    payload_ingress = tuple(range(PAYLOAD_INGRESS, PAYLOAD_INGRESS + 257))
    if set(payload_ingress) & set(c.persistent):
        raise ValueError("Overlapping seed-only hidden ingress")
    # These dot-only operands are naturally zero outside the immediately preceding stage.
    # Give them dedicated ingress/egress and remove the compiler's ordinary persistent refresh.
    ephemeral = tuple(lane for lane in c.persistent if lane not in used and lane not in range(PAYLOAD, PAYLOAD + 257))[:38]
    if len(ephemeral) != 38:
        raise ValueError(f"Not enough exclusively reusable dot lanes: {len(ephemeral)}")
    ports.update(A=ephemeral[:18], X=ephemeral[18:36], UF=ephemeral[36], US=ephemeral[37])
    c.persistent = tuple(lane for lane in c.persistent if lane not in ephemeral)
    b.reads.remove("VX_TILE_READ")
    c.route("VX_TILE_READ", "SET_ILLEGAL")
    # The raw VA base is always 64-byte aligned. The four subheads add word offsets in Q weights.
    b.addr("VX_TILE_QUERY", "va", (*((zero,) * 4), *state(fields["chunk_current"])))
    b.fix("VX_TILE_QUERY", ports["ACC"], 0)
    c.route("VX_TILE_QUERY", VA[7])
    for bit, phase in enumerate(VA):
        b.fix(phase, plane_bits, 1 << bit)
        c.route(phase, CN if bit == 7 else CP)
    c.route(CN, FN)
    c.route(CP, FP)
    c.route(FN, SUM)
    c.route(FP, SUM)
    c.route(SUM, DISPATCH)
    b.addr(DISPATCH, "va", (*((zero,) * 4), *state(fields["chunk_current"])))
    for bit in range(1, 8):
        c.route(DISPATCH, (VA[bit - 1], plane_bits[bit]))
    c.route(DISPATCH, ("VX_TILE_ACC", plane_bits[0]))
    c.add("VX_TILE_ACC", state(acc), b.resize(state(ports["ACC"]), signed=True), "VX_ACC")
    b.move("AFTER_VX_ACC", acc, s.bits(s.WORK + s.SUM))
    # Generic add handles carries in waddr+8. Keep the original scale query and multiplier.
    c.add("AFTER_VX_ACC", b.resize(state(waddr)), b.const(8), "VX_WSTEP")
    # WSTEP is also used by row-to-row transitions. A dedicated retained flag selects the scale path.
    scale_path = fields["tile"][0]
    c.fix("AFTER_VX_ACC", scale_path, 1, 1)
    for phase in ("VX_WBASE_STEP9", "VX_WBASE_STEP27"):
        c.fix(phase, scale_path, 0, 1)
    c.routes["AFTER_VX_WSTEP"] = {}
    c.effects["AFTER_VX_WSTEP"] = {}
    c.branch("AFTER_VX_WSTEP", scale_path, "DOT64_UNUSED_0", "DOT64_UNUSED_1")
    for phase, destination, target in (("DOT64_UNUSED_0", waddr, "VX_SCALE_QUERY"),
                                       ("DOT64_UNUSED_1", fields["wrom_base"], "VX_ROW_INIT")):
        c.routes[phase] = {}
        b.move(phase, destination, s.bits(s.WORK + s.SUM))
        c.route(phase, target)
    b.inc("VX_CHUNK_NEXT", waddr)
    b.inc("VX_CHUNK_NEXT", fields["chunk_current"])
    c.route("VX_CHUNK_NEXT", "VX_TILE_QUERY")

    # The dot FFN owns its stage outputs. Exclude the corresponding default persistence rows.
    owned = {
        "va": set(gates), "coarse": set(ports["ACC"] + ports["Q"] + ports["A"]) | {ports["ONE"], ports["ZERO"], ports["UF"]} | set(gates),
        "fine": set(ports["X"] + ports["G"]) | {ports["ONE"], ports["ZERO"], ports["US"]},
        "sum": set(ports["ACC"]) | {ports["ONE"], ports["ZERO"]},
    }
    for phase in CORE:
        key = "va" if phase in VA else "coarse" if phase in (CN, CP) else "fine" if phase in (FN, FP) else "sum"
        for lane in owned[key]:
            c.effects[phase][lane] = ()
    for phase in VA:
        for lane, dest in enumerate(gates):
            c.put(phase, dest, s.buffer(VA_OUT + lane))
    if transform is not None:
        transform(c)
    # Keep canonical normalization on every persistent background bit.
    compiled = vx.compile_program(c)
    ids, scores, gate, up, down, count = compiled
    add = action_ff_rows(ports, s.WORK, PLANE, (REF, REF2),
                         {"COARSE": (s.ACTION + ids[CN], s.ACTION + ids[CP]),
                          "FINE": (s.ACTION + ids[FN], s.ACTION + ids[FP]), "SUM": (s.ACTION + ids[SUM],)})
    gate += [((row + count, col), coef) for (row, col), coef in add["gate"]]
    up += [((row + count, col), coef) for (row, col), coef in add["up"]]
    down += [((dest, row + count), coef) for (dest, row), coef in add["down"]]
    count += add["rows"]
    for lane in gates:
        for include, bias, polarity in ((1, 4, 1), (1, -4, -1), (0, 4, -1), (0, -4, 1)):
            if include:
                gate.append(((count, s.WORK + lane), 16777216))
            for phase in (CN, CP):
                gate.append(((count, s.ACTION + ids[phase]), bias))
                up.append(((count, s.ACTION + ids[phase]), 16))
            down.append(((lane, count), polarity))
            count += 1
    compiled = ids, scores, gate, up, down, count
    if count > 21248 or max(col for _, col in scores) >= REF or SIDE_PAYLOAD <= s.ACTION + len(ids) or SIDE_PAYLOAD + 256 >= REF:
        raise ValueError("Dot circuit exceeds the frozen FFN/dispatch budget")
    c.dot_ephemeral = ephemeral
    c.dot_payload_ingress = payload_ingress
    return c, compiled, ports, gates


def attention(c, ids, ports, gates, q, k, v, o):
    # Retain the scalar/ordinary read contract in group 0 and duplicate tags to the other groups.
    for phase in c.vx.reads:
        q[HEAD:HEAD + 57, s.ACTION + ids[phase]] = 128
    for group in range(1, GROUPS):
        k[group * HEAD:(group + 1) * HEAD] = k[:HEAD]
    for group in range(GROUPS):
        for lane in range(64):
            v[group * HEAD + 64 + lane, SIDE_PAYLOAD + group * 64 + lane] = 1
        for sub in range(6):
            head = 6 * group + sub
            start = 2 * HEAD * head
            # Sixteen VA heads read one word each. Only the chosen byte bit gates onto each output lane.
            if sub < 4:
                word = group * 4 + sub
                if head:
                    q[start + HEAD:start + HEAD + 32, s.ACTION:s.ACTION + len(ids)] = -65536
                for bit, phase in enumerate(VA):
                    action = s.ACTION + ids[phase]
                    for tag in range(33):
                        source = s.WORK + s.READ_ADDR + tag
                        if head:
                            q[start + 64 + 4 * tag:start + 68 + 4 * tag, source] = -1 if 2 <= tag < 6 and word >> (tag - 2) & 1 else 1
                    for byte in range(4):
                        slot = 8 * byte + bit
                        q[start + HEAD + slot, action] = 65536
                        o[VA_OUT + word * 4 + byte, head * HEAD + slot] = 1
                if head:
                    q[start + 196, s.WORK + s.READ_MARK] = 11.125
                if head == 0:
                    for bit, phase in enumerate(VA):
                        for slot in range(32):
                            q[HEAD + slot, s.ACTION + ids[phase]] = 65536 if slot % 8 == bit else -65536
                # Each auxiliary head reads the same VA word values from its KV group.
                continue

            for tag in range(33):
                if tag < 2:
                    source = s.WORK + s.ZERO
                elif tag < 18:
                    source = s.WORK + c.vx.fields["wrom_word"][tag - 2]
                else:
                    source = s.WORK + (s.ONE if (SIDE_BASE >> tag) & 1 else s.ZERO)
                q[start + 64 + 4 * tag:start + 68 + 4 * tag, source] = 1
            allowed = (CN, FN) if sub == 5 else (CP, FP)
            for slot in range(64, 128):
                dest = start + HEAD + slot
                q[dest, s.WORK + gates[group * 16 + (slot - 64) % 16]] = 16777216
                for phase, token in ids.items():
                    if phase not in allowed:
                        q[dest, s.ACTION + token] = -1e8
                o[PLANE, head * HEAD + slot] = (1 if sub == 4 else -1) * (1, 2, 4, -8)[(slot - 64) // 16] / 512
    # Group 0's normal side head supplies two equal references and the coarse quotient correction.
    rh = 4
    start = 2 * HEAD * rh
    for slot in range(128, 136):
        v[slot, SIDE_PAYLOAD + 256] = 1
        for phase, token in ids.items():
            q[start + HEAD + slot, s.ACTION + token] = 65536 if phase in (CN, CP, FN, FP) else -65536
    o[REF, HEAD * rh + 128] = o[REF2, HEAD * rh + 128] = 1
    for bit, port in enumerate(ports["Q"]):
        slot = 129 + bit
        q[start + HEAD + slot, :] = 0
        q[start + HEAD + slot, s.WORK + port] = 16777216
        for phase, token in ids.items():
            if phase not in (FN, FP):
                q[start + HEAD + slot, s.ACTION + token] = -1e8
        o[PLANE, HEAD * rh + slot] = -(32 << bit) / 512
    for phase in (CN, CP):
        q[start + HEAD + 135, s.ACTION + ids[phase]] = -65536
    o[PLANE, HEAD * rh + 135] = 1
    for group in range(1, GROUPS):
        v[group * HEAD:group * HEAD + 32] = v[:32]


def export(path, c, compiled, ports, gates):
    if path.exists():
        raise FileExistsError(path)
    if shutil.disk_usage(path.parent).free < 2 * 1024**3:
        raise OSError("Need 2 GiB free for the staged model and evidence")
    ids, scores, *_ = compiled
    ff = 21248
    w = gguf.GGUFWriter(path, "qwen35")
    for name, value in {"context_length": 2048, "embedding_length": WIDTH, "vocab_size": 1024,
                        "feed_forward_length": ff, "attention.head_count": HEADS, "attention.head_count_kv": GROUPS,
                        "attention.key_length": HEAD, "attention.value_length": HEAD, "rope.dimension_count": 64,
                        "ssm.conv_kernel": 4, "ssm.state_size": HEAD, "ssm.group_count": 1, "ssm.time_step_rank": 1,
                        "ssm.inner_size": HEAD, "full_attention_interval": 4}.items():
        w.add_uint32("qwen35." + name, value)
    w.add_array("qwen35.rope.dimension_sections", [16, 8, 8, 0])
    w.add_float32("qwen35.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_float32("qwen35.rope.freq_base", 10000.)
    w.add_string("tokenizer.ggml.model", "none")
    w.add_name("Staged stock VX 64-lane VMAT")
    w.add_block_count(2)
    w.add_uint32("qwen35.nextn_predict_layers", 1)
    w.add_tensor("token_embd.weight", np.zeros((1024, WIDTH), np.float32))
    w.add_tensor("output_norm.weight", np.ones(WIDTH, np.float32))
    qkvo = {}

    class Writer:
        def add_tensor(self, name, data, **kwargs):
            tail = name.removeprefix("blk.1.")
            if tail in ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"):
                qkvo[tail] = data
                if len(qkvo) == 4:
                    attention(c, ids, ports, gates, *qkvo.values())
                    for suffix, matrix in qkvo.items():
                        w.add_tensor("blk.1." + suffix, matrix)
                    qkvo.clear()
                return
            if tail == "nextn.hnorm.weight":
                data[list(c.dot_payload_ingress)] = np.float32(1 / np.sqrt(WIDTH))
                data[list(c.dot_ephemeral)] = np.float32(1 / np.sqrt(WIDTH))
            elif tail == "nextn.eh_proj.weight":
                if np.any(data[SIDE_PAYLOAD:SIDE_PAYLOAD + 257]):
                    raise ValueError("Existing projection overlaps dot seed work lanes")
                for index, lane in enumerate(c.dot_payload_ingress):
                    data[SIDE_PAYLOAD + index, WIDTH + lane] = 1
                for lane in c.dot_ephemeral:
                    data[s.WORK + lane, WIDTH + lane] = 1
            elif tail == "nextn.shared_head_norm.weight":
                data[list(c.dot_ephemeral)] = 1
            elif tail == "ffn_up.weight":
                raw = data.view(np.uint32)
                if np.any(raw & 65535):
                    raise ValueError("Inexact BF16 FFN up coefficient")
                data = (raw >> 16).astype("<u2").view(np.uint8)
                kwargs["raw_dtype"] = gguf.GGMLQuantizationType.BF16
            w.add_tensor(name, data, **kwargs)

    manifest = s.add_block(Writer(), 1, WIDTH, ff, HEADS, GROUPS, 1024, False, compiled, c, query_f32=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    manifest.update(bytes=path.stat().st_size, sha256=sha(path), read_tokens=sorted(set(manifest["read_tokens"]) | {ids[p] for p in c.vx.reads}),
                    dot_actions={phase: ids[phase] for phase in CORE}, side_wrom_rows=SIDE_ROWS, side_base=SIDE_BASE,
                    side_payload_ingress=list(c.dot_payload_ingress),
                    dot_accumulator_lanes=list(ports["ACC"]), dot_dispatch_action=ids[DISPATCH],
                    vx_feedback_lanes=c.vx.fields, ffn_types={"gate":"F32", "up":"BF16", "down":"F32"})
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    def word(lanes):
        return f"{len(lanes)} " + " ".join(map(str, lanes))
    observer = ["QPVXOBSERVER2", *(word(c.vx.fields[name]) for name in ("vmx", "vsum", "varg", "vbest")),
                f"{c.vx.fields['vbest_valid'][0]} {ids['VX_TILE_READ']}", word(c.vx.fields["wrom_word"]),
                word(s.bits(985)), word(s.bits(1017)), word(manifest["read_tokens"])]
    path.with_suffix(".observer.txt").write_text("\n".join(observer) + "\n", encoding="ascii")
    print(json.dumps({key: manifest[key] for key in ("bytes", "sha256", "mtp_ffn_rows_used", "ffn_types")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    c, compiled, ports, gates = circuit()
    if args.output:
        export(args.output, c, compiled, ports, gates)
    else:
        print(json.dumps(dict(rows=compiled[-1], allocated_rows=21248, phases=len(compiled[0]),
                              old_dot_id=compiled[0]["VX_TILE_READ"], new_dot_actions={phase: compiled[0][phase] for phase in CORE},
                              dispatch_last=max(col for _, col in compiled[1]),
                              payload_ingress=[c.dot_payload_ingress[0], c.dot_payload_ingress[-1]],
                              maximum_export_bytes=1900000000), indent=2))


if __name__ == "__main__":
    main()
