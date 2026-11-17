"""Emit the audited raw little-endian tensors and observer for the 9B packer."""

from hashlib import file_digest, sha256
import json
from pathlib import Path
import shutil

import numpy as np

from compile_full9b import HERE, PORTABILITY, STUDY, WIDTH, FF, QUERY, GROUPS, HEAD
from compile_full9b import ORDINARY, SECOND, build, dot, remap, s, vmatr

PREFIX = "blk.32."
ACTIONS, GUARD, VOCAB = 513, 513, 248320


def hash_path(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def q4_head(entries, shape):
    groups = {}
    packed = np.zeros((shape[0], shape[1] // 32, 18), dtype=np.uint8)
    packed[..., 2:] = 0x88
    for (row, col), value in entries.items():
        key = (row, col // 32)
        if key not in groups:
            groups[key] = np.zeros(32, np.float32)
        groups[key][col % 32] = value
    for position, values in groups.items():
        scale = np.float16(np.max(np.abs(values)) / 4)
        quants = np.rint(values / np.float32(scale)).astype(np.int16)
        if not scale or np.any((quants < -8) | (quants > 7)) or np.any(quants * np.float32(scale) != values):
            raise ValueError(f"Inexact Q4_0 draft-head tile {position}")
        codes = (quants + 8).astype(np.uint8)
        packed[position][:2] = np.asarray([scale], dtype="<f2").view(np.uint8)
        packed[position][2:] = codes[:16] | codes[16:] << 4
    return packed, len(groups)


def action_embedding(stream, action_start):
    for first in range(0, VOCAB, 4096):
        n = min(4096, VOCAB - first)
        data = np.zeros((n, WIDTH // 128, 34), dtype=np.uint8)
        data[..., 2:] = 0x55
        live = np.arange(min(n, max(0, ACTIONS - first)))
        block, lane = np.divmod(action_start + first + live, 128)
        data[live, block, 1] = 0x3C  # exact F16 scale 1
        data[live, block, 2 + lane // 4] ^= (3 << (2 * (lane % 4))).astype(np.uint8)
        data.tofile(stream)
    return {"nonzero": ACTIONS, "nonzero_rows": ACTIONS, "nonzero_columns": ACTIONS, "exact_pq2_tiles": ACTIONS}


def write_tensor(stream, key, desc, sparse, action_start):
    kind, shape = desc["type_id"], tuple(reversed(desc["dims"]))
    if key == "nextn.embed_tokens.weight":
        return action_embedding(stream, action_start)
    if key == "nextn.draft_head_ids.weight":
        np.arange(GUARD + 1, dtype="<i4").tofile(stream)
        return {"nonzero": GUARD, "identity_ids": GUARD + 1}
    entries = {pos: value for pos, value in sparse[key].items() if value}
    if any(len(pos) != len(shape) or any(x < 0 or x >= n for x, n in zip(pos, shape, strict=True)) for pos in entries):
        raise ValueError(f"Out-of-bounds entries in {key}")
    counters = {"nonzero": len(entries)}
    if len(shape) == 2:
        counters.update(nonzero_rows=len({pos[0] for pos in entries}), nonzero_columns=len({pos[1] for pos in entries}))
    if kind == 2:
        data, counters["exact_q4_tiles"] = q4_head(entries, shape)
    else:
        vals = np.asarray(list(entries.values()), dtype="<f4")
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"Nonfinite coefficients in {key}")
        if kind == 0:
            packed = vals
        elif kind == 1:
            packed = vals.astype("<f2")
            if not np.array_equal(packed.astype("<f4"), vals):
                raise ValueError(f"Inexact FP16 coefficients in {key}")
        elif kind == 30:
            bits = vals.view("<u4")
            if np.any(bits & 65535):
                raise ValueError(f"Inexact BF16 coefficients in {key}")
            packed = (bits >> 16).astype("<u2")
        else:
            raise ValueError(f"Unsupported encoding for {key}: type {kind}")
        data = np.zeros(shape, dtype=packed.dtype)
        data[tuple(np.asarray(list(entries)).T)] = packed
    data.tofile(stream)
    return counters


def write_observer(c, ids, target):
    fields = c.vx.fields
    reads = sorted({ids[p] for p in set(ORDINARY) | c.vx.reads})

    def word(lanes):
        return f"{len(lanes)} " + " ".join(map(str, lanes))

    rows = ["QPVXOBSERVER2", *(word(fields[name]) for name in ("vmx", "vsum", "varg", "vbest")),
            f"{fields['vbest_valid'][0]} {ids['VX_TILE_READ']}", word(fields["wrom_word"]),
            word(s.bits(985)), word(s.bits(1017)), word(reads)]
    target.write_text("\n".join(rows) + "\n", encoding="ascii")
    return reads


def main():
    plan = json.loads((PORTABILITY / "layout.json").read_text())
    expected = plan["layouts"]["Ternary-Bonsai-2-9B-PQ2_0.gguf"][str(FF)]["circuit_tensors"]
    if len(expected) != 18 or any(not key.startswith(PREFIX) for key in expected):
        raise ValueError("Unexpected 9B tensor contract")
    targets = {name: HERE / f"{name}.bin" for name in expected}
    index = HERE / "index.json"
    if index.exists() or any(path.exists() for path in targets.values()):
        raise FileExistsError("Use a fresh raw tensor output; existing result was not overwritten")
    if shutil.disk_usage(HERE).free < sum(t["bytes"] for t in expected.values()) + 2 * 2**30:
        raise OSError("Need room for raw tensors and evidence")

    c, compiled, ports, gates, report = build()
    sparse, action_start, report = remap(c, compiled, ports, gates, report)
    ids = compiled[0]
    checks = {}
    output = {}
    for name, desc in expected.items():
        path = targets[name]
        with path.open("xb") as stream:
            checks[name] = write_tensor(stream, name.removeprefix(PREFIX), desc, sparse, action_start)
        if path.stat().st_size != desc["bytes"]:
            raise ValueError(f"Encoded byte count differs for {name}")
        output[name] = {**desc, "sha256": hash_path(path)}

    report["tensor_nonzero_support"] = checks
    sources = (Path(__file__), HERE / "compile_full9b.py", Path(vmatr.__file__), Path(dot.__file__),
               STUDY / "stage4-dot64-later/build_vew16.py", Path(s.__file__), Path(dot.vx.__file__))
    report["source_sha256"] = {path.name: hash_path(path) for path in sources}
    observer = HERE / "full9b.observer.txt"
    report["scalar_read_tokens"] = write_observer(c, ids, observer)
    report["observer_sha256"] = hash_path(observer)
    (HERE / "compile-audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest = {"width": WIDTH, "embedding_length": WIDTH, "head_width": HEAD, "query_heads": QUERY, "kv_heads": GROUPS,
                "ffn_rows_used": compiled[-1], "ffn_rows_allocated": FF, "mtp_ffn_rows_used": compiled[-1],
                "mtp_ffn_rows": FF, "tokens": ids, "guard": GUARD, "read_tokens": report["scalar_read_tokens"],
                "dot_actions": {**{phase: ids[phase] for phase in dot.CORE}, "DOT64_VA_SECOND": ids[SECOND]},
                "dot_dispatch_action": ids[dot.DISPATCH], "side_payload_ingress": list(c.dot_payload_ingress),
                "side_base": dot.SIDE_BASE, "vx_feedback_lanes": c.vx.fields,
                "dot_accumulator_lanes": list(ports["ACC"]), "va1_lanes": list(gates[i] for i in range(64) if i % 16 < 8),
                "va2_lanes": list(gates[i] for i in range(64) if i % 16 >= 8),
                "plane_marker_lanes": list(c.vx.fields["tile_weights"][:8])}
    (HERE / "full9b.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    index.write_text(json.dumps({"ffn_rows_used": compiled[-1], "tensors": output}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"index": str(index), "ffn_rows_used": compiled[-1], "tensors": len(output),
                      "bytes": sum(t["bytes"] for t in output.values())}, indent=2))


if __name__ == "__main__":
    main()
