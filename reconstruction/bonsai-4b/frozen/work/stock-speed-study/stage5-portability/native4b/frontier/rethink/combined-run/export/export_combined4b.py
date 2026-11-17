"""Compile and pack the frozen native Bonsai 4B placement into a GGUF."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
PORTABILITY = HERE.parents[4]
FROZEN = HERE.parents[1] / "combined-map"
sys.path[:0] = [str(FROZEN), str(PORTABILITY / "full9b/pack")]
import placement as placement
import pack_full9b as pack9
from export_tensors import q4_head, write_tensor

pad = pack9.pad
gguf = pad.gguf
WIDTH, TRUNK, OLD_FFN, FFN, VOCAB, GUARD = 2560, 32, 9216, 36352, 248320, 513
PREFIX = f"blk.{TRUNK}."
EMBED, HEAD, IDS = (PREFIX + "nextn." + name + ".weight" for name in ("embed_tokens", "draft_head", "draft_head_ids"))
MODEL_NAME = "Ternary-Bonsai-2-4B-PQ2_0.gguf"
MODEL_SHA = "d406b1e912c0d2a4c124ad10bf38a76ae662fdc4d24b176451403dc7b7f30409"
MAP_SHA = "456109cf4eabd58f6419a66444b0d87d873bef7f99bc0c0ac41dfb93857bc27d"
PROJECTIONS = {"EH": "nextn.eh_proj", "Q": "attn_q", "K": "attn_k", "V": "attn_v", "O": "attn_output"}
FFN_DTYPES = {"gate": "<f4", "up": "<u2", "down": "<f2"}


def frozen_inputs():
    receipt = FROZEN / "receipt.json"
    raw = receipt.read_bytes()
    if sha256(raw).hexdigest() != MAP_SHA:
        raise ValueError(f"Frozen placement receipt differs from {MAP_SHA}")
    report = json.loads(raw)
    for name, expected in report["source_sha256"].items():
        if pad.file_sha(placement.ROOT / name) != expected:
            raise ValueError(f"Frozen compiler dependency changed: {name}")
    for name, key in (("projections.json", "projection_artifact_sha256"), ("phase-masks.json", "phase_masks_sha256")):
        if pad.file_sha(FROZEN / name) != report[key]:
            raise ValueError(f"Frozen placement artifact changed: {name}")
    pin = json.loads((PORTABILITY / "headers.json").read_bytes())["remote"][MODEL_NAME]
    if (pin["published_lfs_sha256"] != MODEL_SHA
            or pin["header_sha256"] != report["tensor_contract"]["bonsai4b_header_sha256"]):
        raise ValueError("Source header pin differs from the placement receipt")
    return report, pin


def circuit_layout(report):
    geom = report["tensor_contract"]["attention_geometry"]
    head = geom["head_width"]
    query, kv = geom["heads"] * head, geom["kv_groups"] * head

    def tensor(dims, kind):
        desc = {"dims": dims, "type_id": int(kind)}
        return desc | {"bytes": pad.tensor_bytes(desc)}

    circuit = {
        EMBED: tensor([WIDTH, VOCAB], pad.PQ2),
        HEAD: tensor([WIDTH, GUARD + 1], pad.Q4),
        IDS: tensor([GUARD + 1], 26),
        PREFIX + "nextn.eh_proj.weight": tensor([2 * WIDTH, WIDTH], 0),
        PREFIX + "attn_q.weight": tensor([WIDTH, 2 * query], 0),
        PREFIX + "attn_k.weight": tensor([WIDTH, kv], 0),
        PREFIX + "attn_v.weight": tensor([WIDTH, kv], 0),
        PREFIX + "attn_output.weight": tensor([query, WIDTH], 0),
        PREFIX + "ffn_gate.weight": tensor([WIDTH, FFN], 0),
        PREFIX + "ffn_up.weight": tensor([WIDTH, FFN], 30),
        PREFIX + "ffn_down.weight": tensor([FFN, WIDTH], 1),
    }
    for name in report["tensor_contract"]["norms_f32"]:
        circuit[PREFIX + name] = tensor([head if name in ("attn_q_norm.weight", "attn_k_norm.weight") else WIDTH], 0)
    if len(circuit) != 18 or report["grouped_ffn"]["rows"] != FFN:
        raise ValueError("Frozen circuit tensor inventory no longer matches the exporter")
    return circuit


def compile_rows(report):
    print("Recompiling the frozen placement on CPU", flush=True)
    _, details, _, _, _, _, _ = placement.production_source()
    signed, anchors, _ = placement.side_roles(details[0])
    place = placement.allocate(details, signed, anchors, WIDTH)
    if json.loads(json.dumps(place)) != report["slots"]:
        raise ValueError("Recompiled native placement differs from the frozen lane map")
    projected = placement.project(details, place, signed | anchors, placement.signed_attention(details))
    if {kind: placement.nibble.digest_matrix(matrix) for kind, matrix in projected.items()} != report["projection_sha256"]:
        raise ValueError("Recompiled projections differ from the frozen coefficients")
    if placement.tensor_contract(place, projected) != report["tensor_contract"]:
        raise ValueError("Recompiled norms and geometry differ from the frozen contract")
    if placement.pack_action_tables(details, place) != report["packed_actions"]:
        raise ValueError("Recompiled action tables differ from the frozen action tables")

    rows = []
    summarize = placement.compact.row_summary

    def capture_rows(source):
        # Capture the rows at the same boundary and order as the frozen checksum.
        for gate, up in sorted(source):
            if down := placement.compact.sparse(source[gate, up]):
                rows.append((gate, up, down))
        return summarize(source)

    with patch.object(placement.compact, "row_summary", capture_rows):
        summary = placement.spill.regroup_rows(details, place)
    if summary != report["grouped_ffn"] or len(rows) != FFN:
        raise ValueError("Grouped coefficient stream differs from the frozen FFN receipt")
    print(f"Recompiled {len(rows):,} exact FFN rows, SHA-256 {summary['coefficient_stream_sha256']}", flush=True)
    return rows, details[3], details[5], place, projected


def action_tables(report, codes, selector, place):
    row_bytes = len(pad.PQ2_ZERO) * (WIDTH // 128)
    embedding = bytearray(pad.PQ2_ZERO * ((GUARD + 1) * (WIDTH // 128)))
    scores = {}
    for token, code in codes.items():
        tile_start = token * row_bytes + place["input_tile_start"] // 128 * len(pad.PQ2_ZERO)
        slots = {place["input"][lane] % 128 for lane, value in selector.embedding(code) if value == 0.5}
        if len(slots) != 4:
            raise ValueError(f"Action {token} does not use four selector coordinates")
        struct.pack_into("<e", embedding, tile_start, 0.5)
        for slot in slots:
            embedding[tile_start + 2 + slot // 4] ^= 3 << (2 * (slot % 4))
        for bit in range(placement.SCORE_SIGNS):
            scores[token, place["score_start"] + bit] = placement.s.sign(code, bit)
    head, tiles = q4_head(scores, (GUARD + 1, WIDTH))
    head = head.tobytes()
    expected = report["packed_actions"]
    if (sha256(embedding).hexdigest() != expected["action_pq2_rows_sha256"]
            or sha256(head).hexdigest() != expected["action_q4_rows_sha256"]
            or tiles != expected["active_rows"] or GUARD in codes):
        raise ValueError("Serialized action bytes differ from the frozen action tables")
    return embedding, head


def verify_ffn(raw_paths, rows, report):
    """Read stored terms back in the frozen coefficient-stream order."""
    matrices = [np.memmap(raw_paths[PREFIX + f"ffn_{kind}.weight"], mode="r", dtype=dtype,
                          shape=(WIDTH, FFN) if kind == "down" else (FFN, WIDTH))
                for kind, dtype in FFN_DTYPES.items()]
    digest = sha256()
    for row, terms in enumerate(rows):
        digest.update(struct.pack("<3H", *map(len, terms)))
        for matrix, entries in zip((matrices[0][row], matrices[1][row], matrices[2][:, row]), terms, strict=True):
            lanes = np.asarray([lane for lane, _ in entries], dtype=np.intp)
            values = matrix[lanes].tobytes()
            size = matrix.itemsize
            for index, lane in enumerate(lanes):
                digest.update(struct.pack("<H", lane) + values[index * size:(index + 1) * size])
    result = digest.hexdigest()
    if result != report["grouped_ffn"]["coefficient_stream_sha256"]:
        raise ValueError("Read-back FFN coefficient stream differs from frozen coefficients")
    for kind, matrix in zip(FFN_DTYPES, matrices, strict=True):
        count = sum(np.count_nonzero(matrix[i:i + 128]) for i in range(0, len(matrix), 128))
        if count != report["grouped_ffn"][kind + "_nonzeros"]:
            raise ValueError(f"Unexpected nonzero coefficient outside the frozen {kind} support")
    return result


def export_raw(raw_dir, report, circuit):
    index_path = raw_dir / "index.json"
    paths = {name: raw_dir / (name + ".bin") for name in circuit}
    if index_path.exists() or any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Expected a fresh raw output directory: {raw_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(raw_dir).free < sum(desc["bytes"] for desc in circuit.values()):
        raise OSError("Insufficient room for circuit payloads")
    rows, codes, selector, place, projected = compile_rows(report)
    embedding, head = action_tables(report, codes, selector, place)
    special = {EMBED: embedding, HEAD: head, IDS: np.arange(GUARD + 1, dtype="<i4").tobytes()}
    sparse = {target + ".weight": projected[kind] for kind, target in PROJECTIONS.items()}
    for name, desc in report["tensor_contract"]["norms_f32"].items():
        coords = desc["coordinates"]
        coords = range(WIDTH) if coords == "all 2560 coordinates" else coords
        sparse[name] = {(lane,): desc["coefficient"] for lane in coords}
    output = {}
    for name, desc in circuit.items():
        tail = name.removeprefix(PREFIX)
        with paths[name].open("xb") as stream:
            if name in special:
                stream.write(special[name])
                if name == EMBED:
                    for data in pad.pattern_chunks(pad.PQ2_ZERO, desc["bytes"] - len(embedding)):
                        stream.write(data)
            elif tail.startswith("ffn_"):
                kind = tail.removeprefix("ffn_").removesuffix(".weight")
                selector_index = tuple(FFN_DTYPES).index(kind)
                entries = {(lane, row) if kind == "down" else (row, lane): value
                           for row, coeffs in enumerate(rows) for lane, value in coeffs[selector_index]}
                write_tensor(stream, tail, desc, {tail: entries}, None)
                del entries
            else:
                write_tensor(stream, tail, desc, sparse, None)
        if paths[name].stat().st_size != desc["bytes"]:
            raise ValueError(f"Wrong byte count for {name}")
        output[name] = desc | {"sha256": pad.file_sha(paths[name])}
    stream_sha = verify_ffn(paths, rows, report)
    index = {"ffn_rows_used": FFN, "coefficient_stream_sha256": stream_sha, "placement_sha256": MAP_SHA,
             "tensors": output, "exporter_sha256": pad.file_sha(Path(__file__))}
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Exported {len(output)} tensors to {index_path}", flush=True)
    return index_path


def extended_metadata(fields, tensors, alignment, staged):
    widths = struct.unpack("<3i", pad.sign_array(pad.SIGN_WIDTHS, fields[pad.SIGN_WIDTHS], pad.VALUE.INT32))
    raw = bytes(pad.sign_array(pad.SIGN_VALUES, fields[pad.SIGN_VALUES], pad.VALUE.INT32))
    if widths != (WIDTH, 4096, OLD_FFN) or len(raw) != 4 * sum(widths):
        raise ValueError("Unexpected pinned 4B Hadamard sign geometry")
    raw_signs = raw + struct.pack("<i", 1) * (FFN - OLD_FFN)
    signs = struct.unpack(f"<{len(raw_signs) // 4}i", raw_signs)
    widths = [*widths[:2], FFN]

    # As in the 9B writer, splice changed KVs into the original raw metadata.
    writer = gguf.GGUFWriter(staged, "qwen35")
    writer.data_alignment = alignment
    writer.remove_key("general.architecture")
    writer.add_block_count(TRUNK + 1)
    writer.add_feed_forward_length(FFN)
    writer.add_uint32(pad.NEXTN, 1)
    writer.add_key_value(pad.SIGN_WIDTHS, widths, pad.VALUE.ARRAY, pad.VALUE.INT32)
    writer.add_key_value(pad.SIGN_VALUES, signs, pad.VALUE.ARRAY, pad.VALUE.INT32)
    for name, desc in tensors.items():
        writer.add_tensor_info(name, tuple(reversed(desc["dims"])), np.dtype("int8"), pad.tensor_bytes(desc),
                               raw_dtype=gguf.GGMLQuantizationType(desc["type_id"]))
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
    finally:
        writer.close()
    with staged.open("r+b") as file:
        count, edits = pad.metadata_records(file)
        descriptors = file.read()
        metadata = fields | edits
        file.seek(0)
        file.write(struct.pack("<4sIQQ", b"GGUF", 3, count, len(metadata)))
        for record in metadata.values():
            file.write(record)
        file.write(descriptors)
        writer.write_padding(file, file.tell())
        start = file.tell()
    return start, metadata, edits, widths, raw_signs


def verify_header(file, tensors, start, metadata, widths, signs):
    header = pad.Header(file).parse()
    m = header["metadata"]
    if (header["tensors"] != tensors or list(header["tensors"]) != list(tensors)
            or header["tensor_data_offset"] != start or header["metadata_count"] != len(metadata)
            or m["qwen35.block_count"] != TRUNK + 1 or m["qwen35.feed_forward_length"] != FFN
            or m[pad.NEXTN] != 1 or m[pad.SIGN_WIDTHS] != widths):
        raise ValueError("Unexpected output GGUF descriptors or geometry")
    offset = 0
    for width in widths:
        span = signs[offset:offset + width * 4]
        if sha256(span).hexdigest() != m["prism.hadamard.sign_values_summary"][width]["sha256_i32_le"]:
            raise ValueError(f"Changed Hadamard sign bytes at width {width}")
        offset += width * 4
    file.seek(24)
    for key, expected in metadata.items():
        if pad.read_exact(file, len(expected)) != expected:
            raise ValueError(f"Unexpected serialized metadata for {key}")
    return header


def pack_model(source_path, index_path, target, report, pin, expected):
    manifest = target.with_suffix(".manifest.json")
    if target.suffix.lower() != ".gguf" or target.exists() or manifest.exists():
        raise ValueError("Expected a fresh GGUF output and manifest path")
    index_raw = index_path.read_bytes()
    with patch.multiple(pad, WIDTH=WIDTH, OLD_FFN=OLD_FFN, NEW_FFN=FFN, TRUNK=TRUNK, DRAFT_IDS=IDS):
        rows, circuit, index_sha = pack9.validate_index(index_path, {"circuit_tensors": expected})
        payloads = pack9.payload_paths(index_path, circuit)
        originals = pin["tensors"]
        host = pin["metadata"]
        alignment = host.get("general.alignment", 32)
        tensors, downs, changes = pad.padded_tensors(originals, circuit, alignment)
    if (rows != FFN or json.loads(index_raw)["placement_sha256"] != MAP_SHA
            or json.loads(index_raw)["coefficient_stream_sha256"] != report["grouped_ffn"]["coefficient_stream_sha256"]):
        raise ValueError("Raw tensor index differs from the frozen placement")
    trunk = {name: tensors[name] for name in originals}
    folded = set(host["prism.hadamard.weight_names"]) | set(host["prism.hadamard.inverse_weight_names"])
    block = host["prism.hadamard.block_size"]
    if (not downs <= folded or not folded <= originals.keys() or block != 512 or FFN % block
            or host["qwen35.block_count"] != TRUNK or host.get(pad.NEXTN, 0)
            or originals["token_embd.weight"]["dims"] != [WIDTH, VOCAB]):
        raise ValueError("Pinned 4B host geometry or Hadamard binding changed")
    estimated = (pin["file_bytes"] + report["grouped_ffn"]["storage"]["language_padding_bytes"]
                 + sum(desc["bytes"] for desc in circuit.values()))
    if shutil.disk_usage(target.parent).free < estimated + 1024**2:
        raise OSError(f"Need about {estimated:,} free bytes for the packed GGUF")
    with source_path.open("rb") as source, tempfile.TemporaryDirectory(dir=target.parent, prefix="pack4b-") as temp:
        initial = os.fstat(source.fileno())
        if initial.st_size != pin["file_bytes"] or file_digest(source, "sha256").hexdigest() != MODEL_SHA:
            raise ValueError(f"Source bytes do not match pinned SHA-256 {MODEL_SHA}")
        source.seek(0)
        source_start = pin["tensor_data_offset"]
        if sha256(pad.read_exact(source, source_start)).hexdigest() != pin["header_sha256"]:
            raise ValueError("Source GGUF header differs from the pin")
        count, fields = pad.metadata_records(source)
        if count != len(originals) or len(fields) != pin["metadata_count"]:
            raise ValueError("Input tensor or metadata count differs from the pin")

        staged = Path(temp) / target.name
        start, metadata, edits, widths, signs = extended_metadata(fields, tensors, alignment, staged)
        print("Writing padded language tensors and appending the circuit", flush=True)
        pad.write_payloads(source, staged, originals, trunk, downs, source_start, alignment)
        pack9.write_circuit(staged, circuit, payloads, alignment)
        size = staged.stat().st_size
        last = list(tensors.values())[-1]
        if size != start + last["offset"] + gguf.GGUFWriter.ggml_pad(pad.tensor_bytes(last), alignment):
            raise ValueError("Packed file length differs from the laid-out tensor extents")
        print("Verifying packed metadata and tensor bytes against the source and index", flush=True)
        with staged.open("rb") as output:
            header = verify_header(output, tensors, start, metadata, widths, signs)
            output.seek(0)
            digest = sha256(pad.read_exact(output, start))
            mtp_start = start + tensors[EMBED]["offset"]
            kept, _ = pad.verify_payloads(source, pack9.TrunkReader(output, mtp_start), originals, trunk, downs,
                                           source_start, alignment, digest)
            if output.tell() != mtp_start or kept.keys() != originals.keys():
                raise ValueError("Language trunk verification did not consume the complete source inventory")
            pack9.verify_circuit(output, circuit, alignment, digest)
        end = os.fstat(source.fileno())
        if (initial.st_size, initial.st_mtime_ns) != (end.st_size, end.st_mtime_ns):
            raise ValueError("Source bytes changed during packing")
        if ({name for name, desc in kept.items() if desc["padding_bytes"]} != changes.keys()
                or len(changes) != 3 * TRUNK
                or sum(desc["padding_bytes"] for desc in kept.values()) != report["grouped_ffn"]["storage"]["language_padding_bytes"]):
            raise ValueError("Packed language FFN padding differs from the frozen byte ledger")
        counters = Counter({"language_tensors_byte_equal": len(kept), "language_tensors_with_padding": len(changes),
                            "wholly_unchanged_language_tensors": len(kept) - len(changes), "mtp_tensors": len(circuit)})
        result = {"output": str(target), "sha256": digest.hexdigest(), "bytes": size,
                  "header_sha256": header["header_sha256"], "header_data_offset": start,
                  "source": str(source_path), "source_sha256": MODEL_SHA, "source_bytes": initial.st_size,
                  "source_revision": pin["revision"], "runtime_revision": pad.RUNTIME_REVISION,
                  "placement_sha256": MAP_SHA, "circuit_index_sha256": index_sha,
                  "coefficient_stream_sha256": report["grouped_ffn"]["coefficient_stream_sha256"],
                  "width": WIDTH, "language_layers": TRUNK, "mtp_layers": 1,
                  "source_ffn": OLD_FFN, "padded_ffn": FFN, "ffn_rows_used": rows,
                  "source_nonpadding_bytes": sum(desc["bytes"] for desc in kept.values()),
                  "pq2_padding_block_hex": pad.PQ2_ZERO.hex(), "metadata_fields_byte_equal": len(fields.keys() - edits.keys()),
                  "edited_metadata_fields": list(edits), "original_hadamard_sign_vectors_byte_equal": True,
                  "hadamard_sign_widths": widths, "hadamard_signs": header["metadata"]["prism.hadamard.sign_values_summary"],
                  "new_positive_hadamard_signs": FFN - OLD_FFN, "draft_token_id_range": [0, GUARD],
                  "packed_actions": report["packed_actions"], "source_tensors": kept, "circuit_tensors": circuit,
                  "script_sha256": {Path(__file__).name: pad.file_sha(Path(__file__)),
                                    "export_tensors.py": pad.file_sha(Path(sys.modules["export_tensors"].__file__)),
                                    "export_pad9b.py": pad.file_sha(Path(pad.__file__)),
                                    "pack_full9b.py": pad.file_sha(Path(pack9.__file__))},
                  "reproduction_argv": [sys.executable, "-B", *sys.argv], **counters}
        staged_manifest = Path(temp) / manifest.name
        staged_manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
        os.link(staged, target)
        try:
            os.link(staged_manifest, manifest)
        except OSError:
            target.unlink()
            raise
    print(json.dumps({key: result[key] for key in ("output", "sha256", "bytes", "header_sha256", *counters)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    raw = modes.add_parser("export-raw")
    raw.add_argument("raw_dir", type=Path)
    pack = modes.add_parser("pack")
    pack.add_argument("source", type=Path)
    pack.add_argument("index", type=Path)
    pack.add_argument("output", type=Path)
    options = parser.parse_args()
    report, pin = frozen_inputs()
    circuit = circuit_layout(report)
    if options.mode == "export-raw":
        export_raw(options.raw_dir.resolve(), report, circuit)
    else:
        pack_model(options.source.resolve(strict=True), options.index.resolve(strict=True), options.output.resolve(),
                   report, pin, circuit)


if __name__ == "__main__":
    main()
