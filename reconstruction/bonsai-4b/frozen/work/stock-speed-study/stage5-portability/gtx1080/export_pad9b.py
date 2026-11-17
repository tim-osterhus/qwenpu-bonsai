"""Stream an exact-extent 20,480-FFN control with a zero-weight MTP block."""

import argparse
import hashlib
import json
from math import prod
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
sys.path.insert(0, str(AUDIT))
sys.path.insert(0, str(HERE.parents[2] / "prism-upstream/build-source/gguf-py"))
import gguf
from read_headers import Header


NAME = "Ternary-Bonsai-2-9B-PQ2_0.gguf"
SOURCE_SHA256 = "7a89ab48f5038c704480f60781f9b8d4035351f5f6beb7d11fad80001fde23f9"
MODEL_REVISION = "82265ccc78ddaac4700286a35e9badcc32376dd9"
RUNTIME_REVISION = "d7b28968306207fd176e7ff10796ccb6be607865"
WIDTH, OLD_FFN, NEW_FFN = 4096, 12288, 20480
TRUNK = 32
CHUNK = 4 * 1024**2
PQ2 = gguf.GGMLQuantizationType.PQ2_0
Q4 = gguf.GGMLQuantizationType.Q4_0
VALUE = gguf.GGUFValueType
PQ2_ZERO = b"\x00\x00" + b"\x55" * 32   # scale +0 and all ternary codes 1
Q4_ZERO = b"\x00\x00" + b"\x88" * 16    # scale +0 and all quant codes 8
SIGN_WIDTHS = "prism.hadamard.sign_widths"
SIGN_VALUES = "prism.hadamard.sign_values"
NEXTN = "qwen35.nextn_predict_layers"
DRAFT_IDS = f"blk.{TRUNK}.nextn.draft_head_ids.weight"


def read_exact(file, size):
    data = file.read(size)
    if len(data) != size:
        raise EOFError(f"Expected {size} bytes at offset {file.tell() - len(data)}")
    return data


def file_sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def metadata_records(file):
    """Locate each serialized KV without expanding the large tokenizer arrays."""
    file.seek(0)
    header = Header(file)
    if header.read(4) != b"GGUF" or header.num("I") != 3:
        raise ValueError("Expected little-endian GGUF v3")
    tensor_count, count = header.num("Q"), header.num("Q")
    spans = {}
    for _ in range(count):
        start = header.position
        key = header.string()
        header.value(header.num("I"), keep=False)
        spans[key] = slice(start, header.position)
    if len(spans) != count:
        raise ValueError("Duplicate GGUF metadata fields")
    file.seek(0)
    raw = memoryview(read_exact(file, header.position))
    return tensor_count, {key: raw[span] for key, span in spans.items()}


def sign_array(key, record, subtype):
    start = 8 + len(key)
    kind, element_type, count = struct.unpack_from("<IIQ", record, start)
    raw = record[start + 16:]
    if (kind, element_type, len(raw)) != (VALUE.ARRAY, subtype, 4 * count):
        raise ValueError(f"Unexpected metadata array type or length for {key}")
    return raw


def tensor_bytes(desc):
    values, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(desc["type_id"])]
    if desc["dims"][0] % values:
        raise ValueError(f"Unaligned quantized dimension {desc}")
    return prod(desc["dims"]) // values * size


def padded_tensors(originals, mtp, alignment):
    downs, changes = set(), {}
    for layer in range(TRUNK):
        for kind in ("down", "gate", "up"):
            name = f"blk.{layer}.ffn_{kind}.weight"
            down = kind == "down"
            old = originals[name]
            expected = [OLD_FFN, WIDTH] if down else [WIDTH, OLD_FFN]
            if old["dims"] != expected or old["type_id"] != PQ2:
                raise ValueError(f"Unexpected language FFN descriptor {name}")
            changes[name] = [NEW_FFN, WIDTH] if down else [WIDTH, NEW_FFN]
            if down:
                downs.add(name)

    tensors, cursor = {}, 0
    for name, desc in (originals | mtp).items():
        new = {"dims": changes.get(name, desc["dims"]), "type_id": desc["type_id"], "offset": cursor}
        size = tensor_bytes(new)
        if name in mtp and size != desc["bytes"]:
            raise ValueError(f"Unexpected MTP byte count in the saved layout for {name}")
        tensors[name] = new
        cursor += gguf.GGUFWriter.ggml_pad(size, alignment)
    return tensors, downs, changes


def write_header(path, alignment, old_metadata, tensors, extended_signs):
    # GGUFWriter serializes the changed KVs and all descriptors. Preserve every
    # other source KV byte-for-byte, including the tokenizer table and its order.
    writer = gguf.GGUFWriter(path, "qwen35")
    writer.data_alignment = alignment
    writer.remove_key("general.architecture")
    writer.add_block_count(TRUNK + 1)
    writer.add_feed_forward_length(NEW_FFN)
    writer.add_uint32(NEXTN, 1)
    writer.add_key_value(SIGN_WIDTHS, [WIDTH, NEW_FFN], VALUE.ARRAY, VALUE.INT32)
    writer.add_key_value(SIGN_VALUES, extended_signs, VALUE.ARRAY, VALUE.INT32)
    for name, desc in tensors.items():
        writer.add_tensor_info(name, tuple(reversed(desc["dims"])), np.dtype("int8"),
                               tensor_bytes(desc), raw_dtype=gguf.GGMLQuantizationType(desc["type_id"]))
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
    finally:
        writer.close()

    with path.open("r+b") as file:
        count, edits = metadata_records(file)
        descriptors = file.read()
        fields = old_metadata | edits
        file.seek(0)
        file.write(struct.pack("<4sIQQ", b"GGUF", 3, count, len(fields)))
        for record in fields.values():
            file.write(record)
        file.write(descriptors)
        writer.write_padding(file, file.tell())
        return file.tell(), edits


def pattern_chunks(pattern, size):
    if size % len(pattern):
        raise ValueError("Partial zero block")
    span = CHUNK // len(pattern) * len(pattern)
    chunk = pattern * (min(size, span) // len(pattern))
    for first in range(0, size, span):
        yield chunk[:size - first]


def stub_pattern(name, desc):
    if name == DRAFT_IDS:
        return struct.pack(f"<{desc['dims'][0]}i", *range(desc["dims"][0]))
    if desc["type_id"] == PQ2:
        return PQ2_ZERO
    if desc["type_id"] == Q4:
        return Q4_ZERO
    return b"\x00"


def row_batches(old, new):
    rows = old["dims"][1]
    before, after = tensor_bytes(old) // rows, tensor_bytes(new) // rows
    n_rows = CHUNK // after
    for first in range(0, rows, n_rows):
        yield min(n_rows, rows - first), before, after


def write_payloads(source, path, originals, tensors, downs, start, alignment):
    with path.open("ab") as output:
        for name, desc in tensors.items():
            size = tensor_bytes(desc)
            old = originals.get(name)
            if old is None:
                for chunk in pattern_chunks(stub_pattern(name, desc), size):
                    output.write(chunk)
            else:
                source.seek(start + old["offset"])
                if name in downs:
                    for rows, before, after in row_batches(old, desc):
                        block = np.empty((rows, after), dtype=np.uint8)
                        block[:, :before] = np.frombuffer(read_exact(source, rows * before), np.uint8).reshape(rows, before)
                        block[:, before:] = np.frombuffer(PQ2_ZERO * ((after - before) // len(PQ2_ZERO)), np.uint8)
                        output.write(block)
                else:
                    old_size = tensor_bytes(old)
                    for first in range(0, old_size, CHUNK):
                        output.write(read_exact(source, min(CHUNK, old_size - first)))
                    for chunk in pattern_chunks(PQ2_ZERO, size - old_size):
                        output.write(chunk)
            output.write(bytes((-size) % alignment))


def verify_header(file, originals, edits, tensors, output_start, signs):
    parsed = Header(file).parse()
    metadata = parsed["metadata"]
    widths = metadata[SIGN_WIDTHS]
    sign_summary = metadata["prism.hadamard.sign_values_summary"]
    if (parsed["tensor_data_offset"] != output_start or list(parsed["tensors"]) != list(tensors)
            or parsed["tensors"] != tensors or metadata["qwen35.block_count"] != TRUNK + 1
            or metadata[NEXTN] != 1 or metadata["qwen35.feed_forward_length"] != NEW_FFN
            or widths != [WIDTH, NEW_FFN]):
        raise ValueError("Unexpected padded GGUF geometry")
    for width, expected in ((WIDTH, signs[:4 * WIDTH]), (NEW_FFN, signs[4 * WIDTH:])):
        if sign_summary[width]["sha256_i32_le"] != hashlib.sha256(expected).hexdigest():
            raise ValueError(f"Unexpected Hadamard signs for width {width}")

    file.seek(24)
    for key, expected in (originals | edits).items():
        if read_exact(file, len(expected)) != expected:
            raise ValueError(f"Unexpected raw metadata for {key}")
    if parsed["metadata_count"] != len(originals | edits):
        raise ValueError("Unexpected padded GGUF metadata count")
    return parsed


def verify_payloads(source, output, originals, tensors, downs, source_start, alignment, output_digest):
    def checked_read(size):
        data = read_exact(output, size)
        output_digest.update(data)
        return data

    preserved, stub = {}, {}
    for name, desc in tensors.items():
        size = tensor_bytes(desc)
        old = originals.get(name)
        digest = hashlib.sha256()
        if old is None:
            for chunk in pattern_chunks(stub_pattern(name, desc), size):
                actual = checked_read(len(chunk))
                if actual != chunk:
                    raise ValueError(f"Unexpected MTP stub payload in {name}")
                digest.update(actual)
            stub[name] = {"dims": desc["dims"], "type_id": desc["type_id"], "bytes": size,
                          "sha256": digest.hexdigest()}
        else:
            source.seek(source_start + old["offset"])
            old_size = tensor_bytes(old)
            if name in downs:
                for rows, before, after in row_batches(old, desc):
                    actual = np.frombuffer(checked_read(rows * after), np.uint8).reshape(rows, after)
                    old_data = read_exact(source, rows * before)
                    prefix = actual[:, :before].tobytes()
                    padding = actual[:, before:].tobytes()
                    if prefix != old_data or padding != PQ2_ZERO * (len(padding) // len(PQ2_ZERO)):
                        raise ValueError(f"Changed language down-matrix row prefix or nonneutral padding in {name}")
                    digest.update(prefix)
            else:
                for first in range(0, old_size, CHUNK):
                    actual = checked_read(min(CHUNK, old_size - first))
                    if actual != read_exact(source, len(actual)):
                        raise ValueError(f"Changed language tensor bytes in {name} at offset {first}")
                    digest.update(actual)
                for chunk in pattern_chunks(PQ2_ZERO, size - old_size):
                    if checked_read(len(chunk)) != chunk:
                        raise ValueError(f"Nonneutral language gate/up padding in {name}")
            preserved[name] = {"dims": old["dims"], "type_id": old["type_id"], "bytes": old_size,
                               "sha256": digest.hexdigest(), "padding_bytes": size - old_size}
        pad = bytes((-size) % alignment)
        if checked_read(len(pad)) != pad:
            raise ValueError(f"Unexpected GGUF alignment bytes after {name}")

    if output.read(1):
        raise ValueError("Unexpected bytes after the final MTP tensor")
    return preserved, stub


def export(source_path, target):
    receipt = target.with_suffix(".manifest.json")
    if target.suffix.lower() != ".gguf" or target.exists() or receipt.exists():
        raise ValueError("Expected fresh .gguf and .manifest.json output paths")
    header_path, layout_path = AUDIT / "headers.json", AUDIT / "layout.json"
    pin = json.loads(header_path.read_text(encoding="utf-8"))["remote"][NAME]
    layout = json.loads(layout_path.read_text(encoding="utf-8"))["layouts"][NAME][str(NEW_FFN)]
    if (pin["published_lfs_sha256"] != SOURCE_SHA256 or pin["revision"] != MODEL_REVISION
            or int(PQ2) != 142 or gguf.GGML_QUANT_SIZES[PQ2] != (128, 34)):
        raise ValueError("Unexpected source pin or local gguf-py PQ2_0 layout")
    low, high = layout["gguf_file_size_bytes_range"]
    if shutil.disk_usage(target.parent).free < high + 1024**2:
        raise OSError(f"Need at least {high + 1024**2:,} free bytes for the verified output")

    host = pin["metadata"]
    originals = pin["tensors"]
    alignment = host.get("general.alignment", 32)
    tensors, downs, changes = padded_tensors(originals, layout["circuit_tensors"], alignment)
    folded = set(host["prism.hadamard.weight_names"]) | set(host["prism.hadamard.inverse_weight_names"])
    if (host["general.architecture"] != "qwen35" or host["qwen35.block_count"] != TRUNK
            or host["qwen35.embedding_length"] != WIDTH or host["qwen35.feed_forward_length"] != OLD_FFN
            or host.get(NEXTN, 0) or host["prism.hadamard.block_size"] != WIDTH
            or OLD_FFN % WIDTH or NEW_FFN % WIDTH or not downs <= folded
            or any(originals[name]["dims"][0] != WIDTH for name in folded - downs)):
        raise ValueError("Unexpected pinned language or Hadamard geometry")

    with source_path.open("rb") as source, tempfile.TemporaryDirectory(dir=target.parent, prefix="pad9b-") as temp:
        source_stat = os.fstat(source.fileno())
        if (source_stat.st_size != pin["file_bytes"]
                or hashlib.file_digest(source, "sha256").hexdigest() != SOURCE_SHA256):
            raise ValueError(f"Input does not match pinned SHA-256 {SOURCE_SHA256}")
        source_start = pin["tensor_data_offset"]
        source.seek(0)
        if hashlib.sha256(read_exact(source, source_start)).hexdigest() != pin["header_sha256"]:
            raise ValueError("Input header differs from the audited tensor inventory")
        source_count, metadata = metadata_records(source)
        if source_count != len(originals) or len(metadata) != pin["metadata_count"]:
            raise ValueError("Input tensor or metadata count mismatch")
        sign_widths = sign_array(SIGN_WIDTHS, metadata[SIGN_WIDTHS], VALUE.INT32)
        sign_values = sign_array(SIGN_VALUES, metadata[SIGN_VALUES], VALUE.INT32)
        original_signs = list(struct.unpack(f"<{WIDTH + OLD_FFN}i", sign_values))
        if (struct.unpack("<ii", sign_widths) != (WIDTH, OLD_FFN) or set(original_signs) != {-1, 1}):
            raise ValueError("Unexpected original Hadamard sign vectors")
        plus_one = struct.pack("<i", 1)
        extended_signs = bytes(sign_values) + plus_one * (NEW_FFN - OLD_FFN)

        staged = Path(temp) / target.name
        output_start, edits = write_header(staged, alignment, metadata, tensors,
                                           original_signs + [1] * (NEW_FFN - OLD_FFN))
        write_payloads(source, staged, originals, tensors, downs, source_start, alignment)
        size = staged.stat().st_size
        if not low <= size <= high:
            raise ValueError(f"Expected the audited padded file length {low}–{high}, found {size}")
        with staged.open("rb") as output:
            header = verify_header(output, metadata, edits, tensors, output_start, extended_signs)
            output.seek(0)
            digest = hashlib.sha256(read_exact(output, output_start))
            kept, stub = verify_payloads(source, output, originals, tensors, downs, source_start, alignment, digest)
        end_stat = os.fstat(source.fileno())
        if (source_stat.st_size, source_stat.st_mtime_ns) != (end_stat.st_size, end_stat.st_mtime_ns):
            raise ValueError("Source changed during export")
        if {name: row["padding_bytes"] for name, row in kept.items() if row["padding_bytes"]}.keys() != changes.keys():
            raise ValueError("Unexpected set of padded tensors")

        report = dict(source=str(source_path), source_sha256=SOURCE_SHA256, source_bytes=source_stat.st_size,
                      source_revision=MODEL_REVISION, runtime_revision=RUNTIME_REVISION, output=str(target),
                      sha256=digest.hexdigest(), bytes=size, header_sha256=header["header_sha256"],
                      header_data_offset=output_start, language_layers=TRUNK, mtp_layers=1, old_ffn=OLD_FFN,
                      new_ffn=NEW_FFN, metadata_fields_byte_equal=len(metadata.keys() - edits.keys()),
                      changed_metadata_fields=list(edits), source_tensor_extents_byte_equal=len(kept),
                      wholly_unchanged_tensors=len(kept) - len(changes), padded_tensors=len(changes),
                      pq2_padding_block_hex=PQ2_ZERO.hex(),
                      original_hadamard_signs_byte_equal=True, new_positive_hadamard_signs=NEW_FFN - OLD_FFN,
                      hadamard_signs=header["metadata"]["prism.hadamard.sign_values_summary"],
                      draft_token_id_range=[0, tensors[DRAFT_IDS]["dims"][0] - 1],
                      source_tensor_extents=kept, mtp_tensors=stub, script_sha256=file_sha(Path(__file__)),
                      layout_sha256=file_sha(layout_path), reproduction_argv=[sys.executable, "-B", *sys.argv])
        staged_receipt = Path(temp) / receipt.name
        staged_receipt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.link(staged, target)
        try:
            os.link(staged_receipt, receipt)
        except OSError:
            target.unlink()
            raise

    summary = {key: report[key] for key in ("output", "sha256", "bytes", "source_tensor_extents_byte_equal",
                                            "padded_tensors", "original_hadamard_signs_byte_equal")}
    print(json.dumps(summary | {"manifest": str(receipt), "mtp_tensors": len(stub)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.source.resolve(strict=True), args.output.resolve())


if __name__ == "__main__":
    main()
