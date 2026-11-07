"""Pack the existing Bonsai tensors and the exact QwenPU graph into one GGUF."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys

BENCH = Path(__file__).resolve().parents[2] / "dist/local-bench"
OUT = BENCH / "bonsai-in-graph"
for profile in ("bonsai-forward", "bonsai-in-graph"):
    gguf_py = BENCH / profile / "llama.cpp/gguf-py"
    if (gguf_py / "gguf").is_dir():
        sys.path.insert(0, str(gguf_py))
        break
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType

RUN = OUT / "checkpoint"
CHECKPOINT = RUN / "Bonsai-27B-Q1_0-QwenPU.gguf"
VERSION_KEY = "qwenpu.format_version"
PAYLOADS = {
    "qwenpu.graph": OUT / "vector.qpc",
    "qwenpu.program.1": OUT / "program-1.qps",
    "qwenpu.program.2": OUT / "program-2.qps",
    "qwenpu.program.stories260k": OUT / "stories260k/story.qps",
}
CHUNK = 16 * 1024**2


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def align(n, alignment):
    return (n + alignment - 1) // alignment * alignment


def gguf_string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def tensor_signature(tensor):
    return tensor.name, tuple(tensor.shape), tensor.tensor_type, tensor.n_bytes


def verify(source, checkpoint, base, payloads=None):
    if payloads is None:
        payloads = PAYLOADS
    combined = GGUFReader(checkpoint)
    count = len(base.tensors)
    if [tensor_signature(t) for t in combined.tensors[:count]] != [
        tensor_signature(t) for t in base.tensors
    ]:
        raise RuntimeError("The original GGUF tensor declarations changed.")
    old = [key for key in base.fields if not key.startswith("GGUF.")]
    with source.open("rb") as a, checkpoint.open("rb") as b:
        for key in old:
            original, new = base.fields[key], combined.fields[key]
            a.seek(original.offset)
            b.seek(new.offset)
            size = sum(int(part.nbytes) for part in original.parts)
            if a.read(size) != b.read(size):
                raise RuntimeError(f"The original metadata changed: {key}")
        a.seek(base.data_offset)
        b.seek(combined.data_offset)
        remaining = source.stat().st_size - base.data_offset
        digest = hashlib.sha256()
        while remaining:
            block = a.read(min(CHUNK, remaining))
            if not block or block != b.read(len(block)):
                raise RuntimeError("Bonsai's packed tensor bytes changed.")
            digest.update(block)
            remaining -= len(block)
        tensors = combined.tensors[count:]
        if [t.name for t in tensors] != list(payloads):
            raise RuntimeError("Unexpected QwenPU tensor names.")
        for tensor in tensors:
            data = payloads[tensor.name].read_bytes()
            b.seek(tensor.data_offset)
            if tensor.tensor_type != GGMLQuantizationType.I8 or tensor.n_bytes != len(data) or b.read(len(data)) != data:
                raise RuntimeError(f"The checkpoint changed the circuit payload: {tensor.name}")
    return {"original_tensors": count,
            "q1_0_tensors": sum(t.tensor_type == GGMLQuantizationType.Q1_0 for t in base.tensors),
            "original_metadata_equal": True, "original_tensor_bytes_equal": True,
            "original_tensor_data_sha256": digest.hexdigest(), "circuit_payloads_equal": True,
            "circuit_payload_bytes": sum(t.n_bytes for t in tensors)}


def build(source, output, *, payloads=None, version_key=VERSION_KEY):
    if payloads is None:
        payloads = PAYLOADS
    if source.resolve() == output.resolve():
        raise ValueError("The original Bonsai checkpoint and output must be different files.")
    base = GGUFReader(source)
    if not base.tensors or any(key in base.fields for key in (VERSION_KEY, version_key, "split.count")):
        raise ValueError("Expected one original, unsplit Bonsai GGUF.")
    alignment = int(base.alignment)
    first = int(base.tensors[0].field.offset)
    last = base.tensors[-1].field
    table_end = int(last.offset) + sum(int(part.nbytes) for part in last.parts)
    original_data_bytes = source.stat().st_size - base.data_offset
    payload_end = max(t.data_offset - base.data_offset + align(t.n_bytes, alignment) for t in base.tensors)
    if align(original_data_bytes, alignment) != payload_end:
        raise ValueError("The original tensor section has unexpected trailing data.")
    offsets = []
    offset = payload_end
    for name, path in payloads.items():
        length = path.stat().st_size
        if not length:
            raise ValueError(f"Empty circuit payload: {path}")
        offsets.append((name, length, offset))
        offset += align(length, alignment)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".partial")
    with source.open("rb") as a, tmp.open("wb") as b:
        magic, version, tensors, keys = struct.unpack("<4sIQQ", a.read(24))
        if magic != b"GGUF" or version != 3:
            raise ValueError("Expected a little-endian GGUF version 3.")
        b.write(struct.pack("<4sIQQ", magic, version, tensors + len(offsets), keys + 1))
        b.write(a.read(first - 24))
        b.write(gguf_string(version_key) + struct.pack("<II", GGUFValueType.UINT32, 1))
        b.write(a.read(table_end - first))
        for name, length, position in offsets:
            b.write(gguf_string(name))
            b.write(struct.pack("<IQIQ", 1, length, GGMLQuantizationType.I8, position))
        b.write(bytes(align(b.tell(), alignment) - b.tell()))
        a.seek(base.data_offset)
        shutil.copyfileobj(a, b, CHUNK)
        b.write(bytes(payload_end - original_data_bytes))
        for name, length, _ in offsets:
            with payloads[name].open("rb") as payload:
                shutil.copyfileobj(payload, b, CHUNK)
            b.write(bytes(align(length, alignment) - length))
    result = verify(source, tmp, base, payloads)
    tmp.replace(output)
    result.update(source_bytes=source.stat().st_size, checkpoint_bytes=output.stat().st_size)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("The original Bonsai checkpoint and output must be different files.")
    print("Packing the checkpoint and comparing original tensor bytes.", flush=True)
    report = build(args.source, args.output)
    args.output.with_suffix(".verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

