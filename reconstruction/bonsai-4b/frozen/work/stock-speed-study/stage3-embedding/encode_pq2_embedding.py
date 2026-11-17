"""Verify the r8a action embedding and encode its exact values as stock PQ2_0."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import gguf
import numpy as np


TRIMMED_SHA256 = "cba7955daeb75b7b1cf9c6751d96dcc5af358951972c2ef8f8f7bc92e1450172"
MANIFEST_SHA256 = "ab3098038ff20dbd2e73222b1577e7a384c580f9f3045a1d862c89044e4d9e7f"
EMBEDDING_SHA256 = "17f9f2c3ca4c6c8a054e21bbf382a5ed974d96bd810ded69ac340648dd32b399"
RUNTIME_REVISION = "d7b28968306207fd176e7ff10796ccb6be607865"
EMBEDDING = "blk.64.nextn.embed_tokens.weight"
WIDTH, VOCAB, ACTIONS, ACTION_LANE = 5120, 248320, 513, 4096
Q4, PQ2 = gguf.GGMLQuantizationType.Q4_0, gguf.GGMLQuantizationType.PQ2_0
Q4_ZERO = np.frombuffer(b"\x00\x00" + b"\x88" * 16, dtype=np.uint8)
PQ2_ZERO = np.frombuffer(b"\x00\x00" + b"\x55" * 32, dtype=np.uint8)
PROOF_IDS = np.r_[np.arange(ACTIONS + 2), VOCAB - 1]


def sha_file(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def audit_embedding(tensor):
    if (tensor.tensor_type != Q4 or tuple(tensor.shape) != (WIDTH, VOCAB)
            or tensor.data.shape != (VOCAB, WIDTH // 32 * 18)):
        raise ValueError("Unexpected frozen Q4_0 action embedding geometry")
    rows = np.arange(ACTIONS)
    lanes = ACTION_LANE + rows
    expected = np.tile(Q4_ZERO, (ACTIONS, WIDTH // 32, 1))
    blocks, within = divmod(lanes, 32)
    # The frozen nonzero block uses scale 0.25 and code 12, which decode to +1.
    expected[rows, blocks, 1] = 0x34
    expected[rows, blocks, 2 + within % 16] |= (4 << (4 * (within // 16))).astype(np.uint8)
    if not np.array_equal(tensor.data[:ACTIONS], expected.reshape(ACTIONS, -1)):
        raise ValueError("An action row differs from the frozen one-hot Q4_0 encoding")
    for first in range(ACTIONS, VOCAB, 4096):
        chunk = tensor.data[first:first + 4096].reshape(-1, WIDTH // 32, 18)
        if not np.all(chunk == Q4_ZERO):
            raise ValueError(f"Noncanonical zero Q4_0 embedding row in block starting at {first}")
    if hashlib.sha256(tensor.data).hexdigest() != EMBEDDING_SHA256:
        raise ValueError("Unexpected frozen action embedding SHA-256")


def encode_rows(token_ids):
    blocks = np.tile(PQ2_ZERO, (len(token_ids), WIDTH // 128, 1))
    rows = np.flatnonzero(token_ids < ACTIONS)
    lanes = ACTION_LANE + token_ids[rows]
    group, within = divmod(lanes, 128)
    # Stock PQ2_0 is (code - 1) * scale. Change the selected code from 01 to 10.
    blocks[rows, group, 1] = 0x3C  # F16 1.0
    blocks[rows, group, 2 + within // 4] ^= (3 << (2 * (within % 4))).astype(np.uint8)
    return blocks.reshape(len(token_ids), -1)


def verify_encoded(data, token_ids):
    for first in range(0, len(token_ids), 512):
        ids = token_ids[first:first + 512]
        blocks = data[first:first + len(ids)].reshape(-1, WIDTH // 128, 34)
        scale = blocks[:, :, :2].copy().view("<f2").astype(np.float32)
        codes = (blocks[:, :, 2:, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3
        values = ((codes.reshape(-1, WIDTH // 128, 128).astype(np.int8) - 1) * scale).reshape(-1, WIDTH)
        active = np.flatnonzero(ids < ACTIONS)
        expected = np.zeros_like(values)
        expected[active, ACTION_LANE + ids[active]] = 1
        if not np.array_equal(values, expected):
            raise ValueError(f"Decoded PQ2_0 action rows differ in chunk starting at {first}")


def write_gguf(path, arch, tensors, source=None):
    writer = gguf.GGUFWriter(path, arch)
    if source is not None:
        writer.data_alignment = source.alignment
        for name, field in source.fields.items():
            if name.startswith("GGUF.") or name == "general.architecture":
                continue
            subtype = field.types[1] if len(field.types) > 1 else None
            writer.add_key_value(name, field.contents(), field.types[0], subtype)
    for name, data, quant in tensors:
        writer.add_tensor(name, data, raw_dtype=quant)
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()


def verify_checkpoint(source, path, encoded):
    result = gguf.GGUFReader(path)
    source_meta = source.data[24:source.tensors[0].field.offset]
    result_meta = result.data[24:result.tensors[0].field.offset]
    if (result.alignment != source.alignment or result.endianess != source.endianess
            or result.fields["GGUF.version"].contents() != source.fields["GGUF.version"].contents()
            or not np.array_equal(result_meta, source_meta)):
        raise ValueError("Serialized checkpoint metadata changed")
    if [tensor.name for tensor in result.tensors] != [tensor.name for tensor in source.tensors]:
        raise ValueError("Checkpoint tensor names or order changed")
    unchanged = hashlib.sha256()
    q1_count = 0
    for original, clone in zip(source.tensors, result.tensors):
        replacing = original.name == EMBEDDING
        if (not np.array_equal(original.shape, clone.shape)
                or clone.tensor_type != (PQ2 if replacing else original.tensor_type)):
            raise ValueError(f"Unexpected tensor descriptor: {original.name}")
        wanted = (encoded if replacing else original.data).view(np.uint8).reshape(-1)
        actual = clone.data.view(np.uint8).reshape(-1)
        if len(wanted) != len(actual):
            raise ValueError(f"Unexpected tensor payload size: {original.name}")
        for first in range(0, len(wanted), 16 * 1024 * 1024):
            span = slice(first, first + 16 * 1024 * 1024)
            if not np.array_equal(wanted[span], actual[span]):
                raise ValueError(f"Unexpected tensor payload bytes: {original.name} at {first}")
            if not replacing:
                unchanged.update(actual[span])
        q1_count += clone.tensor_type == gguf.GGMLQuantizationType.Q1_0
    return dict(untouched_tensors_byte_equal=len(result.tensors) - 1, q1_0_tensors_byte_equal=q1_count,
                ordered_untouched_tensor_payload_sha256=unchanged.hexdigest(),
                metadata_fields_byte_equal=int(source.fields["GGUF.kv_count"].contents()),
                metadata_sha256=hashlib.sha256(result_meta).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("source_manifest", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--proof", type=Path, metavar="GGUF", help="Write only the bounded GET_ROWS input")
    mode.add_argument("--output", type=Path, metavar="GGUF", help="Write and verify the recoded checkpoint")
    args = parser.parse_args()

    if int(PQ2) != 142 or gguf.GGML_QUANT_SIZES[PQ2] != (128, 34):
        raise ValueError("Expected pinned GGUF Python PQ2_0 type 142 and 128/34 block geometry")
    manifest_hash = sha_file(args.source_manifest)
    if manifest_hash != MANIFEST_SHA256:
        raise ValueError("Expected the exact trimmed r8a manifest")
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    source_hash = sha_file(args.source)
    if (source_hash != TRIMMED_SHA256 or manifest["sha256"] != source_hash
            or manifest["bytes"] != args.source.stat().st_size or manifest["runtime_revision"] != RUNTIME_REVISION):
        raise ValueError("Expected the exact trimmed r8a checkpoint")
    source = gguf.GGUFReader(args.source)
    if source.endianess != gguf.GGUFEndian.LITTLE or source.fields["general.architecture"].contents() != "qwen35":
        raise ValueError("Expected the little-endian qwen35 checkpoint")
    embedding = next(tensor for tensor in source.tensors if tensor.name == EMBEDDING)
    audit_embedding(embedding)
    report = dict(source_sha256=source_hash, source_manifest_sha256=manifest_hash, source_bytes=manifest["bytes"],
                  runtime_revision=RUNTIME_REVISION, tensor=EMBEDDING, source_embedding_sha256=EMBEDDING_SHA256,
                  source_embedding_bytes=embedding.n_bytes, action_token_ids=[0, ACTIONS - 1], action_lane_base=ACTION_LANE,
                  exact_one_hot_rows=ACTIONS, exact_zero_rows=VOCAB - ACTIONS, pq2_0_type=int(PQ2),
                  pq2_0_block=[128, 34], output_embedding_bytes=VOCAB * WIDTH // 128 * 34,
                  script_sha256=sha_file(Path(__file__)), reproduction_argv=[sys.executable, "-B", *sys.argv])
    report["embedding_bytes_saved"] = embedding.n_bytes - report["output_embedding_bytes"]

    if args.audit_only:
        verify_encoded(encode_rows(PROOF_IDS), PROOF_IDS)
    else:
        target = (args.proof or args.output).resolve()
        evidence = target.with_suffix(".manifest.json")
        if target.suffix.lower() != ".gguf":
            parser.error("proof and output paths must end in .gguf")
        for destination in (target, evidence):
            if destination.exists():
                raise FileExistsError(destination)
        if args.output and shutil.disk_usage(target.parent).free < 8 * 1024**3:
            raise OSError("Need 8 GiB free for the recoded checkpoint and validation")
        with tempfile.TemporaryDirectory(prefix="pq2-embedding-", dir=target.parent) as temporary:
            candidate = Path(temporary) / target.name
            evidence_tmp = Path(temporary) / evidence.name
            ids = PROOF_IDS if args.proof else np.arange(VOCAB)
            packed = encode_rows(ids)
            verify_encoded(packed, ids)
            if args.proof:
                raw = embedding.data[ids]
                write_gguf(candidate, "qwen35", [("source_q4_0", raw, Q4), ("candidate_pq2_0", packed, PQ2)])
                report.update(proof_rows=len(ids), proof_zero_token_ids=ids[ACTIONS:].tolist(),
                              proof_source_q4_0_sha256=hashlib.sha256(raw).hexdigest())
            else:
                tensors = ((tensor.name, packed, PQ2) if tensor.name == EMBEDDING
                           else (tensor.name, tensor.data, tensor.tensor_type) for tensor in source.tensors)
                write_gguf(candidate, "qwen35", tensors, source)
                report.update(verify_checkpoint(source, candidate, packed))
            report.update(path=str(target), bytes=candidate.stat().st_size, sha256=sha_file(candidate),
                          pq2_0_payload_sha256=hashlib.sha256(packed).hexdigest())
            evidence_tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            os.link(candidate, target)
            try:
                os.link(evidence_tmp, evidence)
            except OSError:
                target.unlink()
                raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
