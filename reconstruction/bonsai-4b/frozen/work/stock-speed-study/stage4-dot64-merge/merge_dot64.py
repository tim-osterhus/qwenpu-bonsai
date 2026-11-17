"""Copy a staged 64-lane circuit into the packed Bonsai speed baseline."""

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

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY / "stage3-embedding"))
from encode_pq2_embedding import sha_file, write_gguf

BASE_SHA = "551c835c7b114f122b57785f528415693ed3b7dd3cdf187eacaf0401edcee2c6"
COMBINED, STAGED = "blk.64.", "blk.1."
HEAD = "nextn.draft_head.weight"
STAGED_HEAD = "nextn.shared_head_head.weight"
GUARD, WIDTH = 513, 5120
Q4 = gguf.GGMLQuantizationType.Q4_0
F32 = gguf.GGMLQuantizationType.F32
CHUNK = 16 * 1024**2
REPLACEMENTS = (
    "nextn.eh_proj.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight",
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
    "attn_q_norm.weight", "attn_k_norm.weight",
    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
)


def tensor_bytes(tensor):
    return tensor.data.view(np.uint8).reshape(-1)


def encode_head(tensor):
    if tensor.tensor_type != F32 or tensor.data.shape[1] != WIDTH:
        raise ValueError("Expected the staged F32 action scores")
    if len(tensor.data) <= GUARD or np.any(tensor.data[GUARD:]):
        raise ValueError("Unexpected nonzero score outside the 513 protocol IDs")
    weights = tensor.data[:GUARD + 1].reshape(GUARD + 1, WIDTH // 32, 32)
    result = np.zeros((*weights.shape[:2], 18), dtype=np.uint8)
    result[..., 2:] = 0x88
    for row, block in zip(*np.nonzero(np.any(weights != 0, axis=2)), strict=True):
        values = weights[row, block]
        scale = np.float16(np.max(np.abs(values)) / 4)
        if not np.isfinite(scale) or scale == 0:
            raise ValueError(f"Invalid packed draft-head scale for block {row}, {block}")
        quants = np.rint(values / np.float32(scale)).astype(np.int32)
        if np.any((quants < -8) | (quants > 7)) or not np.array_equal(quants * np.float32(scale), values):
            raise ValueError(f"Inexact packed draft-head block {row}, {block}")
        codes = quants + 8
        result[row, block, :2] = np.asarray([scale], dtype="<f2").view(np.uint8)
        result[row, block, 2:] = codes[:16] | (codes[16:] << 4)
    return result.reshape(GUARD + 1, -1)


def verify(source, candidate, replacements):
    result = gguf.GGUFReader(candidate)
    metadata = lambda reader: reader.data[24:reader.tensors[0].field.offset]
    if not np.array_equal(metadata(source), metadata(result)):
        raise ValueError("Combined GGUF metadata changed")
    unchanged = hashlib.sha256()
    changes = {}
    retained = 0
    for original, actual in zip(source.tensors, result.tensors, strict=True):
        name = original.name
        replacement = replacements.get(name)
        data, kind = replacement or (original.data, original.tensor_type)
        if actual.name != name or actual.tensor_type != kind or not np.array_equal(actual.shape, original.shape):
            raise ValueError(f"Wrong merged tensor descriptor for {name}")
        expected = data.view(np.uint8).reshape(-1)
        raw = tensor_bytes(actual)
        if len(raw) != len(expected):
            raise ValueError(f"Wrong merged tensor size for {name}")
        digest = hashlib.sha256() if replacement is not None else unchanged
        for first in range(0, len(raw), CHUNK):
            span = slice(first, first + CHUNK)
            if not np.array_equal(raw[span], expected[span]):
                raise ValueError(f"Wrong merged tensor payload for {name} at {first}")
            digest.update(raw[span])
        if replacement is not None:
            changes[name] = {"type": kind.name, "bytes": len(raw), "sha256": digest.hexdigest()}
        else:
            retained += 1
    return {"unchanged_tensors": retained, "unchanged_ordered_payload_sha256": unchanged.hexdigest(),
            "metadata_sha256": hashlib.sha256(metadata(result)).hexdigest(), "changed_tensors": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("staged", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plan", action="store_true", help="Verify inputs and show the exact storage delta without writing")
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = output.with_suffix(".manifest.json")
    if output.suffix != ".gguf" or output.exists() or manifest.exists():
        raise ValueError("Use a fresh .gguf output path")
    staged_manifest = json.loads(args.staged.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    staged_sha = sha_file(args.staged)
    if sha_file(args.baseline) != BASE_SHA or staged_sha != staged_manifest["sha256"]:
        raise ValueError("Input checkpoint hash mismatch")
    if sorted(staged_manifest["tokens"].values()) != list(range(GUARD)):
        raise ValueError("The staged protocol no longer fits the pinned token map")
    source = gguf.GGUFReader(args.baseline)
    staged = gguf.GGUFReader(args.staged)
    source_tensors = {tensor.name: tensor for tensor in source.tensors}
    staged_tensors = {tensor.name.removeprefix(STAGED): tensor for tensor in staged.tensors
                      if tensor.name.startswith(STAGED)}
    replacements = {}
    for name in REPLACEMENTS:
        small, full = staged_tensors[name], source_tensors[COMBINED + name]
        if not np.array_equal(small.shape, full.shape):
            raise ValueError(f"Staged and combined shapes differ for {name}")
        replacements[full.name] = small.data, small.tensor_type
    action_embedding = staged_tensors["nextn.embed_tokens.weight"].data
    tokens = np.arange(GUARD)
    lanes = 4096 + tokens
    if not np.array_equal(np.nonzero(action_embedding), (tokens, lanes)) or not np.all(action_embedding[tokens, lanes] == 1):
        raise ValueError("Staged action embedding changed; cannot reuse the packed table")
    ids = source_tensors[COMBINED + "nextn.draft_head_ids.weight"].data
    if not np.array_equal(ids, np.arange(GUARD + 1, dtype=np.int32)):
        raise ValueError("Baseline draft token map differs")
    replacements[COMBINED + HEAD] = encode_head(staged_tensors[STAGED_HEAD]), Q4
    additional_bytes = sum(data.nbytes - source_tensors[name].n_bytes for name, (data, _) in replacements.items())
    estimated_bytes = args.baseline.stat().st_size + additional_bytes
    if args.plan:
        print(json.dumps({"baseline_bytes": args.baseline.stat().st_size, "estimated_output_bytes": estimated_bytes,
                          "additional_tensor_bytes": additional_bytes, "staged_sha256": staged_sha,
                          "changed_tensors": {name: kind.name for name, (_, kind) in replacements.items()}}, indent=2))
        return
    if shutil.disk_usage(output.parent).free < estimated_bytes + 2 * 1024**3:
        raise OSError("Insufficient free space for the combined checkpoint and retained evidence")
    ordered = ((tensor.name, *replacements.get(tensor.name, (tensor.data, tensor.tensor_type))) for tensor in source.tensors)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="dot64-merge-") as directory:
        candidate = Path(directory) / output.name
        write_gguf(candidate, "qwen35", ordered, source)
        report = verify(source, candidate, replacements)
        report.update(bytes=candidate.stat().st_size, sha256=sha_file(candidate), source_sha256=BASE_SHA,
                      staged_sha256=staged_sha, additional_tensor_bytes=additional_bytes,
                      script_sha256=sha_file(Path(__file__)), dot_actions=staged_manifest["dot_actions"],
                      tokens=staged_manifest["tokens"])
        evidence = Path(directory) / manifest.name
        evidence.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.link(candidate, output)
        try:
            os.link(evidence, manifest)
        except OSError:
            output.unlink()
            raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
