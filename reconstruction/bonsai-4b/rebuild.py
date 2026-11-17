"""Reconstruct the released Bonsai 2 4B QwenPU GGUF from its original base."""

import argparse
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from time import perf_counter


HERE = Path(__file__).resolve().parent
ARCHIVE_RUN = Path("work/stock-speed-study/stage5-portability/native4b/frontier/rethink/combined-run")
MODEL_NAME = "Bonsai-2-4B-PQ2_0-QwenPU-Stock-Prism.gguf"
BASE_REVISION = "50ebba61be85e5be07e8083db468ace0fc1bac43"
BASE_BYTES = 1146465120
BASE_SHA256 = "d406b1e912c0d2a4c124ad10bf38a76ae662fdc4d24b176451403dc7b7f30409"
RELEASE_REVISION = "c0b9de4dd5ad7057ad677fb01f7e99e4ce350e00"
RELEASE_BYTES = 4031384384
RELEASE_SHA256 = "ec6d177b3af28436f8aef2c82166ccd01ee087c5aed889b13a269fa414b6c973"
CHUNK = 8 * 1024**2


def identity(path, size, expected):
    with path.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size != size:
            raise ValueError(f"Unexpected file size for {path}")
        actual = file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual}")
    return actual


def verify_sources():
    pins = json.loads((HERE / "source-checksums.json").read_bytes())
    for name, digest in pins.items():
        path = HERE / name
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source differs from its checksum: {name}")


def compare_release(candidate, released):
    """Compare the model bytes directly and confirm the pinned release identity."""
    digest = sha256()
    checked = 0
    with candidate.open("rb") as rebuild, released.open("rb") as download:
        while chunk := rebuild.read(CHUNK):
            reference = download.read(len(chunk))
            if chunk != reference:
                raise ValueError(f"Released GGUF differs in the chunk starting at byte {checked}")
            digest.update(reference)
            checked += len(reference)
        if download.read(1) or checked != RELEASE_BYTES or digest.hexdigest() != RELEASE_SHA256:
            raise ValueError("Compared bytes differ from the pinned release size or SHA-256")
    return checked


def rebuild(base, output, released):
    import numpy as np

    if sys.platform != "win32" or sys.version_info[:2] != (3, 11) or struct.calcsize("P") != 8:
        raise RuntimeError("Use 64-bit Python 3.11 on native Windows")
    if np.__version__ != "1.23.5":
        raise RuntimeError("Install the dependencies in requirements.txt")
    verify_sources()
    identity(base, BASE_BYTES, BASE_SHA256)
    if shutil.disk_usage(output.parent).free < 10 * 1024**3:
        raise OSError("The rebuild needs at least 10 GiB of free scratch space")
    output.mkdir()
    start = perf_counter()
    env = os.environ | {
        "PYTHONPATH": str(HERE / "vendor"),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
    }

    with tempfile.TemporaryDirectory(dir=output, prefix="build-") as directory:
        scratch = Path(directory)
        shutil.copytree(HERE / "frozen", scratch / "source")
        run = scratch / "source" / ARCHIVE_RUN
        exporter = run / "export/export_combined4b.py"
        raw = scratch / "raw"
        unconditioned = exporter.parent / "Ternary-Bonsai-2-4B-PQ2_0-QwenPU.gguf"
        conditioner = run / "conditioning/export_conditioned4b.py"
        conditioned = conditioner.parent / MODEL_NAME
        for command in (
            [exporter, "export-raw", raw],
            [exporter, "pack", base, raw / "index.json", unconditioned],
            [conditioner, "--output", MODEL_NAME],
        ):
            subprocess.run([sys.executable, "-B", *map(str, command)], check=True, cwd=scratch, env=env)
        identity(conditioned, RELEASE_BYTES, RELEASE_SHA256)
        compared_bytes = compare_release(conditioned, released) if released else None
        exported = json.loads(unconditioned.with_suffix(".manifest.json").read_bytes())
        conditioned_manifest = json.loads(conditioned.with_suffix(".manifest.json").read_bytes())
        receipt = {
            "status": "pass",
            "runtime": {"os": "Windows x64", "python": sys.version.split()[0], "numpy": np.__version__},
            "base": {"revision": BASE_REVISION, "bytes": BASE_BYTES, "sha256": BASE_SHA256},
            "released": {"revision": RELEASE_REVISION, "bytes": RELEASE_BYTES, "sha256": RELEASE_SHA256},
            "rebuild": {
                "bytes": conditioned.stat().st_size,
                "sha256": RELEASE_SHA256,
                "unconditioned_sha256": exported["sha256"],
                "ffn_coefficient_stream_sha256": exported["coefficient_stream_sha256"],
                "unchanged_language_tensor_extents": exported["language_tensors_byte_equal"],
                "padded_language_tensors": exported["language_tensors_with_padding"],
                "circuit_tensors": exported["mtp_tensors"],
                "conditioned_tensor": conditioned_manifest["only_changed_tensor"],
                "changed_gate_coefficients": conditioned_manifest["changed_gate_coefficients"],
            },
            "exact_bytes_compared_with_release": compared_bytes,
            "elapsed_seconds": round(perf_counter() - start, 3),
        }
        staged_receipt = scratch / "rebuild-verification.json"
        staged_receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
        conditioned.rename(output / MODEL_NAME)
        staged_receipt.rename(output / staged_receipt.name)
    print(f"PASS: rebuilt {MODEL_NAME}, SHA-256 {RELEASE_SHA256}", flush=True)
    if compared_bytes is not None:
        print(f"All {compared_bytes:,} bytes matched the released file", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="Pinned original Bonsai 2 4B GGUF")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for output and verification")
    parser.add_argument("--released", type=Path, help="Optional pinned release file for direct byte comparison")
    args = parser.parse_args()
    base = args.base.resolve(strict=True)
    output = args.output_dir.resolve()
    released = args.released.resolve(strict=True) if args.released else None
    rebuild(base, output, released)


if __name__ == "__main__":
    main()
