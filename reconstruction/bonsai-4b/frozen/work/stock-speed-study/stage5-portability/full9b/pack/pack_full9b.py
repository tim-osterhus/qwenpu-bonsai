"""Merge compiled MTP payloads into the validated 20,480-FFN Bonsai 9B layout."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile


HERE = Path(__file__).resolve().parent
PAD_DIR = HERE.parents[1] / "gtx1080"
sys.path.insert(0, str(PAD_DIR))
import export_pad9b as pad


AUDIT = PAD_DIR.parent
CONTROL_MANIFEST = PAD_DIR / "padded-20480.manifest.json"
SHA256 = re.compile(r"[0-9a-f]{64}")


def study_inputs():
    pin = json.loads((AUDIT / "headers.json").read_bytes())["remote"][pad.NAME]
    layout = json.loads((AUDIT / "layout.json").read_bytes())["layouts"][pad.NAME][str(pad.NEW_FFN)]
    control = json.loads(CONTROL_MANIFEST.read_bytes())
    if (pin["published_lfs_sha256"] != pad.SOURCE_SHA256 or pin["revision"] != pad.MODEL_REVISION
            or control["source_sha256"] != pad.SOURCE_SHA256 or control["source_bytes"] != pin["file_bytes"]
            or control["runtime_revision"] != pad.RUNTIME_REVISION or control["new_ffn"] != pad.NEW_FFN
            or control["source_tensor_extents_byte_equal"] != len(pin["tensors"])):
        raise ValueError("Saved source, layout and validated padded control disagree")
    return pin, layout, control


def validate_index(index_path, layout):
    raw = index_path.read_bytes()
    index = json.loads(raw)
    rows = index["ffn_rows_used"]
    if type(rows) is not int or not 1 <= rows <= pad.NEW_FFN:
        raise ValueError(f"Compiled FFN rows must fit 1–{pad.NEW_FFN}: {rows!r}")
    planned = layout["circuit_tensors"]
    supplied = index["tensors"]
    if set(supplied) != set(planned):
        raise ValueError(f"MTP tensor inventory differs: missing {sorted(set(planned) - set(supplied))}, "
                         f"unexpected {sorted(set(supplied) - set(planned))}")
    tensors = {}
    for name, expected in planned.items():
        desc = supplied[name]
        shape = {key: desc[key] for key in ("dims", "type_id", "bytes")}
        if shape != expected or pad.tensor_bytes(shape) != shape["bytes"]:
            raise ValueError(f"MTP layout differs for {name}: expected {expected}, received {shape}")
        sha = desc["sha256"]
        if not isinstance(sha, str) or not SHA256.fullmatch(sha):
            raise ValueError(f"Expected a lowercase SHA-256 for {name}")
        tensors[name] = shape | {"sha256": sha}
    return rows, tensors, hashlib.sha256(raw).hexdigest()


def payload_paths(index_path, tensors):
    paths = {name: index_path.parent / (name + ".bin") for name in tensors}
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size != tensors[name]["bytes"]:
            raise ValueError(f"Missing or wrong-sized MTP payload {path}")
    identity = pad.stub_pattern(pad.DRAFT_IDS, tensors[pad.DRAFT_IDS])
    actual = paths[pad.DRAFT_IDS].read_bytes()
    if actual != identity or hashlib.sha256(actual).hexdigest() != tensors[pad.DRAFT_IDS]["sha256"]:
        raise ValueError("Draft payload must contain the planned I32 identity row-to-token map")
    return paths


def source_header(source, pin):
    start = pin["tensor_data_offset"]
    source.seek(0)
    if hashlib.sha256(pad.read_exact(source, start)).hexdigest() != pin["header_sha256"]:
        raise ValueError("Source header differs from the pinned tensor inventory")
    count, fields = pad.metadata_records(source)
    if count != len(pin["tensors"]) or len(fields) != pin["metadata_count"]:
        raise ValueError("Unexpected source tensor or metadata count")
    widths = pad.sign_array(pad.SIGN_WIDTHS, fields[pad.SIGN_WIDTHS], pad.VALUE.INT32)
    raw = pad.sign_array(pad.SIGN_VALUES, fields[pad.SIGN_VALUES], pad.VALUE.INT32)
    signs = list(struct.unpack(f"<{pad.WIDTH + pad.OLD_FFN}i", raw))
    if struct.unpack("<ii", widths) != (pad.WIDTH, pad.OLD_FFN) or set(signs) != {-1, 1}:
        raise ValueError("Unexpected pinned Hadamard signs")
    extension = [1] * (pad.NEW_FFN - pad.OLD_FFN)
    return fields, signs + extension, bytes(raw) + struct.pack("<i", 1) * len(extension)


def write_circuit(target, tensors, paths, alignment):
    with target.open("ab") as output:
        for name, desc in tensors.items():
            size = desc["bytes"]
            digest = hashlib.sha256()
            with paths[name].open("rb") as payload:
                for first in range(0, size, pad.CHUNK):
                    data = pad.read_exact(payload, min(pad.CHUNK, size - first))
                    digest.update(data)
                    output.write(data)
                if payload.read(1) or digest.hexdigest() != desc["sha256"]:
                    raise ValueError(f"MTP payload differs from the index for {name}")
            output.write(bytes((-size) % alignment))


class TrunkReader:
    """Let the padding verifier treat the MTP boundary as end-of-file."""

    def __init__(self, file, end):
        self.file = file
        self.end = end

    def tell(self):
        return self.file.tell()

    def read(self, size):
        return self.file.read(max(0, min(size, self.end - self.tell())))


def verify_circuit(output, tensors, alignment, digest):
    for name, desc in tensors.items():
        size = desc["bytes"]
        payload_digest = hashlib.sha256()
        for first in range(0, size, pad.CHUNK):
            data = pad.read_exact(output, min(pad.CHUNK, size - first))
            payload_digest.update(data)
            digest.update(data)
        if payload_digest.hexdigest() != desc["sha256"]:
            raise ValueError(f"Serialized MTP payload differs from the index for {name}")
        padding = pad.read_exact(output, (-size) % alignment)
        if any(padding):
            raise ValueError(f"Nonzero GGUF alignment padding after {name}")
        digest.update(padding)
    if output.read(1):
        raise ValueError("Bytes after the last MTP tensor")


def pack(source_path, index_path, target, expected_sha=None):
    receipt = target.with_suffix(".manifest.json")
    if target.suffix.lower() != ".gguf" or target.exists() or receipt.exists():
        raise ValueError("Expected fresh .gguf and .manifest.json output paths")
    pin, layout, control = study_inputs()
    rows, circuit, index_sha = validate_index(index_path, layout)
    paths = payload_paths(index_path, circuit)
    if shutil.disk_usage(target.parent).free < control["bytes"] + 1024**2:
        raise OSError(f"Need at least {control['bytes'] + 1024**2:,} free bytes for the output")

    originals = pin["tensors"]
    alignment = pin["metadata"].get("general.alignment", 32)
    tensors, downs, changes = pad.padded_tensors(originals, circuit, alignment)
    trunk = {name: tensors[name] for name in originals}
    with source_path.open("rb") as source, tempfile.TemporaryDirectory(dir=target.parent, prefix="full9b-") as temp:
        source_stat = os.fstat(source.fileno())
        if (source_stat.st_size != pin["file_bytes"]
                or hashlib.file_digest(source, "sha256").hexdigest() != pad.SOURCE_SHA256):
            raise ValueError(f"Source does not match pinned SHA-256 {pad.SOURCE_SHA256}")
        fields, signs, raw_signs = source_header(source, pin)
        staged = Path(temp) / target.name
        start, edits = pad.write_header(staged, alignment, fields, tensors, signs)
        if start != control["header_data_offset"]:
            raise ValueError("Output data offset differs from the CUDA-validated padding control")
        pad.write_payloads(source, staged, originals, trunk, downs, pin["tensor_data_offset"], alignment)
        write_circuit(staged, circuit, paths, alignment)
        size = staged.stat().st_size
        if size != control["bytes"]:
            raise ValueError(f"Output length {size:,} differs from the validated layout {control['bytes']:,}")
        mtp_start = start + tensors[next(iter(circuit))]["offset"]
        with staged.open("rb") as output:
            header = pad.verify_header(output, fields, edits, tensors, start, raw_signs)
            if header["header_sha256"] != control["header_sha256"]:
                raise ValueError("Output header differs from the CUDA-validated padding control")
            output.seek(0)
            digest = hashlib.sha256(pad.read_exact(output, start))
            kept, _ = pad.verify_payloads(source, TrunkReader(output, mtp_start), originals, trunk, downs,
                                           pin["tensor_data_offset"], alignment, digest)
            if output.tell() != mtp_start or kept != control["source_tensor_extents"]:
                raise ValueError("Language tensor extents differ from the CUDA-validated padding control")
            verify_circuit(output, circuit, alignment, digest)
        end_stat = os.fstat(source.fileno())
        if (source_stat.st_size, source_stat.st_mtime_ns) != (end_stat.st_size, end_stat.st_mtime_ns):
            raise ValueError("Source changed during export")
        if expected_sha is not None and digest.hexdigest() != expected_sha:
            raise ValueError(f"Output SHA-256 {digest.hexdigest()} differs from the expected {expected_sha}")

        report = dict(source=str(source_path), source_sha256=pad.SOURCE_SHA256, source_bytes=source_stat.st_size,
                      source_revision=pad.MODEL_REVISION, runtime_revision=pad.RUNTIME_REVISION, output=str(target),
                      sha256=digest.hexdigest(), bytes=size, header_sha256=header["header_sha256"],
                      header_data_offset=start, verified_control_sha256=control["sha256"],
                      control_manifest_sha256=pad.file_sha(CONTROL_MANIFEST), circuit_index=str(index_path),
                      circuit_index_sha256=index_sha, ffn_rows_used=rows, ffn_rows_allocated=pad.NEW_FFN,
                      language_layers=pad.TRUNK, mtp_layers=1, padded_tensors=len(changes),
                      source_tensor_extents_byte_equal=len(kept), source_tensor_extents=kept,
                      original_hadamard_signs_byte_equal=True,
                      hadamard_signs=header["metadata"]["prism.hadamard.sign_values_summary"],
                      draft_token_id_range=[0, circuit[pad.DRAFT_IDS]["dims"][0] - 1], mtp_tensors=circuit,
                      script_sha256=pad.file_sha(Path(__file__)), padding_script_sha256=pad.file_sha(Path(pad.__file__)),
                      reproduction_argv=[sys.executable, "-B", *sys.argv])
        staged_receipt = Path(temp) / receipt.name
        staged_receipt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.link(staged, target)
        try:
            os.link(staged_receipt, receipt)
        except OSError:
            target.unlink()
            raise

    summary = {key: report[key] for key in ("output", "sha256", "bytes", "source_tensor_extents_byte_equal",
                                            "ffn_rows_used", "header_sha256")}
    print(json.dumps(summary | {"manifest": str(receipt), "mtp_tensors": len(circuit)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog="The circuit index lists ffn_rows_used and tensors "
                                     "with GGUF dims, type_id, bytes and sha256. Put <tensor>.bin files beside the index.")
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("circuit_index", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--plan", type=Path, metavar="CIRCUIT_INDEX", help="Check geometry and size from manifests")
    parser.add_argument("--expected-sha256", help="Require this exact output hash on replay")
    args = parser.parse_args()
    if args.plan:
        if any((args.source, args.circuit_index, args.output, args.expected_sha256)):
            parser.error("--plan takes only the circuit index")
        _, layout, control = study_inputs()
        rows, circuit, index_sha = validate_index(args.plan, layout)
        print(json.dumps(dict(bytes=control["bytes"], ffn_rows_used=rows, ffn_rows_allocated=pad.NEW_FFN,
                              circuit_index_sha256=index_sha, mtp_tensors=len(circuit),
                              mtp_payload_bytes=sum(desc["bytes"] for desc in circuit.values()),
                              mtp_kv_cells=layout["mtp_kv_cells"], mtp_kv_bytes=layout["mtp_kv_bytes"])), flush=True)
        return
    if not all((args.source, args.circuit_index, args.output)):
        parser.error("provide SOURCE.gguf CIRCUIT_INDEX.json OUTPUT.gguf")
    if args.expected_sha256 is not None and not SHA256.fullmatch(args.expected_sha256):
        parser.error("--expected-sha256 requires 64 lowercase hex digits")
    pack(args.source.resolve(strict=True), args.circuit_index.resolve(strict=True), args.output.resolve(),
         args.expected_sha256)


if __name__ == "__main__":
    main()
