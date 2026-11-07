"""Export a retained forward checkpoint for the bundled one- and three-token guests."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from export_fixture import UPSTREAM
from export_forward_fixture import (
    GRAPH_NAME, OUTPUT, ROOT, TOKENIZER, VERSION_KEY, core, export_state,
    make_circuit, ref, write_graph,
)
from pack_checkpoint import build, hash_file
from chipformer.rv import elf

RUN = OUTPUT / "stories260k"
FORMAT = "qwenpu-forward-stories260k-v1"
ONE_GUEST = UPSTREAM.parent / "stories_vx.elf"
THREE_GUEST = RUN / "stories_vx_three.elf"
SNAPSHOTS = {key: f"qwenpu.forward.program.stories260k.{key}" for key in ("one", "three")}
SOURCE_SHA256 = "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0"
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
GUEST_SHA256 = {
    "one": "485bf9b319b81d6ac8f006b03962f6061a030353fc5e47a9df29cefab24a5b6d",
    "three": "5c50c2ac365a5e342f4abf236848751c69e46219712a76ec7f33ea6525a72ed2",
}
REVISIONS = {
    "qwenputer": "bec1a0c9654ac8151b6f36ed0154a19c53141384",
    "qwenpu": "6e2767aaa59b69ea03300c0a5199d7b36debbe50",
    "prism_llama_cpp": "9fcaed763ccda38ea81068ad9d7f991aaddca451",
    "bonsai_27b_gguf": "f10afb355f104535e3e3e98cf7ab7795c72bd292",
    "qwen_3_5_0_8b": "2fc06364715b967f1860aea9cf38778875588b17",
}


def manifest_path(checkpoint):
    return checkpoint.with_suffix(".manifest.json")


def file_info(path):
    return {"bytes": path.stat().st_size, "sha256": hash_file(path)}


def checked_file(path, expected_hash):
    result = file_info(path)
    if result["sha256"] != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {path}. Expected {expected_hash}.")
    return result


def check_source_revisions():
    for label, checkout in (("qwenputer", ROOT), ("qwenpu", UPSTREAM),
                            ("prism_llama_cpp", OUTPUT / "llama.cpp")):
        revision = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True, timeout=10,
        ).strip()
        if revision != REVISIONS[label]:
            raise ValueError(f"{label} must be at revision {REVISIONS[label]}.")


def export(source, tokenizer, guests, checkpoint):
    if checkpoint.suffix.lower() != ".gguf" or source.resolve() == checkpoint.resolve():
        raise ValueError("Choose a new .gguf path distinct from the stock model.")
    manifest = manifest_path(checkpoint)
    if checkpoint.exists() or manifest.exists():
        raise FileExistsError("Choose a new path for the checkpoint and its manifest.")
    source_info = checked_file(source, SOURCE_SHA256)
    tokenizer_info = checked_file(tokenizer, TOKENIZER_SHA256)
    guest_info = {key: checked_file(path, GUEST_SHA256[key]) for key, path in guests.items()}
    check_source_revisions()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".story-export-", dir=checkpoint.parent) as directory:
        staging = Path(directory)
        text, vocab, circuit, layout, widened, ballast, levels = make_circuit(source, tokenizer)
        graph = staging / "forward.qpf"
        write_graph(graph, circuit, levels, text, vocab, widened)
        payloads = {GRAPH_NAME: graph}
        for key, guest in guests.items():
            snapshot = staging / f"{key}.qps"
            core.snapshot(circuit, ref.Machine.from_program(elf.program(guest)))
            export_state(snapshot, levels)
            payloads[SNAPSHOTS[key]] = snapshot
        staged_checkpoint = staging / checkpoint.name
        verification = build(source, staged_checkpoint, payloads=payloads, version_key=VERSION_KEY)
        data = {
            "format": FORMAT,
            "revisions": REVISIONS,
            "stock_gguf": source_info,
            "tokenizer": tokenizer_info,
            "guests": guest_info,
            "checkpoint": file_info(staged_checkpoint),
            "payloads": {name: file_info(path) for name, path in payloads.items()},
            "tensor_verification": verification,
            "circuit": {"text_vocab": vocab, "chip_dims": len(circuit.dims),
                        "allocated_slots": layout.n_slots, "widened_width": widened,
                        "ballast_slots": ballast, "chip_layers": len(levels)},
        }
        staged_manifest = staging / manifest.name
        staged_manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        staged_checkpoint.rename(checkpoint)
        staged_manifest.rename(manifest)
    print(f"Exported {checkpoint}")
    print(f"SHA-256 {data['checkpoint']['sha256']}")
    print(f"Manifest {manifest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned stock Bonsai GGUF")
    parser.add_argument("--checkpoint", type=Path, required=True, help="New .gguf path to retain")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--one-guest", type=Path, default=ONE_GUEST)
    parser.add_argument("--three-guest", type=Path, default=THREE_GUEST)
    args = parser.parse_args()
    export(args.source, args.tokenizer, {"one": args.one_guest, "three": args.three_guest}, args.checkpoint)


if __name__ == "__main__":
    main()
