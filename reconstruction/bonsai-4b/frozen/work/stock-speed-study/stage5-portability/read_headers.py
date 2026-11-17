"""Read pinned GGUF v3 headers without downloading or reading tensor payloads."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import struct
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent
REVISIONS = {
    9: "82265ccc78ddaac4700286a35e9badcc32376dd9",
    4: "50ebba61be85e5be07e8083db468ace0fc1bac43",
    2: "a6032a5089ac27e96be3a80c7b42d1ad0bd89887",
}
VARIANTS = {9: ("PQ2_0", "Q2_0", "PTQ1_0", "PQ2_0-MTP-Q8_0"),
            4: ("PQ2_0", "Q2_0", "PTQ1_0", "PQ2_0-MTP-Q8_0"),
            2: ("PQ2_0", "Q2_0", "PTQ1_0")}
MAX_PREFIX = 12 * 1024**2
FORMAT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
          10: "Q", 11: "q", 12: "d"}


class Header:
    def __init__(self, file):
        self.file = file
        self.position = 0
        self.digest = sha256()

    def read(self, size):
        if size < 0 or self.position + size > MAX_PREFIX:
            raise ValueError("Exceeded maximum GGUF header size")
        data = self.file.read(size)
        if len(data) != size:
            raise EOFError("Incomplete GGUF header")
        self.position += size
        self.digest.update(data)
        return data

    def num(self, kind):
        return struct.unpack("<" + kind, self.read(struct.calcsize("<" + kind)))[0]

    def string(self, keep=True):
        data = self.read(self.num("Q"))
        return data.decode("utf-8") if keep else None

    def value(self, kind, keep=True):
        if kind == 8:
            return self.string(keep)
        if kind == 9:
            subtype = self.num("I")
            count = self.num("Q")
            if subtype == 8:
                if keep:
                    return [self.string() for _ in range(count)]
                for _ in range(count):
                    self.string(False)
            else:
                fmt = "<" + str(count) + FORMAT[subtype]
                data = self.read(struct.calcsize(fmt))
                if keep:
                    return list(struct.unpack(fmt, data))
            return {"elements": count, "element_type": subtype}
        return self.num(FORMAT[kind])

    def parse(self):
        if self.read(4) != b"GGUF" or self.num("I") != 3:
            raise ValueError("Expected little-endian GGUF v3")
        tensor_count, metadata_count = self.num("Q"), self.num("Q")
        metadata, tensors = {}, {}
        for _ in range(metadata_count):
            key = self.string()
            metadata[key] = self.value(self.num("I"), not key.startswith("tokenizer."))
        for _ in range(tensor_count):
            name = self.string()
            dims = [self.num("Q") for _ in range(self.num("I"))]
            tensors[name] = {"dims": dims, "type_id": self.num("I"), "offset": self.num("Q")}
        if len(tensors) != tensor_count or len(metadata) != metadata_count:
            raise ValueError("Duplicate GGUF tensor or metadata names")
        alignment = metadata.get("general.alignment", 32)
        self.read((-self.position) % alignment)
        signs = metadata.pop("prism.hadamard.sign_values", [])
        if signs:
            start = 0
            details = {}
            for width in metadata["prism.hadamard.sign_widths"]:
                section = signs[start:start + width]
                if len(section) != width or set(section) - {-1, 1}:
                    raise ValueError("Invalid explicit Hadamard signs")
                details[width] = {"sha256_i32_le": sha256(struct.pack(f"<{width}i", *section)).hexdigest(),
                                  "negative": section.count(-1)}
                start += width
            if start != len(signs):
                raise ValueError("Unconsumed Hadamard signs")
            metadata["prism.hadamard.sign_values_summary"] = details
        return {"gguf_version": 3, "tensor_count": tensor_count, "metadata_count": metadata_count,
                "tensor_data_offset": self.position, "header_sha256": self.digest.hexdigest(),
                "metadata": metadata, "tensors": tensors}


def remote_inventory(size):
    model = f"Ternary-Bonsai-2-{size}B"
    repo = f"prism-ml/{model}-gguf"
    revision = REVISIONS[size]
    tree_url = f"https://huggingface.co/api/models/{repo}/tree/{revision}"
    with urlopen(tree_url, timeout=30) as response:
        tree = {entry["path"]: entry for entry in json.load(response)}
    return [(repo, revision, tree[f"{model}-{variant}.gguf"]) for variant in VARIANTS[size]]


def remote_header(item):
    repo, revision, entry = item
    name = entry["path"]
    end = MAX_PREFIX - 1
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true"
    request = Request(url, headers={"Range": f"bytes=0-{end}", "Accept-Encoding": "identity"})
    with urlopen(request, timeout=40) as response:
        expected = f"bytes 0-{end}/{entry['size']}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise ValueError(f"Server did not honor bounded range for {name}")
        report = Header(response).parse()
    report.update(repo=repo, revision=revision, filename=name, file_bytes=entry["size"],
                  published_lfs_sha256=entry["lfs"]["oid"], content_range=expected)
    print(f"{name} header={report['tensor_data_offset']} tensors={report['tensor_count']}", flush=True)
    return name, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "headers.json")
    args = parser.parse_args()
    down = HERE.parent / "stage4-dot64-later/down.gguf"
    with down.open("rb") as file:
        baseline = Header(file).parse()
    baseline.update(file_bytes=down.stat().st_size, filename=down.name)
    with ThreadPoolExecutor(max_workers=3) as pool:
        inventory = [item for group in pool.map(remote_inventory, REVISIONS) for item in group]
        remote = dict(pool.map(remote_header, inventory))
    args.out.write_text(json.dumps({"stage4_down": baseline, "remote": remote}, indent=2) + "\n",
                        encoding="utf-8")


if __name__ == "__main__":
    main()
