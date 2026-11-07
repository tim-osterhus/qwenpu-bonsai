"""Redact home-directory paths in a little-endian ELF32 guest without moving any bytes."""

import re
import struct
import sys
from pathlib import Path


HOME_PATH = re.compile(rb"(?:/(?:home|Users)/|[a-z]:[\\/]Users[\\/])[ -~]+", re.IGNORECASE)


def redact(elf):
    data = bytearray(elf.read_bytes())
    if data[:6] != b"\x7fELF\x01\x01":
        raise ValueError("Expected a little-endian ELF32 guest")
    phoff, shoff = struct.unpack_from("<II", data, 28)
    phsize, phnum, shsize, shnum, shstrndx = struct.unpack_from("<5H", data, 42)
    if phsize != 32 or shsize != 40 or not 0 < shstrndx < shnum:
        raise ValueError("Unexpected ELF32 header table layout")
    loads = []
    for i in range(phnum):
        kind, offset, _, _, size = struct.unpack_from("<5I", data, phoff + i * phsize)
        if kind == 1:  # PT_LOAD
            loads.append((offset, offset + size))

    sections = [struct.unpack_from("<6I", data, shoff + i * shsize) for i in range(shnum)]
    _, _, _, _, names_offset, names_size = sections[shstrndx]
    names = data[names_offset:names_offset + names_size]
    for name, kind, flags, _, offset, size in sections:
        if not names[name:].startswith(b".debug_"):
            continue
        end = offset + size
        if (kind != 1 or flags & 2 or end > len(data)  # SHT_PROGBITS, SHF_ALLOC
                or any(offset < hi and lo < end for lo, hi in loads)):
            raise ValueError("Expected non-loaded debug data")
        data[offset:end] = HOME_PATH.sub(lambda m: b"_" * len(m[0]), data[offset:end])

    if HOME_PATH.search(data):
        raise ValueError("A home-directory path remains in the guest")
    elf.write_bytes(data)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python redact_debug_paths.py GUEST.elf")
    redact(Path(sys.argv[1]))
