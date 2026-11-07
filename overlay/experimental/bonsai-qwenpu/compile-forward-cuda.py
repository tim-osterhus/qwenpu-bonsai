"""Compile the forward kernels for the GTX 1080 using the existing NVRTC DLL."""

import argparse
import ctypes
import os
from pathlib import Path


def compile_cuda(nvrtc_directory: Path) -> bytes:
    source = Path(__file__).resolve().parent
    pointer = ctypes.c_void_p
    char_pointer = ctypes.c_char_p
    integer = ctypes.c_int
    with os.add_dll_directory(str(nvrtc_directory)):
        nvrtc = ctypes.CDLL(str(nvrtc_directory / "nvrtc64_120_0.dll"))

        def bind(name, *arguments):
            function = getattr(nvrtc, name)
            function.argtypes = arguments
            function.restype = integer
            return function

        def check(result):
            if result:
                raise RuntimeError(f"NVRTC returned status {result}")

        version = bind("nvrtcVersion", ctypes.POINTER(integer), ctypes.POINTER(integer))
        major, minor = integer(), integer()
        check(version(ctypes.byref(major), ctypes.byref(minor)))
        # NVRTC's lazy LoadLibrary call does not use Python's added search directory.
        _builtins = ctypes.CDLL(str(nvrtc_directory / f"nvrtc-builtins64_{major.value}{minor.value}.dll"))

        create = bind("nvrtcCreateProgram", ctypes.POINTER(pointer), char_pointer, char_pointer,
                      integer, ctypes.POINTER(char_pointer), ctypes.POINTER(char_pointer))
        compile_program = bind("nvrtcCompileProgram", pointer, integer, ctypes.POINTER(char_pointer))
        log_size = bind("nvrtcGetProgramLogSize", pointer, ctypes.POINTER(ctypes.c_size_t))
        get_log = bind("nvrtcGetProgramLog", pointer, char_pointer)
        cubin_size = bind("nvrtcGetCUBINSize", pointer, ctypes.POINTER(ctypes.c_size_t))
        get_cubin = bind("nvrtcGetCUBIN", pointer, char_pointer)
        destroy = bind("nvrtcDestroyProgram", ctypes.POINTER(pointer))

        program = pointer()
        kernel = source / "qwenpu-forward-cuda.cu"
        header = source / "qwenpu-forward-cuda-layout.h"
        headers = (char_pointer * 1)(header.read_bytes())
        includes = (char_pointer * 1)(header.name.encode())
        check(create(ctypes.byref(program), kernel.read_bytes(), kernel.name.encode(), 1, headers, includes))
        try:
            flags = (b"--gpu-architecture=sm_61", b"--std=c++17", b"--fmad=false",
                     b"--ftz=false", b"--prec-div=true", b"--prec-sqrt=true")
            options = (char_pointer * len(flags))(*flags)
            result = compile_program(program, len(options), options)
            size = ctypes.c_size_t()
            if result:
                check(log_size(program, ctypes.byref(size)))
                log = ctypes.create_string_buffer(size.value)
                check(get_log(program, log))
                raise RuntimeError(log.value.decode(errors="replace"))
            check(cubin_size(program, ctypes.byref(size)))
            cubin = ctypes.create_string_buffer(size.value)
            check(get_cubin(program, cubin))
            return cubin.raw
        finally:
            destroy(ctypes.byref(program))


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--nvrtc-directory", type=Path,
                        default=root / ".venv/Lib/site-packages/torch/lib")
    args = parser.parse_args()
    args.output.write_bytes(compile_cuda(args.nvrtc_directory.resolve()))


if __name__ == "__main__":
    main()
