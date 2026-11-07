# Guest programs

These stories260K builds use the vector guest in QwenPU at
`6e2767aaa59b69ea03300c0a5199d7b36debbe50`. Each ELF embeds int4 weights derived from
Karpathy's stories260K (MIT). [Pinned QwenPU guest source](https://github.com/divagr18/QwenPU/tree/6e2767aaa59b69ea03300c0a5199d7b36debbe50/guests/stories260k)

| File | Story tokens | Trace tokens | Output | Release SHA-256 |
|---|---|---|---|---|
| `stories_vx_one.elf` | 1 (`NTOK=1`) | 51,187 | `Once` | `485bf9b319b81d6ac8f006b03962f6061a030353fc5e47a9df29cefab24a5b6d` |
| `stories_vx_three.elf` | 3 (`NTOK=3`) | 150,405 | `Once upon a` | `5c50c2ac365a5e342f4abf236848751c69e46219712a76ec7f33ea6525a72ed2` |

The release redacts build machine home-directory paths in `.debug_line_str` in place.
The ELF header, all program and section headers, and all four `PT_LOAD` segments are
byte-identical to the original for each guest. The pinned QwenPU CPU reference also
reproduced both instruction trace hashes and story outputs in the
[original validation report](../results/forward/stories260k/validation.json).
That report records these pre-redaction `elf_sha256` values:

| File | Original SHA-256 |
|---|---|
| `stories_vx_one.elf` | `2a9f6da85bb5abc71f178711f0d9eef2ca64b7c6c4eb59bfa88074fe86821a40` |
| `stories_vx_three.elf` | `bed6feefe630ae8079aba4f0d7d1b6d2c25f3f23f4f9149e99fa5d33a35a7b11` |

## Inputs

Use a clean checkout of the pinned QwenPU revision. The checkout already contains
`guests/stories260k/qat_int4.npz`. The exporter uses that archive for the quantized matrices,
`stories260K.bin` for float-model tables and calibration, and `tok512.bin` for token pieces.
Place the two downloaded `.bin` files alongside the archive in the checkout.

| Input | Source | SHA-256 |
|---|---|---|
| `qat_int4.npz` | [Tracked archive at the pinned QwenPU revision](https://github.com/divagr18/QwenPU/blob/6e2767aaa59b69ea03300c0a5199d7b36debbe50/guests/stories260k/qat_int4.npz) | `13f9c67ade64d9c8ec9836b736575fa95d4ea486fc71875977084a2515a8dccf` |
| `stories260K.bin` | [Karpathy's TinyLlamas](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin) | `b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696` |
| `tok512.bin` | [Karpathy's TinyLlamas](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.bin) | `037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312` |
| `xpack-riscv-none-elf-gcc-14.3.0-1-win32-x64.zip` | [xPack GNU RISC-V Embedded GCC 14.3.0-1](https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/tag/v14.3.0-1) | `44749ebaac273114f6689360d6e1a10b7e0debe9cee7cbc24e83ef84b0c21cc5` |

The rebuild was verified on Windows x64 with Python 3.11.9, NumPy 2.4.6, and the xPack archive
above. All build sources come from the pinned checkout, including the HALT syscall shim
`guests/stories260k/exit_xp.c`, the `vx.ld` linker script, and `chipformer/rv/env/crt0.S`
with its runtime headers. The recipe uses the checked-in QAT archive directly.

## Rebuild both ELFs in PowerShell

Run from the qwenpu-bonsai root. Set the first three paths to the prepared QwenPU checkout,
Python interpreter with NumPy installed, and extracted xPack compiler. Compare the input hashes
to the table above before exporting. The export writes generated sources into the QwenPU
checkout, and the recipe writes the rebuilt release ELFs to `guests/work/rebuild`.

```powershell
$ErrorActionPreference = 'Stop'
$upstream = (Resolve-Path 'C:/path/to/QwenPU').Path
$python = (Resolve-Path 'C:/path/to/python.exe').Path
$gcc = (Resolve-Path 'C:/path/to/xpack-riscv-none-elf-gcc-14.3.0-1/bin/riscv-none-elf-gcc.exe').Path
$revision = git -C $upstream rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $revision -ne '6e2767aaa59b69ea03300c0a5199d7b36debbe50') {
    throw 'Use the pinned QwenPU revision.'
}

$guest = Join-Path $upstream 'guests/stories260k'
$runtime = Join-Path $upstream 'chipformer/rv/env'
$bundled = (Resolve-Path 'guests').Path
$out = Join-Path $bundled 'work/rebuild'
New-Item -ItemType Directory -Force -Path $out | Out-Null

& $python -B "$guest/export_vx.py" "$guest/qat_int4.npz"
if ($LASTEXITCODE -ne 0) { throw 'VX export failed.' }

$builds = @(
    @{ NTOK = 1; ELF = 'stories_vx_one.elf'; Startup = 'cccL9GWo.o' }
    @{ NTOK = 3; ELF = 'stories_vx_three.elf'; Startup = 'ccNlqp2u.o' }
)
foreach ($build in $builds) {
    $cflags = @('-march=rv32im_zba', '-mabi=ilp32', '-O2', "-DNTOK=$($build.NTOK)", "-I$runtime", "-I$guest")
    $startup = Join-Path $out $build.Startup
    $elf = Join-Path $out $build.ELF
    & $gcc @cflags -c "$runtime/crt0.S" -o $startup
    if ($LASTEXITCODE -ne 0) { throw 'Startup assembly failed.' }
    & $gcc @cflags -nostartfiles -static "-T$guest/vx.ld" -o $elf `
        $startup "$guest/stories_vx.c" "$guest/wrom_vx.c" "$guest/exit_xp.c" `
        '-Wl,--start-group' -lc -lgcc '-Wl,--end-group'
    if ($LASTEXITCODE -ne 0) { throw "Link failed for $($build.ELF)." }
    & $python -B (Join-Path $bundled 'redact_debug_paths.py') $elf
    if ($LASTEXITCODE -ne 0) { throw "Debug path redaction failed for $($build.ELF)." }
    $rebuilt = Get-FileHash -LiteralPath $elf
    $release = Get-FileHash -LiteralPath (Join-Path $bundled $build.ELF)
    if ($rebuilt.Hash -ne $release.Hash) { throw "SHA-256 mismatch for $($build.ELF)." }
    $rebuilt
}
```

The exporter generated these SHA-256 hashes on Windows.

| Generated source | SHA-256 |
|---|---|
| `weights_vx.h` | `276543c7c2c482124e06d4b85a41e7d5a7e9718f458cb9d3d688e144a4d3ca92` |
| `wrom_vx.c` | `4f2db6f3fa63d947d983440e7ca1efeb3da94da37b39bc094cfb22bfdb8d8150` |

Both rebuilt ELFs matched the bundled SHA-256 hashes after redaction. The separate assembly step
preserves the startup object names recorded in the original ELF symbol string tables. Passing
`crt0.S` directly to the link command let GCC choose random temporary object names and changed six
bytes in each tested rebuild. Those bytes are outside the loadable segments.
