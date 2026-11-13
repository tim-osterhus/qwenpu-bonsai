# Reproduce Bonsai–QwenPU on stock Prism b9596

This guide uses the **6,920,450,208-byte stock GGUF** and the packaged Windows launcher. The full three-token story passed on Linux with an RTX 3070. The Windows GTX 1080 passed the vector smoke test and the story through its first matrix multiply. **A full stock Windows run is pending.** The earlier v0.1.0 artifact uses a patched CUDA runtime. The b12128 Step 3.5 experiments use a separate research checkpoint and rebuild procedure. [Stock release write-up](STOCK-PRISM-RELEASE.md) · [Earlier patched build](https://github.com/tim-osterhus/qwenpu-bonsai/blob/v0.1.0/docs/REPRODUCE.md) · [Step 3.5 scope](STOCK-PRISM-RELEASE.md#later-windows-research)

## Download and verify the Windows package

Use Windows x64, PowerShell 7, Git, Python 3.11 with the `py` launcher, the Hugging Face `hf` CLI, and an NVIDIA driver compatible with CUDA 12.4. The combined checkpoint and original Bonsai control require about 10.72 GB on disk, plus space for the runtime, dependencies, and output. Close other GPU jobs if memory is tight. [Pinned package README](https://huggingface.co/millrace-ai/qwenpu-bonsai/blob/122fc7ca76562f618ba701966df2092c3f23b305/stock-prism/README.md) · [Official Prism b9596 binaries](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b9596-9fcaed7)

From an empty working directory, download the pinned stock package and enter its folder. `Setup.ps1` fetches and verifies Prism's release archives, checks out the pinned QwenPU reference, and installs the Python verifier dependencies in a local virtual environment.

```powershell
hf download millrace-ai/qwenpu-bonsai --revision 122fc7ca76562f618ba701966df2092c3f23b305 --include "stock-prism/*" --local-dir .\stock-prism-download
Set-Location .\stock-prism-download\stock-prism
pwsh -NoProfile -File .\Setup.ps1
```

Download the original Bonsai Q1_0 control, or set `$original` to an existing copy with the same hash. `Run.ps1` verifies the package files, the original control, the runtime DLLs, and the reference revision before running the guest. [Original Bonsai revision](https://huggingface.co/prism-ml/Bonsai-27B-gguf/tree/f10afb355f104535e3e3e98cf7ab7795c72bd292) · [Package checksum list](https://huggingface.co/millrace-ai/qwenpu-bonsai/blob/122fc7ca76562f618ba701966df2092c3f23b305/stock-prism/SHA256SUMS.txt)

```powershell
hf download prism-ml/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf --revision f10afb355f104535e3e3e98cf7ab7795c72bd292 --local-dir .\control
$original = (Resolve-Path -LiteralPath .\control\Bonsai-27B-Q1_0.gguf).Path
Get-FileHash -Algorithm SHA256 -LiteralPath @('.\bonsai-stock-vx-r8a.gguf', $original)
```

| Input | Expected SHA-256 or revision |
| --- | --- |
| `bonsai-stock-vx-r8a.gguf` | `bc94c1ab8112499ba5b2690f3e0a1ceb11beaa5ae92ea11606812504b37353bc` |
| Original `Bonsai-27B-Q1_0.gguf` | `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0` |
| `probe-stock-vx.exe` | `62bd4df06b11c6c16d5c0d5f9c331704b45862d1141a0dd7f644ac73b4233818` |
| Prism b9596 CUDA 12.4 | `9fcaed763ccda38ea81068ad9d7f991aaddca451` |
| QwenPU CPU reference | `6e2767aaa59b69ea03300c0a5199d7b36debbe50` |

## Run the vector smoke test

```powershell
pwsh -NoProfile -File .\Run.ps1 -Mode Smoke -OriginalBonsai $original 2>&1 |
    Tee-Object -FilePath .\smoke-console.txt
if ($LASTEXITCODE -ne 0) { throw "Smoke test failed with exit code $LASTEXITCODE" }
```

The verifier reports **40 exact architectural attempts**, exact retained-row witnesses, and exact parked state and matched-history handoff against original Bonsai. The expected result is `vmx=1252,vsum=0,varg=0,vbest=null`. The published GTX 1080 run offloaded all 66 layers. Use the verifier's verdict to assess correctness. Timings and GPU memory depend on the reader's machine. [Clean-download smoke report](https://github.com/tim-osterhus/qwenpu-bonsai/blob/aefc7d348a794926d8804feda2c363fdfb668db9/results/forward/standard-layer/stock-mtp-vx-package-smoke.json)

The console prints the run directory under `runs/`. For an independent reproduction, keep `smoke-console.txt`, the generated `trace.json`, `bonsai-control.json`, measurement JSON, and stderr logs. Record the checkpoint and runtime identities, GPU and driver, OS, outcome, and SHA-256 of the new trace. Compare architectural results and state checks. Instrumentation and elapsed times can change the raw JSON hash between runs.

## Run the verified Windows story prefix

```powershell
pwsh -NoProfile -File .\Run.ps1 -Mode Prefix -OriginalBonsai $original
```

The one-token guest stops after the first matrix multiply. In `trace.validation.json`, expect `status` equal to `pass`, `runs[0].through_first_vmat` equal to `true`, and `runs[0].architectural_attempts` equal to `1830`. This run ends before story output. The verifier compares Bonsai's state before and after the guest in the same run. The smoke test above supplies the comparison against original Bonsai for a completed guest and handoff. [Windows validation summary](https://github.com/tim-osterhus/qwenpu-bonsai/blob/aefc7d348a794926d8804feda2c363fdfb668db9/results/forward/standard-layer/stock-mtp-vx-research.json)

## Full story evidence and Windows status

The complete published run uses the same stock GGUF on Linux and an RTX 3070. The report pins the Linux runtime archive and launcher and records the capture hashes, reference trace, memory, and handoff. The download supplies a Windows launcher. [Full Linux report](https://github.com/tim-osterhus/qwenpu-bonsai/blob/fecd02bd0764263f1456b4bbd787293d5268a928/results/forward/standard-layer/stock-mtp-vx-stories-three.json)

The package also exposes `Run.ps1 -Mode Three -OriginalBonsai $original` for an **optional, unvalidated full Windows run** of this b9596 checkpoint. The package requires at least 5 GiB of free capture space. Its roughly 32-hour GTX 1080 estimate is projected from the prefix, and its script uses a 42-hour timeout. A passing full run must reach halt, match the architectural trace and printed bytes `Once upon a\n`, and verify the Bonsai handoff. The Windows status stays **pending** until a completed validation is recorded. [Packaged Windows launcher](https://huggingface.co/millrace-ai/qwenpu-bonsai/blob/122fc7ca76562f618ba701966df2092c3f23b305/stock-prism/Run.ps1) · [Package timing notes](https://huggingface.co/millrace-ai/qwenpu-bonsai/blob/122fc7ca76562f618ba701966df2092c3f23b305/stock-prism/README.md)
