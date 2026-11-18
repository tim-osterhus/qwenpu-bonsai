# Reproduce Bonsai–QwenPU on stock Prism b9596

This guide uses the **6,920,450,208-byte stock GGUF** and the packaged Windows launcher. The full three-token story passed on Linux with an RTX 3070. The Windows GTX 1080 passed the vector smoke test and the story through its first matrix multiply. The separate [6.047 GB b12128 checkpoint](https://huggingface.co/millrace-ai/qwenpu-bonsai-27b-stock-prism/tree/764da1a0a9f6f5cd68670a7a28eb2410a773c491/stock-prism-b12128) matches the full Windows run. The commands below apply to the b9596 package. The original v0.1.0 artifact uses a patched CUDA runtime. [Full Windows b12128 evidence](../results/forward/standard-layer/windows-stories260k/README.md) · [Stock release write-up](STOCK-PRISM-RELEASE.md) · [Earlier patched build](https://github.com/tim-osterhus/qwenpu-bonsai/blob/v0.1.0/docs/REPRODUCE.md) · [FP16 projection experiment](STOCK-PRISM-RELEASE.md#fp16-circuit-projections-on-windows)

Bonsai 2 4B, Bonsai 2 9B, and Qwen3.5-4B UD-IQ2_M also have verified full Windows runs on b12128. The Bonsai 4B package includes full-story commands. The 9B and Qwen3.5 packages document the bounded capture and link the full-run results. [Bonsai 4B package](https://huggingface.co/millrace-ai/qwenpu-bonsai-4b-stock-prism) · [9B package](https://huggingface.co/millrace-ai/qwenpu-bonsai-9b-stock-prism) · [Qwen3.5 package](https://huggingface.co/millrace-ai/qwenpu-qwen35-4b-iq2-m-stock-prism)

## Download and verify the Windows package

Use Windows x64, PowerShell 7, Git, Python 3.11 with the `py` launcher, the Hugging Face `hf` CLI, and an NVIDIA driver compatible with CUDA 12.4. The combined checkpoint and original Bonsai control require about 10.72 GB on disk, plus space for the runtime, dependencies, and output. Close other GPU jobs if memory is tight. [Pinned package README](https://huggingface.co/millrace-ai/qwenpu-bonsai-27b-stock-prism/blob/stock-prism-b9596/stock-prism/README.md) · [Official Prism b9596 binaries](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b9596-9fcaed7)

From an empty working directory, download the pinned stock package and enter its folder. `Setup.ps1` fetches and verifies Prism's release archives, checks out the pinned QwenPU reference, and installs the Python verifier dependencies in a local virtual environment.

```powershell
hf download millrace-ai/qwenpu-bonsai-27b-stock-prism --revision stock-prism-b9596 --include "stock-prism/*" --local-dir .\stock-prism-download
Set-Location .\stock-prism-download\stock-prism
pwsh -NoProfile -File .\Setup.ps1
```

Download the original Bonsai Q1_0 control, or set `$original` to an existing copy with the same hash. `Run.ps1` verifies the package files, the original control, the runtime DLLs, and the reference revision before running the guest. [Original Bonsai revision](https://huggingface.co/prism-ml/Bonsai-27B-gguf/tree/f10afb355f104535e3e3e98cf7ab7795c72bd292) · [Package checksum list](https://huggingface.co/millrace-ai/qwenpu-bonsai-27b-stock-prism/blob/stock-prism-b9596/stock-prism/SHA256SUMS.txt)

```powershell
hf download prism-ml/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf --revision f10afb355f104535e3e3e98cf7ab7795c72bd292 --local-dir .\control
$original = (Resolve-Path -LiteralPath .\control\Bonsai-27B-Q1_0.gguf).Path
Get-FileHash -Algorithm SHA256 -LiteralPath @('.\Bonsai-27B-Q1_0-QwenPU-Stock-Prism-b9596.gguf', $original)
```

| Input | Expected SHA-256 or revision |
| --- | --- |
| `Bonsai-27B-Q1_0-QwenPU-Stock-Prism-b9596.gguf` | `bc94c1ab8112499ba5b2690f3e0a1ceb11beaa5ae92ea11606812504b37353bc` |
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

## Full story evidence for the b9596 package

The complete b9596 run uses this same downloadable GGUF on Linux and an RTX 3070. The report pins the Linux runtime archive and launcher and records the capture hashes, reference trace, memory, and handoff. The download supplies a Windows launcher. [Full Linux report](https://github.com/tim-osterhus/qwenpu-bonsai/blob/fecd02bd0764263f1456b4bbd787293d5268a928/results/forward/standard-layer/stock-mtp-vx-stories-three.json)

The package also exposes `Run.ps1 -Mode Three -OriginalBonsai $original` for an **optional, unvalidated full Windows run** of this b9596 checkpoint. The package requires at least 5 GiB of free capture space. The launcher uses a 42-hour timeout. A passing full run must reach halt, match the architectural trace and printed bytes `Once upon a\n`, and verify the Bonsai handoff. [Packaged Windows launcher](https://huggingface.co/millrace-ai/qwenpu-bonsai-27b-stock-prism/blob/stock-prism-b9596/stock-prism/Run.ps1) · [Package timing notes](https://huggingface.co/millrace-ai/qwenpu-bonsai-27b-stock-prism/blob/stock-prism-b9596/stock-prism/README.md)
