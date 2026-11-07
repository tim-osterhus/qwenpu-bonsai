# Standalone QwenPU probe

Native Windows PyTorch CUDA on the GTX 1080, reusing the parent Qwenputer `.venv`. The independent RV reference produced the expected tokens. Each circuit step received the previous circuit prediction.

- Upstream QwenPU commit `6e2767aaa59b69ea03300c0a5199d7b36debbe50`.
- `stories_vx.elf` uses the upstream vector guest, `export_vx.py qat_int4.npz`, and xPack GNU RISC-V GCC 14.3.0-1. It was built with `NTOK=1` for the probe. The run stops at trace token 3252, before any story text.
- The first `VPUT` starts at trace token 1370, `VQ` at 3001, and `VMAT` covers 3184 through 3252. The 768-token prefix ends before these operations.
- Cold setup allocates temporary NumPy matrices on the host. PyTorch CUDA reservation reports allocator-owned memory. CUDA context overhead and Windows shared GPU memory are outside that counter.
- The probe limits the worker to two logical CPUs. The supervisor skips a run above 80% average preflight CPU or below 6 GiB available RAM and stops it at 150 seconds or below 4 GiB available RAM.

Run from the Qwenputer checkout root (see docs/REPRODUCE.md) in PowerShell. In that checkout this folder lives at `dist/local-bench/qwenpu-trace`.

```powershell
$p = Join-Path (Get-Location) 'dist/local-bench/qwenpu-trace'
$env:OMP_NUM_THREADS = $env:MKL_NUM_THREADS = $env:OPENBLAS_NUM_THREADS = '2'
$env:TEMP = $env:TMP = $p
$env:CUDA_CACHE_PATH = Join-Path $p 'cuda-cache'
& ./.venv/Scripts/python.exe -B "$p/probe.py" rv
& ./.venv/Scripts/python.exe -B "$p/probe.py" vx
& ./.venv/Scripts/python.exe -B "$p/probe.py" stories --steps 3252 --elf "$p/stories_vx.elf"
```

Each run emits JSON lines with graph size, setup time, trace tokens, parity, throughput, and sampled CPU and RAM. The result is also in the corresponding `*-child.log`.
