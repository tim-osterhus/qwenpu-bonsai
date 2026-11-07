# Reproduce the short CUDA demo on Windows

This workflow builds the patched Prism decoder and tests the retained v0.1 GGUF against reference traces regenerated from the bundled guest ELFs. The short suite passed on a clean Windows build using the exact retained GGUF. [Validation report](../results/forward/release-v0.1/validation.json) · [Runtime identity](../results/forward/release-v0.1/runtime-identity.json)

Use Windows x64, PowerShell 7, Git, uv, and Visual Studio 2022 Build Tools with the Desktop development with C++ workload, CMake tools, and Windows SDK. Use a Pascal NVIDIA GPU supported by the pinned Prism CUDA build. The validated system ran Windows 10 22H2 on a GTX 1080 (8 GB, `sm_61`) with driver 582.66. The validation preflight requires at least 5,500 MiB free GPU memory and 8 GiB available system RAM. Allow space for the 3.8 GB story GGUF, downloads, and build outputs. Local export and the separate vector validation also need room for the stock GGUF and their derived files. [uv installation](https://docs.astral.sh/uv/getting-started/installation/)

## 1. Assemble a dedicated workspace

Run these commands from the directory containing `setup-overlay.ps1`.

~~~powershell
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

uv tool install 'huggingface-hub==2.4.0'
$env:Path = "$(uv tool dir --bin);$env:Path"
hf download --help

.\setup-overlay.ps1 -Workspace .\work
$workspace = (Resolve-Path .\work).Path
~~~

Setup checks out Qwenputer at `bec1a0c9654ac8151b6f36ed0154a19c53141384` and QwenPU at `6e2767aaa59b69ea03300c0a5199d7b36debbe50`, applies the Qwenputer patch, and copies in the overlay, trace tool, and guest ELFs. It downloads the Qwen3.5-0.8B `tokenizer.json` at `2fc06364715b967f1860aea9cf38778875588b17` and checks SHA-256 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`. [Pinned tokenizer](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17) · [Guest hashes](../guests/README.md)

## 2. Fetch and verify the Prism runtime inputs

The two runtime archives correspond to Prism llama.cpp source revision `9fcaed763ccda38ea81068ad9d7f991aaddca451`. [Prism llama.cpp release](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b9596-9fcaed7)

~~~powershell
function Assert-Sha256($file, $expected) {
    if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $expected) {
        throw "SHA-256 mismatch for $file"
    }
}

$inputs = Join-Path $workspace 'inputs'
$runtime = Join-Path $inputs 'prism-win-cuda'
New-Item -ItemType Directory -Force -Path $inputs, $runtime | Out-Null

$release = 'https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b9596-9fcaed7'
$prismZip = 'llama-prism-b1-9fcaed7-bin-win-cuda-12.4-x64.zip'
$cudaZip = 'cudart-llama-bin-win-cuda-12.4-x64.zip'
Invoke-WebRequest "$release/$prismZip" -OutFile (Join-Path $inputs $prismZip)
Invoke-WebRequest "$release/$cudaZip" -OutFile (Join-Path $inputs $cudaZip)
Assert-Sha256 (Join-Path $inputs $prismZip) '6d109e2930c0eaf2f729c3a6fc58dd7809ce2ba7047bfb294547cc389af6de5d'
Assert-Sha256 (Join-Path $inputs $cudaZip) '8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6'
Expand-Archive -LiteralPath (Join-Path $inputs $prismZip) -DestinationPath $runtime -Force
Expand-Archive -LiteralPath (Join-Path $inputs $cudaZip) -DestinationPath $runtime -Force
~~~

Use the directory containing `llama.dll`, the GGML backend DLLs, and the CUDA runtime DLLs as `$runtime`. If the archives create nested folders, collect those files in that directory before building.

## 3. Install Python dependencies and build

Run the following commands inside the assembled Qwenputer checkout. The CUDA Torch wheel provides NVRTC 12.6 for building the custom kernel. The runtime itself uses the Prism CUDA DLLs from step 2. Use `uv pip` for this environment because the Qwenputer lockfile targets a Linux Torch build.

~~~powershell
Set-Location (Join-Path $workspace 'qwenputer')
uv venv --python 3.11.9
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
$torch = 'https://download-r2.pytorch.org/whl/cu126/torch-2.14.1%2Bcu126-cp311-cp311-win_amd64.whl#sha256=4cff57ddd28c8bdf8bf2fc473f149c1b5f0063b1a6bd57593288e0bf18459adb'
uv pip install --python $python $torch
uv pip install --python $python -e '.[hf]' 'numpy==2.4.6' 'safetensors==0.8.0' `
    'transformers==5.19.0' 'tokenizers==0.23.3' 'huggingface-hub==2.4.0' `
    'psutil==7.2.2' 'PyYAML==6.0.3' 'tqdm==4.66.5'

& $python -c "import torch; assert torch.version.cuda == '12.6'; print(torch.__version__)"
$nvrtc = (Resolve-Path .\.venv\Lib\site-packages\torch\lib).Path
Get-Item (Join-Path $nvrtc 'nvrtc64_120_0.dll'), (Join-Path $nvrtc 'nvrtc-builtins64_126.dll')
nvidia-smi

.\experimental\bonsai-qwenpu\prepare-forward.ps1 -InstalledRuntime $runtime `
    -Python $python -NvrtcDirectory $nvrtc
~~~

The build script discovers Visual Studio 2022 with `vswhere`. If needed, pass `-VisualStudioPath` with the VS 2022 installation root. The script checks out the pinned Prism source and builds the patched `llama.dll`, `probe_forward.exe`, and `sm_61` cubin. It reuses GGML and CUDA DLLs from the Prism archives.

## 4. Get the story GGUF

Download the retained GGUF and its sibling manifest. Sign in with `hf auth login` if the model repository requires access. [Model files on Hugging Face](https://huggingface.co/millrace-ai/qwenpu-bonsai)

~~~powershell
$storyFile = 'Bonsai-27B-Q1_0-QwenPU-stories260k.gguf'
$manifestFile = 'Bonsai-27B-Q1_0-QwenPU-stories260k.manifest.json'
$storyCheckpoint = Join-Path $workspace $storyFile
hf download millrace-ai/qwenpu-bonsai $storyFile $manifestFile --local-dir $workspace
Assert-Sha256 $storyCheckpoint '688571d20993ccd9080fb4dc5348205c99180d03c54e922bf83b13e6377a3ef1'
Assert-Sha256 (Join-Path $workspace $manifestFile) 'b2f9e5692f54f672ce3ab49cdd9d99b5d834efe8de11a8a802a593a63d384d14'
~~~

For a local export, use a new output path and the verified stock Bonsai input. The exporter writes both the GGUF and its sibling manifest. [Pinned stock Bonsai release](https://huggingface.co/prism-ml/Bonsai-27B-gguf/tree/f10afb355f104535e3e3e98cf7ab7795c72bd292)

~~~powershell
$model = Join-Path $inputs 'Bonsai-27B-Q1_0.gguf'
hf download prism-ml/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf `
    --revision f10afb355f104535e3e3e98cf7ab7795c72bd292 --local-dir $inputs
Assert-Sha256 $model '17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0'

$storyCheckpoint = Join-Path $workspace 'Bonsai-27B-Q1_0-QwenPU-stories260k-local.gguf'
& $python -B .\experimental\bonsai-qwenpu\export_forward_story.py `
    --source $model --checkpoint $storyCheckpoint
Assert-Sha256 $storyCheckpoint '688571d20993ccd9080fb4dc5348205c99180d03c54e922bf83b13e6377a3ef1'
~~~

## 5. Validate the checkpoint

Run this from the assembled Qwenputer checkout. The validator checks the manifest against the GGUF and regenerates independent reference traces from the guest ELFs. Pass `--manifest` if the manifest is stored separately.

~~~powershell
& $python -B .\experimental\bonsai-qwenpu\validate_forward_story.py `
    --checkpoint $storyCheckpoint
~~~

The suite checks execution through the first VMAT, then the complete one- and three-token guests. Require `"full_suite_passed": true` in the run's `validation.json` under `dist/local-bench/bonsai-forward/stories260k/`. The overall `status` covers the requested tests, with exit codes 0 for `PASS`, 1 for `FAIL`, and 2 for `SKIP`. For a smoke check, add `--tests first-vmat`. The report records the exact GGUF, guest, and runtime hashes from the new run. [v0.1 story validation report](../results/forward/release-v0.1/validation.json)

## Run a retained story checkpoint

Run this from the assembled Qwenputer checkout after steps 1–4. The probe uses the three-token snapshot and the fixed prefix `Run<tool_call></tool_call>` with a 250,000-step ceiling. It writes the execution trace to `stdout.jsonl` and prints the decoded output, `Once upon a`. Step 5 compares the execution trace against an independent reference.

~~~powershell
$build = Join-Path (Get-Location) 'dist/local-bench/bonsai-forward'
$candidate = Join-Path $build 'candidate'
$demo = Join-Path $workspace 'story-demo'
New-Item -ItemType Directory -Force -Path $demo | Out-Null
$events = Join-Path $demo 'stdout.jsonl'
Remove-Item Env:QWENPU_FORWARD_CUBIN, Env:GGML_BACKEND_PATH -ErrorAction SilentlyContinue
Push-Location -LiteralPath $candidate
try {
    & (Join-Path $build 'probe_forward.exe') $candidate `
        $storyCheckpoint $demo story 250000 three 6516 248058 248059 > $events
    if ($LASTEXITCODE -ne 0) { throw 'The native story probe failed.' }
} finally {
    Pop-Location
}
$result = Get-Content -LiteralPath $events -Tail 1 | ConvertFrom-Json
if ($result.phase -ne 'result' -or !$result.released) { throw 'The story did not finish.' }
[Text.Encoding]::UTF8.GetString([byte[]]$result.echoed_bytes).TrimEnd()
~~~

## Validate the separate vector fixture

With the build from step 3 and the verified stock `$model` from the local export instructions, run the following from the assembled Qwenputer checkout. The mutation and stock logit comparisons use their own fixture GGUF. [v0.1 vector fixture report](../results/forward/release-v0.1/vector-validation.json) · [Historical vector fixture report](../results/forward/decoder-validation/validation.json)

~~~powershell
& $python -B .\experimental\bonsai-qwenpu\export_forward_fixture.py --source $model
& $python -B .\experimental\bonsai-qwenpu\select_forward_mutation.py --source $model
& $python -B .\experimental\bonsai-qwenpu\validate_forward.py `
    --stock-checkpoint $model --stock-runtime $runtime
~~~
