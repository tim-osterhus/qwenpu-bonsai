<#
.SYNOPSIS
Assemble the pinned Qwenputer checkout with the qwenpu-bonsai overlay.
.DESCRIPTION
Use a dedicated workspace. Repeat the command to resume setup or refresh the
overlay. Git and the Hugging Face hf CLI must be available on PATH.
#>
param(
    [string]$Workspace = (Join-Path $PSScriptRoot 'work')
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$QwenputerRev = 'bec1a0c9654ac8151b6f36ed0154a19c53141384'
$QwenPURev = '6e2767aaa59b69ea03300c0a5199d7b36debbe50'
$TokenizerRepo = 'Qwen/Qwen3.5-0.8B'
$TokenizerRev = '2fc06364715b967f1860aea9cf38778875588b17'
$TokenizerHash = '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'

function Invoke-Checked([string]$exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$exe $($arguments -join ' ') failed with exit code $LASTEXITCODE" }
}

function Initialize-Checkout([string]$path, [string]$remote, [string]$revision) {
    if (!(Test-Path -LiteralPath $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
        Invoke-Checked git @('-C', $path, 'init')
        Invoke-Checked git @('-C', $path, 'remote', 'add', 'origin', $remote)
    }
    $root = Invoke-Checked git @('-C', $path, 'rev-parse', '--show-toplevel')
    if ((Resolve-Path -LiteralPath $root).Path -ne (Resolve-Path -LiteralPath $path).Path) {
        throw "Expected a Git checkout at $path."
    }
    $origin = Invoke-Checked git @('-C', $path, 'remote', 'get-url', 'origin')
    if ($origin -ne $remote) { throw "Unexpected origin for $path. Expected $remote." }
    $head = & git -C $path rev-parse --verify --quiet HEAD
    if ($LASTEXITCODE -eq 1) {
        Invoke-Checked git @('-C', $path, 'fetch', '--depth', '1', 'origin', $revision)
        Invoke-Checked git @('-C', $path, 'checkout', '--detach', $revision)
    } elseif ($LASTEXITCODE -ne 0 -or $head -ne $revision) {
        throw "$path must be at pinned revision $revision."
    }
}

Get-Command git -CommandType Application -ErrorAction Stop | Out-Null
$Workspace = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Workspace)
$qwenputer = Join-Path $Workspace 'qwenputer'
$marker = Join-Path $Workspace '.qwenpu-bonsai-workspace'
if ((Test-Path -LiteralPath $qwenputer) -and !(Test-Path -LiteralPath $marker -PathType Leaf)) {
    throw "An existing qwenputer directory was found. Choose a dedicated workspace."
}
$tokenizer = Join-Path $qwenputer 'dist/qwen3.5-0.8b-base'
$tokenizerFile = Join-Path $tokenizer 'tokenizer.json'
if (!(Test-Path -LiteralPath $tokenizerFile -PathType Leaf)) {
    Get-Command hf -CommandType Application -ErrorAction Stop | Out-Null
}
New-Item -ItemType Directory -Force -Path $Workspace | Out-Null
Set-Content -LiteralPath $marker -Value 'qwenpu-bonsai setup workspace' -Encoding ascii

Initialize-Checkout $qwenputer 'https://github.com/divagr18/qwenputer.git' $QwenputerRev
$patch = Join-Path $PSScriptRoot 'patches/qwenputer-evaluate-order.patch'
& git -C $qwenputer apply --reverse --check $patch 2>$null
if ($LASTEXITCODE -ne 0) { Invoke-Checked git @('-C', $qwenputer, 'apply', $patch) }

$bench = Join-Path $qwenputer 'dist/local-bench'
$trace = Join-Path $bench 'qwenpu-trace'
$upstream = Join-Path $trace 'upstream'
Initialize-Checkout $upstream 'https://github.com/divagr18/QwenPU.git' $QwenPURev

Copy-Item -Recurse -Force -Path (Join-Path $PSScriptRoot 'overlay/*') -Destination $qwenputer
Copy-Item -Force -LiteralPath (Join-Path $PSScriptRoot 'tools/qwenpu-trace/probe.py') -Destination $trace

$story = Join-Path $bench 'bonsai-forward/stories260k'
New-Item -ItemType Directory -Force -Path $story | Out-Null
Copy-Item -Force -LiteralPath (Join-Path $PSScriptRoot 'guests/stories_vx_one.elf') -Destination (Join-Path $trace 'stories_vx.elf')
Copy-Item -Force -LiteralPath (Join-Path $PSScriptRoot 'guests/stories_vx_three.elf') -Destination (Join-Path $story 'stories_vx_three.elf')

if (!(Test-Path -LiteralPath $tokenizerFile -PathType Leaf)) {
    Invoke-Checked hf @('download', $TokenizerRepo, 'tokenizer.json', '--revision', $TokenizerRev, '--local-dir', $tokenizer)
}
if ((Get-FileHash -LiteralPath $tokenizerFile -Algorithm SHA256).Hash -ne $TokenizerHash) {
    throw "Tokenizer hash mismatch at $tokenizerFile. Move that file aside and rerun setup."
}

Write-Host "Workspace ready at $qwenputer"
Write-Host 'Follow docs/REPRODUCE.md to prepare the Python environment, Prism runtime, and Bonsai GGUF.'
