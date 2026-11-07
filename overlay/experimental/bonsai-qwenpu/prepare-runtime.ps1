param(
    [Parameter(Mandatory)][ValidateSet('forward','in-graph')][string]$Profile,
    [Parameter(Mandatory)][string]$InstalledRuntime,
    [string]$VisualStudioPath,
    [string]$Python,
    [string]$NvrtcDirectory,
    [int]$BuildJobs
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$workspace = (Resolve-Path -LiteralPath "$PSScriptRoot/../..").Path
$output = Join-Path $workspace "dist/local-bench/bonsai-$Profile"
$checkout = Join-Path $output 'llama.cpp'
$tag = 'prism-b9596-9fcaed7'
$revision = '9fcaed763ccda38ea81068ad9d7f991aaddca451'
$forward = $Profile -eq 'forward'
if (!$BuildJobs) { $BuildJobs = if ($forward) { 6 } else { 2 } }

function Invoke-NativeChecked($program, [string[]]$arguments) {
    & $program @arguments
    if ($LASTEXITCODE -ne 0) { throw "$program exited with status $LASTEXITCODE" }
}

function Resolve-File([string]$path) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required file missing at $path." }
    (Resolve-Path -LiteralPath $path).Path
}

$InstalledRuntime = (Resolve-Path -LiteralPath $InstalledRuntime).Path
foreach ($name in 'llama.dll','ggml.dll','ggml-base.dll','ggml-cuda.dll') {
    Resolve-File (Join-Path $InstalledRuntime $name) | Out-Null
}
Get-Command git -CommandType Application -ErrorAction Stop | Out-Null
if (!$Python) { $Python = Join-Path $workspace '.venv/Scripts/python.exe' }
$Python = Resolve-File $Python
if ($forward) {
    if (!$NvrtcDirectory) {
        $sitePackages = Invoke-NativeChecked $Python @('-c', "import sysconfig; print(sysconfig.get_path('platlib'))")
        $NvrtcDirectory = Join-Path $sitePackages 'torch/lib'
    }
    Resolve-File (Join-Path $NvrtcDirectory 'nvrtc64_120_0.dll') | Out-Null
    $NvrtcDirectory = (Resolve-Path -LiteralPath $NvrtcDirectory).Path
}

if (!$VisualStudioPath) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (!(Test-Path -LiteralPath $vswhere -PathType Leaf)) {
        throw 'Visual Studio Installer was not found. Install VS 2022 C++ build tools or pass -VisualStudioPath.'
    }
    $VisualStudioPath = Invoke-NativeChecked $vswhere @('-latest','-products','*','-version','[17.0,18.0)',
        '-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64','Microsoft.VisualStudio.Component.VC.CMake.Project',
        '-property','installationPath')
    if (!$VisualStudioPath) { throw 'No VS 2022 install with the x64 C++ and CMake tools was found.' }
}
$developerShell = Resolve-File (Join-Path $VisualStudioPath 'Common7/Tools/Launch-VsDevShell.ps1')
$cmakeRoot = Join-Path $VisualStudioPath 'Common7/IDE/CommonExtensions/Microsoft/CMake'
$cmake = Resolve-File (Join-Path $cmakeRoot 'CMake/bin/cmake.exe')
$ninja = Resolve-File (Join-Path $cmakeRoot 'Ninja/ninja.exe')

New-Item -ItemType Directory -Force -Path $output | Out-Null
if (!(Test-Path -LiteralPath $checkout)) {
    Invoke-NativeChecked git @('-c','advice.detachedHead=false','clone','--config','core.longpaths=true','--depth','1','--branch',$tag,
        'https://github.com/PrismML-Eng/llama.cpp.git',$checkout)
}
$root = Invoke-NativeChecked git @('-C',$checkout,'rev-parse','--show-toplevel')
$head = Invoke-NativeChecked git @('-C',$checkout,'rev-parse','HEAD')
if ((Resolve-Path -LiteralPath $root).Path -ne (Resolve-Path -LiteralPath $checkout).Path -or $head -ne $revision) {
    throw "The Prism checkout at $checkout must be at $revision."
}
if ($forward) {
    $patch = Join-Path $PSScriptRoot 'runtime-forward.patch'
    $sources = 'llama-qwenpu-forward.cpp','llama-qwenpu-forward.h','qwenpu-forward-cuda.cpp',
        'qwenpu-forward-cuda.h','qwenpu-forward-cuda-layout.h'
    $probe = 'probe_forward'
} else {
    $patch = Join-Path $PSScriptRoot 'runtime.patch'
    $sources = 'llama-qwenpu.cpp','qwenpu-circuit.h'
    $probe = 'probe'
}
& git -C $checkout apply --reverse --check $patch 2>$null
if ($LASTEXITCODE -ne 0) { Invoke-NativeChecked git @('-C',$checkout,'apply',$patch) }
foreach ($name in $sources) {
    Copy-Item -LiteralPath "$PSScriptRoot/$name" -Destination "$checkout/src"
}

& $developerShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
Get-Command cl.exe, dumpbin.exe, lib.exe -CommandType Application -ErrorAction Stop | Out-Null

$imports = Join-Path $output 'ggml-imports'
$candidate = Join-Path $output 'candidate'
New-Item -ItemType Directory -Force -Path $imports,$candidate | Out-Null
foreach ($name in 'ggml-base','ggml') {
    $dll = Join-Path $InstalledRuntime "$name.dll"
    $exports = Invoke-NativeChecked dumpbin.exe @('/nologo','/exports',$dll)
    $symbols = foreach ($line in $exports) {
        if ($line -match '^\s+\d+\s+[\dA-Fa-f]+\s+[\dA-Fa-f]+\s+(?<symbol>\w+)\s*$') {
            '    ' + $Matches.symbol
        }
    }
    if (!$symbols) { throw "No exports found in $dll." }
    $definition = Join-Path $imports "$name.def"
    @("LIBRARY $name", 'EXPORTS') + $symbols | Set-Content -LiteralPath $definition -Encoding ascii
    Invoke-NativeChecked lib.exe @('/nologo','/machine:x64',"/def:$definition","/out:$imports/$name.lib")
}
$include = "$checkout/ggml/include".Replace('\','/')
$importCmake = $imports.Replace('\','/')
@"
add_library(ggml::ggml INTERFACE IMPORTED)
set_target_properties(ggml::ggml PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "$include"
    INTERFACE_COMPILE_DEFINITIONS "GGML_SHARED"
    INTERFACE_LINK_LIBRARIES "$importCmake/ggml.lib;$importCmake/ggml-base.lib")
"@ | Set-Content -LiteralPath "$imports/ggmlConfig.cmake" -Encoding ascii
Get-ChildItem -LiteralPath $InstalledRuntime -Filter '*.dll' |
    Where-Object { $_.Name -match '^(ggml|cublas|cudart)' } |
    Copy-Item -Destination $candidate

$build = Join-Path $output 'build'
Invoke-NativeChecked $cmake @('-S',$checkout,'-B',$build,'-G','Ninja',"-DCMAKE_MAKE_PROGRAM=$ninja",
    '-DCMAKE_BUILD_TYPE=Release','-DBUILD_SHARED_LIBS=ON','-DLLAMA_USE_SYSTEM_GGML=ON',
    "-Dggml_DIR=$imports",'-DLLAMA_BUILD_COMMON=OFF','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_TOOLS=OFF',
    '-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_SERVER=OFF','-DLLAMA_BUILD_APP=OFF',
    '-DLLAMA_BUILD_HTML=OFF')
Invoke-NativeChecked $cmake @('--build',$build,'--target','llama','--parallel',"$BuildJobs")
Copy-Item -LiteralPath "$build/bin/llama.dll" -Destination $candidate

Push-Location -LiteralPath $output
try {
    Invoke-NativeChecked cl.exe @('/nologo','/std:c++17','/EHsc','/O2','/utf-8',
        "/I$checkout/include","/I$checkout/ggml/include", "$PSScriptRoot/$probe.cpp","/Fe:$probe.exe")
    if ($forward) {
        Invoke-NativeChecked $Python @('-B', "$PSScriptRoot/compile-forward-cuda.py", "$candidate/qwenpu-forward-cuda.cubin",
            '--nvrtc-directory', $NvrtcDirectory)
    } else {
        Invoke-NativeChecked $Python @('-B', "$PSScriptRoot/export_fixture.py", $output)
    }
} finally {
    Pop-Location
}
