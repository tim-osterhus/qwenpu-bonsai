<#
.SYNOPSIS
Build the forward probe and its SM 6.1 CUDA module against the pinned Prism runtime.
#>
param(
    [Parameter(Mandatory)][string]$InstalledRuntime,
    [string]$VisualStudioPath,
    [string]$Python,
    [string]$NvrtcDirectory,
    [ValidateRange(1, 64)][int]$BuildJobs
)

& "$PSScriptRoot/prepare-runtime.ps1" -Profile 'forward' @PSBoundParameters
