<#
.SYNOPSIS
Build the earlier shared-context probe against the pinned Prism runtime.
#>
param(
    [Parameter(Mandatory)][string]$InstalledRuntime,
    [string]$VisualStudioPath,
    [string]$Python,
    [ValidateRange(1, 64)][int]$BuildJobs
)

& "$PSScriptRoot/prepare-runtime.ps1" -Profile 'in-graph' @PSBoundParameters
