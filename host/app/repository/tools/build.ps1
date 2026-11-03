param(
    [string]$ToolRoot = $env:KM1_TOOL_ROOT,
    [string]$BoardRepo,
    [string]$EncoderXdc,
    [ValidateSet('segment', 'segment_open')][string]$Kernel = 'segment_open',
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$RepositoryRoot = Split-Path $PSScriptRoot -Parent
$ProjectRoot = Join-Path $RepositoryRoot 'pynq_km1_vision'
if (-not $BoardRepo) { $BoardRepo = Join-Path $RepositoryRoot 'hardware/board_files' }
if (-not $EncoderXdc) { $EncoderXdc = Join-Path $ProjectRoot 'build/encoder_sen0230_pmodb.xdc' }
$Arguments = @{ Kernel = $Kernel; BoardRepo = $BoardRepo; EncoderXdc = $EncoderXdc }
if ($ToolRoot) { $Arguments.ToolRoot = $ToolRoot }
if ($CheckOnly) { $Arguments.CheckOnly = $true }
Push-Location $ProjectRoot
try {
    & (Join-Path $ProjectRoot 'tools/build_windows.ps1') @Arguments
} finally {
    Pop-Location
}
