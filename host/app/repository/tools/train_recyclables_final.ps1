[CmdletBinding()]
param(
    [string]$Dataset,
    [string]$Weights,
    [string]$Output,
    [string]$Python,
    [string]$Device = "0",
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$taskRepo = Split-Path -Parent $PSScriptRoot
$taskProject = Join-Path $taskRepo "pynq_km1_vision"
if (-not $Dataset) { $Dataset = Join-Path $taskRepo "local-only/recyclables-final-16-v1" }
if (-not $Weights) { $Weights = Join-Path $taskProject "yolo11n-seg.pt" }
if (-not $Python) { $Python = Join-Path $taskRepo ".venv-rocm/Scripts/python.exe" }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Training Python is missing. Pass -Python with your existing ROCm environment interpreter."
}
$taskPython = [System.IO.Path]::GetFullPath($Python)
$taskDataset = [System.IO.Path]::GetFullPath($Dataset)
$taskWeights = [System.IO.Path]::GetFullPath($Weights)
$taskArgs = @("-m", "src.train_recyclables_final", "--dataset", $taskDataset,
              "--weights", $taskWeights, "--device", $Device)
if ($Output) { $taskArgs += @("--output", [System.IO.Path]::GetFullPath($Output)) }
if ($CheckOnly) { $taskArgs += "--check-only" }
Push-Location -LiteralPath $taskProject
try {
    & $taskPython @taskArgs
    $taskCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $taskCode
