param(
    [ValidateSet('rocm', 'cpu')][string]$Backend = 'rocm',
    [ValidateRange(1024,65535)][int]$Port = 8811
)
$ErrorActionPreference = 'Stop'
$recyclableRoot = Split-Path $PSScriptRoot -Parent
$recyclablePython = Join-Path $recyclableRoot $(if ($Backend -eq 'rocm') { '.venv-rocm/Scripts/python.exe' } else { 'local-only/yoloe-deployment/venv/Scripts/python.exe' })
$recyclableDevice = if ($Backend -eq 'rocm') { 'cuda:0' } else { 'cpu' }
$env:PYTHONUTF8 = '1'
Push-Location (Join-Path $recyclableRoot 'pynq_km1_vision')
try {
    & $recyclablePython -u -m src.yoloe_recyclable_monitor --port $Port --device $recyclableDevice
    if ($LASTEXITCODE -ne 0) { throw 'TACO prompted monitor exited with an error' }
} finally { Pop-Location }
