param(
    [ValidateSet('rocm', 'cpu')][string]$Backend = 'rocm',
    [ValidateRange(1024,65535)][int]$Port = 8811,
    [string]$CameraUrl = 'http://127.0.0.1:8780/frame.jpg'
)
$ErrorActionPreference = 'Stop'
$officialRoot = Split-Path $PSScriptRoot -Parent
$officialPython = Join-Path $officialRoot $(if ($Backend -eq 'rocm') { '.venv-rocm/Scripts/python.exe' } else { 'local-only/yoloe-deployment/venv/Scripts/python.exe' })
$officialDevice = if ($Backend -eq 'rocm') { 'cuda:0' } else { 'cpu' }
$env:PYTHONUTF8 = '1'
& $officialPython -u (Join-Path $officialRoot 'pynq_km1_vision/src/yoloe_official_monitor.py') --port $Port --device $officialDevice --camera-url $CameraUrl
if ($LASTEXITCODE -ne 0) { throw 'Official YOLOE monitor exited with an error' }
