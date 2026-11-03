param(
    [ValidateRange(1024, 65535)][int]$Port = 8778,
    [string]$CameraUrl = 'http://127.0.0.1:8780/frame.jpg',
    [switch]$Fp32
)
$ErrorActionPreference = 'Stop'
$monitorRepoRoot = Split-Path $PSScriptRoot -Parent
$monitorPython = Join-Path $monitorRepoRoot '.venv-rocm/Scripts/python.exe'
$monitorArgs = @('-u', '-m', 'src.yolov8s_preview_server', '--port', $Port, '--camera-url', $CameraUrl)
if ($Fp32) { $monitorArgs += '--fp32' }
$env:PYTHONUTF8 = '1'
Push-Location (Join-Path $monitorRepoRoot 'pynq_km1_vision')
try {
    & $monitorPython @monitorArgs
    if ($LASTEXITCODE -ne 0) { throw 'YOLOv8s monitor exited with an error' }
} finally { Pop-Location }
