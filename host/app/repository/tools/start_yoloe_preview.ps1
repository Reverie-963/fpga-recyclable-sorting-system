param(
    [ValidateSet('rocm', 'cpu')][string]$Backend = 'rocm',
    [switch]$ImagesOnly,
    [ValidateRange(1024, 65535)][int]$Port = 8778,
    [string]$CameraUrl = 'http://127.0.0.1:8780/frame.jpg'
)
$ErrorActionPreference = 'Stop'
$yoloeRepoRoot = Split-Path $PSScriptRoot -Parent
$yoloePythonRelative = if ($Backend -eq 'rocm') { '.venv-rocm/Scripts/python.exe' } else { 'local-only/yoloe-deployment/venv/Scripts/python.exe' }
$yoloeRuntimePython = Join-Path $yoloeRepoRoot $yoloePythonRelative
if (-not (Test-Path -LiteralPath $yoloeRuntimePython)) {
    throw "Missing $Backend runtime: $yoloeRuntimePython. See pynq_km1_vision/docs/yoloe_realtime_rocm.md."
}
$yoloeLaunchArgs = @('-u', (Join-Path $yoloeRepoRoot 'pynq_km1_vision/src/yoloe_preview_server.py'), '--port', $Port)
if ($Backend -eq 'rocm') { $yoloeLaunchArgs += @('--device', 'cuda:0', '--half') }
if (-not $ImagesOnly) { $yoloeLaunchArgs += @('--camera-url', $CameraUrl) }
$env:PYTHONUTF8 = '1'
& $yoloeRuntimePython @yoloeLaunchArgs
if ($LASTEXITCODE -ne 0) { throw "YOLOE $Backend monitor exited with an error" }
