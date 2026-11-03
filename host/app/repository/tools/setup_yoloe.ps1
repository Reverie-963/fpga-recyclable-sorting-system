param([switch]$StartPreview, [string]$References)
$ErrorActionPreference = 'Stop'
$yoloeRepoRoot = Split-Path $PSScriptRoot -Parent
$yoloeRoot = Join-Path $yoloeRepoRoot 'local-only/yoloe-deployment'
$yoloePython = Join-Path $yoloeRoot 'venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $yoloePython)) {
    py -3.11 -m venv (Join-Path $yoloeRoot 'venv')
    if ($LASTEXITCODE -ne 0) { throw 'YOLOE Python 3.11 environment creation failed' }
}
& $yoloePython -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'YOLOE CPU runtime installation failed' }
& $yoloePython -m pip install -r (Join-Path $yoloeRepoRoot 'pynq_km1_vision/requirements_yoloe_cpu.txt') --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'YOLOE dependencies failed' }
& $yoloePython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'YOLOE dependency consistency check failed' }
$yoloePrepareArgs = @((Join-Path $yoloeRepoRoot 'tools/prepare_yoloe.py'))
if ($References) { $yoloePrepareArgs += @('--references', $References) }
& $yoloePython @yoloePrepareArgs
if ($LASTEXITCODE -ne 0) { throw 'YOLOE asset preparation failed' }
Write-Output 'YOLOE prompted weights are ready. Quality must be evaluated on real images.'
if ($StartPreview) {
    & $yoloePython (Join-Path $yoloeRepoRoot 'pynq_km1_vision/src/yoloe_preview_server.py')
    if ($LASTEXITCODE -ne 0) { throw 'YOLOE preview failed' }
}
