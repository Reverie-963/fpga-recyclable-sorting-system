param([switch]$StartPreview)
$ErrorActionPreference = 'Stop'
$tacoRepoRoot = Split-Path $PSScriptRoot -Parent
$tacoRoot = Join-Path $tacoRepoRoot 'local-only/taco-deployment'
New-Item -ItemType Directory -Force -Path $tacoRoot | Out-Null

function Get-TacoAsset($Url, $Path, $Hash) {
    if (-not (Test-Path -LiteralPath $Path)) { Invoke-WebRequest -Uri $Url -OutFile $Path }
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Hash) {
        throw "Hash mismatch; preserve and inspect file: $Path"
    }
}

$tacoPythonZip = Join-Path $tacoRoot 'python37.zip'
Get-TacoAsset 'https://www.python.org/ftp/python/3.7.8/python-3.7.8-embed-amd64.zip' $tacoPythonZip '88a033a378437f0d000cfaa2752f4ca860d06b06f17e58a7f31be44b12787a35'
$tacoPythonDir = Join-Path $tacoRoot 'python37'
if (-not (Test-Path -LiteralPath $tacoPythonDir)) {
    Expand-Archive -LiteralPath $tacoPythonZip -DestinationPath $tacoPythonDir
    Set-Content -LiteralPath (Join-Path $tacoPythonDir 'python37._pth') -Encoding ascii -Value @('python37.zip', '.', 'Lib/site-packages', 'import site')
}
$tacoPython = Join-Path $tacoPythonDir 'python.exe'
if (-not (Test-Path -LiteralPath (Join-Path $tacoPythonDir 'Lib/site-packages/pip/__main__.py'))) {
    $tacoBootstrap = Join-Path $tacoRoot 'get-pip.py'
    Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/pip/3.7/get-pip.py' -OutFile $tacoBootstrap
    & $tacoPython $tacoBootstrap
    if ($LASTEXITCODE -ne 0) { throw 'Portable pip setup failed' }
}
& $tacoPython -m pip install --only-binary=tensorflow,numpy,scipy,scikit-image,h5py,pycocotools,matplotlib,opencv-python-headless -r (Join-Path $tacoRepoRoot 'pynq_km1_vision/requirements_taco_legacy.txt')
if ($LASTEXITCODE -ne 0) { throw 'Legacy dependency installation failed' }
$tacoSource = Join-Path $tacoRoot 'TACO'
if (-not (Test-Path -LiteralPath $tacoSource)) {
    git clone https://github.com/pedropro/TACO.git $tacoSource
    if ($LASTEXITCODE -ne 0) { throw 'TACO clone failed' }
    git -C $tacoSource checkout --detach 70479ad9adc3da971a8ae75ac97eb1026910eec7
    if ($LASTEXITCODE -ne 0) { throw 'TACO pinned checkout failed' }
}
$tacoActualCommit = git -C $tacoSource rev-parse HEAD
if ($tacoActualCommit -ne '70479ad9adc3da971a8ae75ac97eb1026910eec7') { throw 'Existing TACO checkout is a different version; not replacing it' }
$tacoWeightZip = Join-Path $tacoRoot 'taco_10_0.zip'
Get-TacoAsset 'https://github.com/pedropro/TACO/releases/download/1.0/taco_10_0.zip' $tacoWeightZip 'd61a58aed158adb30bac5e2f7c59beb67aa82c6eb09ef52d2f30a5e2325e4a38'
$tacoWeightsDir = Join-Path $tacoRoot 'weights-fold0'
if (-not (Test-Path -LiteralPath $tacoWeightsDir)) {
    Expand-Archive -LiteralPath $tacoWeightZip -DestinationPath $tacoWeightsDir
}
if ((Get-FileHash -LiteralPath (Join-Path $tacoWeightsDir 'taco_10_0/mask_rcnn_taco_0100.h5') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'd79517accf4e4a0858db5fbe02df3c1f7572def76abf8ecf981398d67a4e73f1') { throw 'Model SHA256 mismatch' }
& $tacoPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed' }
Write-Output 'TACO runtime and original weights are ready. Model loading and real-image inference must be checked separately.'
if ($StartPreview) {
    & $tacoPython (Join-Path $tacoRepoRoot 'pynq_km1_vision/src/taco_preview_server.py')
    if ($LASTEXITCODE -ne 0) { throw 'TACO preview failed' }
}
