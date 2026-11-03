param(
    [Parameter(Mandatory = $true)][string]$SessionRoot,
    [string]$Output = '',
    [ValidateRange(1024, 65535)][int]$Port = 8782
)
$ErrorActionPreference = 'Stop'
$calibrationRepo = Split-Path $PSScriptRoot -Parent
$calibrationProject = Join-Path $calibrationRepo 'pynq_km1_vision'
$calibrationPython = Join-Path $calibrationRepo '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $calibrationPython)) { throw 'Missing project Python environment' }
if (-not [IO.Path]::IsPathRooted($SessionRoot)) { $SessionRoot = Join-Path $calibrationRepo $SessionRoot }
if (-not (Test-Path -LiteralPath $SessionRoot -PathType Container)) { throw 'Commissioning session folder does not exist' }
if (-not $Output) { $Output = Join-Path $calibrationRepo ('local-only/station-guided-' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
elseif (-not [IO.Path]::IsPathRooted($Output)) { $Output = Join-Path $calibrationRepo $Output }
if (Test-Path -LiteralPath $Output) { throw 'Use a new output folder; existing captures are never overwritten' }
New-Item -ItemType Directory -Path $Output | Out-Null
$calibrationHistory = Join-Path $Output 'history'
$calibrationCapture = Join-Path $Output 'capture'
$calibrationStation = Join-Path $calibrationProject 'config/station_markers.design.json'
$env:PYTHONUTF8 = '1'
Push-Location $calibrationProject
try {
    & $calibrationPython -m src.station_history --session $SessionRoot --output $calibrationHistory
    if ($LASTEXITCODE -ne 0) { throw 'Historical evidence did not pass; inspect the original files' }
    Write-Host "Open http://127.0.0.1:$Port/ . Camera-only; uses the existing bridge on port 8780."
    & $calibrationPython -u -m src.station_guided --history $calibrationHistory --station $calibrationStation --output $calibrationCapture --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'Calibration assistant stopped; existing captures remain saved' }
} finally { Pop-Location }
