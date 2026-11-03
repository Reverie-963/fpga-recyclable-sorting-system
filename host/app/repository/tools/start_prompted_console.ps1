param([ValidateRange(1024,65535)][int]$Port=8810)
$ErrorActionPreference='Stop'
$consoleRoot=Split-Path $PSScriptRoot -Parent
$env:PYTHONUTF8='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
Push-Location (Join-Path $consoleRoot 'pynq_km1_vision')
try {
    & (Join-Path $consoleRoot '.venv/Scripts/python.exe') -u -m src.sorting_prompted_console --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'Prompted console exited with an error' }
} finally { Pop-Location }
