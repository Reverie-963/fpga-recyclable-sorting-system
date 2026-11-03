param([ValidateRange(1024,65535)][int]$Port=8812)
$ErrorActionPreference='Stop'
$boardConsoleRoot=Split-Path $PSScriptRoot -Parent
$boardConsoleConfig=Join-Path $boardConsoleRoot 'local-only/board-control-integration/active.json'
if (-not (Test-Path -LiteralPath $boardConsoleConfig)) { throw 'Start the board preview service first.' }
$env:PYTHONUTF8='1'
Push-Location (Join-Path $boardConsoleRoot 'pynq_km1_vision')
try {
    & (Join-Path $boardConsoleRoot '.venv/Scripts/python.exe') -u -m src.board_prompted_console --port $Port --board-controller-config $boardConsoleConfig
    if ($LASTEXITCODE -ne 0) { throw 'Board planning console exited with an error' }
} finally { Pop-Location }
