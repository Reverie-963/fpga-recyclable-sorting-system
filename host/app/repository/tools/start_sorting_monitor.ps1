param([ValidateRange(1024,65535)][int]$Port=8779,[switch]$VoiceResident,[string]$VoicePort='auto',[switch]$VoiceDirect)
$ErrorActionPreference='Stop'
if ($VoiceDirect -and -not $VoiceResident) { throw 'VoiceDirect requires VoiceResident.' }
$dashboardRoot=Split-Path $PSScriptRoot -Parent
$dashboardPython=Join-Path $dashboardRoot '.venv-rocm/Scripts/python.exe'
$env:PYTHONUTF8='1'
Push-Location (Join-Path $dashboardRoot 'pynq_km1_vision')
try {
    $dashboardArgs=@('-u','-m','src.sorting_monitor_server','--port',"$Port")
    if ($VoiceResident) { $dashboardArgs+='--voice-resident'; $dashboardArgs+='--voice-port'; $dashboardArgs+=$VoicePort }
    if ($VoiceDirect) { $dashboardArgs+='--voice-direct' }
    & $dashboardPython @dashboardArgs
    if ($LASTEXITCODE -ne 0) { throw 'Sorting dashboard exited with an error' }
} finally { Pop-Location }
