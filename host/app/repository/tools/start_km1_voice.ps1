param(
    [ValidateSet('Listen','Preview','Execute')][string]$Mode='Listen',
    [string]$Port='auto',
    [string]$Binding='',
    [switch]$Ready,
    [ValidateRange(1,3600)][int]$Seconds=300,
    [string]$DashboardUrl='http://127.0.0.1:8792'
)
$ErrorActionPreference='Stop'
if ($Mode -eq 'Execute' -and -not $Ready) { throw 'Execute requires explicit current-scene Ready; one attempt only.' }
$voiceRepo=Split-Path $PSScriptRoot -Parent
$voicePython=Join-Path $voiceRepo '.venv/Scripts/python.exe'
if (-not $Binding) {
    $voicePointerPath=Join-Path $voiceRepo 'local-only/station-grasp-integration/active.json'
    $voicePointer=Get-Content -LiteralPath $voicePointerPath -Raw | ConvertFrom-Json
    $Binding=[IO.Path]::GetFullPath((Join-Path (Split-Path $voicePointerPath -Parent) $voicePointer.path))
    if ((Get-FileHash -LiteralPath $Binding -Algorithm SHA256).Hash.ToLowerInvariant() -ne $voicePointer.sha256) {
        throw 'Station binding changed; inspect the current integration record.'
    }
}
if (-not [IO.Path]::IsPathRooted($Binding)) { $Binding=Join-Path $voiceRepo $Binding }
$voiceLog=Join-Path $voiceRepo ('local-only/sorting-voice-20261029/session-'+[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()+'.jsonl')
$voiceArgs=@('-u','-m','src.km1_voice_run','--port',$Port,'--binding',$Binding,'--seconds',"$Seconds",'--log',$voiceLog)
if ($DashboardUrl) { $voiceArgs+='--dashboard-url'; $voiceArgs+=$DashboardUrl }
if ($Mode -eq 'Listen') { $voiceArgs+='--listen-only' }
if ($Mode -eq 'Execute') { $voiceArgs+='--execute'; $voiceArgs+='--ready' }
Push-Location (Join-Path $voiceRepo 'pynq_km1_vision')
try {
    & $voicePython @voiceArgs
    if ($LASTEXITCODE -ne 0) { throw 'Hardware voice session ended with an error.' }
} finally { Pop-Location }
