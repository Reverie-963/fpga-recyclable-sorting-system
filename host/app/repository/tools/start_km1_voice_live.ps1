param([string]$Port='auto',[switch]$Foreground)
$ErrorActionPreference='Stop'
$voiceLiveRoot=Split-Path $PSScriptRoot -Parent
$voiceLivePython=Join-Path $voiceLiveRoot '.venv-rocm/Scripts/python.exe'
$voiceLiveLogs=Join-Path $voiceLiveRoot 'local-only/sorting-voice-live'
New-Item -ItemType Directory -Force -Path $voiceLiveLogs | Out-Null
$voiceLiveStamp=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
$voiceLiveLog=Join-Path $voiceLiveLogs ('session-'+$voiceLiveStamp+'.jsonl')
$voiceLiveArgs=@('-u','-m','src.km1_voice_live','--root',$voiceLiveRoot,'--port',$Port,'--log',$voiceLiveLog)
if ($Foreground) {
    Push-Location (Join-Path $voiceLiveRoot 'pynq_km1_vision')
    try {
        & $voiceLivePython @voiceLiveArgs
        if ($LASTEXITCODE -ne 0) { throw 'Voice interface service exited with an error' }
    } finally { Pop-Location }
} else {
    $voiceLiveProcess=Start-Process -FilePath $voiceLivePython -ArgumentList $voiceLiveArgs -WorkingDirectory (Join-Path $voiceLiveRoot 'pynq_km1_vision') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $voiceLiveLogs ('out-'+$voiceLiveStamp+'.log')) -RedirectStandardError (Join-Path $voiceLiveLogs ('error-'+$voiceLiveStamp+'.log')) -PassThru
    [ordered]@{ mode='LiveControllerVoice';parent_pid=$voiceLiveProcess.Id;log=$voiceLiveLog;resident=$true;direct_commands=$true;page_confirmation_required=$false } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $voiceLiveLogs 'active.json') -Encoding utf8
    Write-Output 'Voice interface service started; waiting for recognized commands.'
}
