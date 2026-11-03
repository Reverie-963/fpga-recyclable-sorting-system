param([switch]$TestOnly)
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path $PSScriptRoot -Parent
$LocalFolder = Join-Path $RepoRoot 'local-only/qq-bot'
$ConfigPath = Join-Path $LocalFolder 'credentials.clixml'
$Python = Join-Path $RepoRoot '.venv-qq/Scripts/python.exe'
New-Item -ItemType Directory -Force $LocalFolder | Out-Null
if (-not (Test-Path -LiteralPath $Python)) {
    & py -3.11 -m venv (Join-Path $RepoRoot '.venv-qq')
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create QQ setup environment.' }
}
if (-not $TestOnly) {
    & $Python -m pip install 'websocket-client>=1.8,<2'
    if ($LASTEXITCODE -ne 0) { throw 'Cannot install the binding dependency.' }
    Write-Host 'Target group: BlackShore_41784_FPGA'
    Write-Host 'Add your official bot to this group and allow proactive messages first.'
    $AppId = (Read-Host 'QQ bot AppID').Trim()
    if ($AppId -notmatch '^\d+$') { throw 'AppID must be numeric.' }
    $Secret = Read-Host 'QQ bot AppSecret (hidden input)' -AsSecureString
    if ($Secret.Length -eq 0) { throw 'AppSecret is required.' }
    $Config = [pscustomobject]@{ AppId = $AppId; Secret = $Secret; GroupOpenId = '' }
} else {
    $Config = Import-Clixml -LiteralPath $ConfigPath
}
$PreviousId = $env:QQ_BOT_APP_ID
$PreviousSecret = $env:QQ_BOT_APP_SECRET
$PreviousGroup = $env:QQ_BOT_GROUP_OPENID
try {
    $env:QQ_BOT_APP_ID = $Config.AppId
    $Credential = New-Object System.Management.Automation.PSCredential('qqbot', $Config.Secret)
    $env:QQ_BOT_APP_SECRET = $Credential.GetNetworkCredential().Password
    if (-not $TestOnly) {
        $BindingPath = Join-Path $LocalFolder 'group.local.json'
        & $Python (Join-Path $PSScriptRoot 'qq_bind_group.py') --output $BindingPath
        if ($LASTEXITCODE -ne 0) { throw 'Group binding failed; configuration was not activated.' }
        $Config.GroupOpenId = (Get-Content -LiteralPath $BindingPath -Raw | ConvertFrom-Json).group_openid
        # SecureString is encrypted by Windows DPAPI for this Windows user/machine.
        $Config | Export-Clixml -LiteralPath $ConfigPath
    }
    $env:QQ_BOT_GROUP_OPENID = $Config.GroupOpenId
    & $Python (Join-Path $PSScriptRoot 'qq_merge_notify.py') --test-message
    if ($LASTEXITCODE -ne 0) { throw 'Test notification failed. Check proactive-message permission.' }
    @{ accepted = $true; checkedAt = (Get-Date).ToUniversalTime().ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $LocalFolder 'test-result.local.json') -Encoding UTF8
    Write-Host 'Test accepted by QQ. Please verify it in the target group.'
    Write-Host 'GitHub Actions secrets required: QQ_BOT_APP_ID, QQ_BOT_APP_SECRET, QQ_BOT_GROUP_OPENID'
    Write-Host ('QQ_BOT_APP_ID = ' + $Config.AppId)
    Write-Host ('QQ_BOT_GROUP_OPENID = ' + $Config.GroupOpenId)
    Write-Host 'AppSecret remains hidden; paste it directly from the QQ developer console into GitHub Secrets.'
    Write-Host 'https://github.com/Reverie-963/FPGA/settings/secrets/actions'
    Write-Host 'After testing, set repository variable QQ_NOTIFY_ENABLED to true.'
} finally {
    $env:QQ_BOT_APP_ID = $PreviousId
    $env:QQ_BOT_APP_SECRET = $PreviousSecret
    $env:QQ_BOT_GROUP_OPENID = $PreviousGroup
    $Credential = $null
}
