param([switch]$Apply, [switch]$ResumeDisplayOnly)
$ErrorActionPreference = 'Stop'
$gpuConsoleRoot = Split-Path $PSScriptRoot -Parent
$gpuConsoleCatalog = Join-Path $gpuConsoleRoot 'local-only/model-switch-console/models.json'
$gpuConsoleDeployment = Join-Path $gpuConsoleRoot 'local-only/yoloe11m-recyclables26/console-deployment-20261030'
function Read-GpuConsole([string]$Path) {
    # Status GETs are safe to retry; never replay a motion/control POST.
    for ($gpuReadAttempt = 0; $gpuReadAttempt -lt 3; $gpuReadAttempt++) {
        try { return Invoke-RestMethod ('http://127.0.0.1:8810' + $Path) -TimeoutSec 10 }
        catch {
            if ($gpuReadAttempt -eq 2) { throw }
            Start-Sleep -Milliseconds 300
        }
    }
}
function Find-GpuConsoleProcess([int]$Port, [string]$Pattern) {
    $gpuListeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop |
        Select-Object -ExpandProperty OwningProcess -Unique)
    if ($gpuListeners.Count -ne 1) { throw "Port $Port must have exactly one listener" }
    $gpuProcess = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $gpuListeners[0])
    if ($gpuProcess.Name -ne 'python.exe' -or $gpuProcess.CommandLine -notmatch $Pattern) {
        throw "Port $Port is not the expected project service"
    }
    $gpuParent = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $gpuProcess.ParentProcessId)
    if ($gpuParent -and $gpuParent.Name -eq 'python.exe' -and $gpuParent.CommandLine -match $Pattern) {
        return [int]$gpuParent.ProcessId
    }
    return [int]$gpuProcess.ProcessId
}
# The complete workbench now owns selection. Do not recreate a second manager
# on its retired preview address or overwrite the running integrated catalog.
try { $gpuIntegratedState = Read-GpuConsole '/model-switch/status' }
catch { $gpuIntegratedState = $null }
if ($gpuIntegratedState.integrated_workbench) {
    throw 'Model selection is integrated at http://127.0.0.1:8810/. Use tools/restart_native_board_proxy.py for its PC display process.'
}
$gpuDetectorBefore = Invoke-RestMethod 'http://127.0.0.1:8824/health' -TimeoutSec 8
if ($ResumeDisplayOnly) {
    if (-not $gpuDetectorBefore.ready -or $gpuDetectorBefore.execution.device -ne 'cuda:0') {
        throw 'Display-only resume requires an already running GPU detector; detector will not be restarted'
    }
    # Do not infer an idle physical state when the separate PS host is absent.
    try { $gpuConsoleState = Read-GpuConsole '/operation' }
    catch { $gpuConsoleState = [pscustomobject]@{ operation_id = $null; compute_location = 'unavailable'; busy = $null } }
} else {
    $gpuConsoleState = Read-GpuConsole '/operation'
    if ($gpuConsoleState.busy -or $gpuConsoleState.compute_location -ne 'pynq_ps') {
        throw 'PYNQ must be reachable with no active preview, standby or motion'
    }
}
$gpuDetectorPID = Find-GpuConsoleProcess 8824 'src\.yoloe_trained26_monitor.*--port[ =]+8824'
$gpuSelectorPID = Find-GpuConsoleProcess 8830 'switchable_console\.py.*--port[ =]+8830'
$gpuConsolePlan = [ordered]@{ apply = [bool]$Apply; detector_process = $gpuDetectorPID;
    selector_process = $gpuSelectorPID; current_device = $gpuDetectorBefore.execution.device;
    requested_device = 'cuda:0'; original_operation_id = $gpuConsoleState.operation_id;
    pynq_control_restart = $false; training_restart = $false; mechanical_command = $false }
if (-not $Apply) { $gpuConsolePlan | ConvertTo-Json; return }
# Apply only to the two identified services after checking station quiescence.
if (-not $ResumeDisplayOnly) {
    $gpuConsoleNow = Read-GpuConsole '/operation'
    if ($gpuConsoleNow.busy -or $gpuConsoleNow.operation_id -ne $gpuConsoleState.operation_id) {
        throw 'PYNQ operation changed; restart deferred'
    }
}
$gpuConsoleFolder = Join-Path $gpuConsoleRoot ('local-only/model-switch-console/operator-gpu-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $gpuConsoleFolder | Out-Null
$gpuDetectorBefore | ConvertTo-Json -Depth 12 | Set-Content (Join-Path $gpuConsoleFolder 'before-health.json') -Encoding utf8
foreach ($gpuLogName in @('detector.stdout.log', 'detector.stderr.log')) {
    $gpuOldLog = Join-Path $gpuConsoleDeployment $gpuLogName
    if (Test-Path -LiteralPath $gpuOldLog) { Copy-Item -LiteralPath $gpuOldLog -Destination (Join-Path $gpuConsoleFolder $gpuLogName) }
}
if ($gpuDetectorBefore.execution.device -ne 'cuda:0') {
    & taskkill /PID $gpuDetectorPID /T /F
    if ($LASTEXITCODE -ne 0) { throw 'Original detector did not stop' }
    & (Join-Path $gpuConsoleRoot 'tools/start_trained26_console.ps1') -Deployment $gpuConsoleDeployment -Backend rocm -DetectorOnly
}
$gpuDeadline = [DateTime]::UtcNow.AddSeconds(60)
$gpuDetectorAfter = $null
while ([DateTime]::UtcNow -lt $gpuDeadline) {
    try {
        $gpuDetectorAfter = Invoke-RestMethod 'http://127.0.0.1:8824/health' -TimeoutSec 3
        if ($gpuDetectorAfter.ready -and $gpuDetectorAfter.camera_connected -and
            $gpuDetectorAfter.execution.device -eq 'cuda:0') { break }
    } catch { $gpuDetectorAfter = $null }
    Start-Sleep -Milliseconds 300
}
if (-not $gpuDetectorAfter -or -not $gpuDetectorAfter.camera_connected -or $gpuDetectorAfter.execution.device -ne 'cuda:0' -or
    $gpuDetectorAfter.model_sha256 -ne $gpuDetectorBefore.model_sha256 -or
    $gpuDetectorAfter.settings.settings_sha256 -ne $gpuDetectorBefore.settings.settings_sha256) {
    throw 'GPU identity not verified; no silent CPU fallback or automatic motion'
}
$gpuCheck = Invoke-RestMethod 'http://127.0.0.1:8824/detect' -Method Post -ContentType application/json -Body '{"source":"camera","control_only":true}' -TimeoutSec 45
if ($gpuCheck.execution.device -ne 'cuda:0' -or $gpuCheck.model_sha256 -ne $gpuDetectorBefore.model_sha256) {
    throw 'Real camera result did not confirm GPU model'
}
Invoke-RestMethod 'http://127.0.0.1:8824/configure' -Method Post -ContentType application/json -Body '{"enabled":true}' -TimeoutSec 8 | Out-Null
& (Join-Path $gpuConsoleRoot '.venv/Scripts/python.exe') (Join-Path $gpuConsoleRoot 'tools/prepare_console_models.py') --device cuda:0 --output $gpuConsoleCatalog
if ($LASTEXITCODE -ne 0) { throw 'GPU model registration failed' }
& taskkill /PID $gpuSelectorPID /T /F
if ($LASTEXITCODE -ne 0) { throw 'Original selector did not stop' }
$gpuSelectorArgs = @('-u', 'tools/switchable_console.py', '--catalog', 'local-only/model-switch-console/models.json', '--port', '8830')
$gpuSelectorNew = Start-Process -FilePath (Join-Path $gpuConsoleRoot '.venv/Scripts/python.exe') -ArgumentList $gpuSelectorArgs `
    -WorkingDirectory $gpuConsoleRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $gpuConsoleFolder 'console.stdout.log') `
    -RedirectStandardError (Join-Path $gpuConsoleFolder 'console.stderr.log')
$gpuSelectorDeadline = [DateTime]::UtcNow.AddSeconds(25)
$gpuSelectorHealth = $null
while ([DateTime]::UtcNow -lt $gpuSelectorDeadline) {
    try {
        $gpuSelectorHealth = Invoke-RestMethod 'http://127.0.0.1:8830/health' -TimeoutSec 3
        if ($gpuSelectorHealth.execution.device -eq 'cuda:0' -and
            $gpuSelectorHealth.model_sha256 -eq $gpuDetectorBefore.model_sha256) { break }
    } catch { $gpuSelectorHealth = $null }
    Start-Sleep -Milliseconds 300
}
if (-not $gpuSelectorHealth -or $gpuSelectorHealth.execution.device -ne 'cuda:0' -or
    $gpuSelectorHealth.model_sha256 -ne $gpuDetectorBefore.model_sha256) {
    throw 'New model selector did not verify its GPU service; inspect local logs'
}
$gpuDetectorAfter | ConvertTo-Json -Depth 12 | Set-Content (Join-Path $gpuConsoleFolder 'gpu-health.json') -Encoding utf8
[ordered]@{ status = 'gpu_detector_and_selector_verified'; launcher_pid = $gpuSelectorNew.Id;
    device = $gpuCheck.execution.device; precision = $gpuCheck.execution.precision;
    model_sha256 = $gpuCheck.model_sha256; physical_motion = $false;
    display_only_resume = [bool]$ResumeDisplayOnly;
    control_status = $gpuConsoleState.compute_location } | ConvertTo-Json
