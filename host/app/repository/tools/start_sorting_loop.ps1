param(
    [ValidateSet('Preflight', 'Preview', 'Execute', 'Demo', 'CaptureEmpty', 'BindVision', 'Stop', 'Review')][string]$Mode = 'Preflight',
    [string]$Config = '',
    [switch]$KnownItem,
    [switch]$Unified,
    [ValidateSet('AUTO','PAPER','METAL','PLASTIC')][string]$Material='AUTO',
    [string]$StationBinding = '',
    [ValidateSet('', 'qualified', 'compact')][string]$MotionProfile = '',
    [switch]$Ready,
    [string]$TaskId = '',
    [string]$Observation = '',
    [ValidateRange(1, 100)][int]$MaxTasks = 1,
    [ValidateRange(1, 3600)][int]$MaxSeconds = 300,
    [string]$DemoOutput = ''
)
$ErrorActionPreference = 'Stop'
$sortingRepoRoot = Split-Path $PSScriptRoot -Parent
$sortingProject = Join-Path $sortingRepoRoot 'pynq_km1_vision'
$sortingVisionActive = Join-Path $sortingRepoRoot 'local-only/station-vision-deployment/active.json'
if ($Unified -and $KnownItem) { throw 'Select Unified or KnownItem.' }
if (-not $Unified -and $Material -ne 'AUTO') { throw 'Material selection requires -Unified.' }
if ($Unified -and $Mode -notin @('Preflight','Preview','Execute','Stop','Review')) { throw 'Unified supports Preflight, Preview, Execute, Stop and Review.' }
if ($Unified -and -not $PSBoundParameters.ContainsKey('MaxTasks')) { $MaxTasks = 6 }
if ($Unified -and -not $PSBoundParameters.ContainsKey('MaxSeconds')) { $MaxSeconds = 1200 }
if (-not $Config) {
    $Config = if ($KnownItem -or $Unified) { 'pynq_km1_vision/config/sorting_known.local.json' } else { 'pynq_km1_vision/config/sorting_station.example.json' }
}
if ($Mode -eq 'Stop') {
    $sortingStopFile = Join-Path $sortingRepoRoot 'local-only/STOP_SORTING'
    New-Item -ItemType Directory -Path (Split-Path $sortingStopFile -Parent) -Force | Out-Null
    Set-Content -LiteralPath $sortingStopFile -Value 'Operator requested stop. Review physical state before resuming.' -Encoding utf8
    Write-Output 'Stop requested. Any uncertain cycle remains locked for review.'
    return
}
if ($Mode -eq 'CaptureEmpty' -and -not $KnownItem) { throw 'CaptureEmpty requires -KnownItem.' }
if ($KnownItem -and $MaxTasks -ne 1) { throw 'KnownItem allows one supervised cycle per launch.' }
if (-not $KnownItem -and $MotionProfile) { throw 'MotionProfile requires -KnownItem.' }
$sortingPython = Join-Path $sortingRepoRoot $(if ($Mode -in @('Preflight', 'Demo')) {
    '.venv/Scripts/python.exe'
} else { '.venv-rocm/Scripts/python.exe' })
if (-not (Test-Path -LiteralPath $sortingPython)) { throw "Missing runtime: $sortingPython" }
if (-not [IO.Path]::IsPathRooted($Config)) { $Config = Join-Path $sortingRepoRoot $Config }
if ($Mode -eq 'Execute' -and (Test-Path -LiteralPath $sortingVisionActive)) {
    $sortingRuntimeConfig = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
    if ((-not $KnownItem -and -not $Unified) -or $sortingRuntimeConfig.statistics.backend -ne 'pl_board_local') {
        throw 'New board vision is active. The legacy motion adapter cannot execute against the replaced dual-core interface.'
    }
}
if ($Mode -eq 'BindVision' -and -not $KnownItem) { throw 'BindVision requires -KnownItem.' }
$sortingStationPointer = Join-Path $sortingRepoRoot 'local-only/station-grasp-integration/active.json'
if (($KnownItem -or $Unified) -and ($Mode -in @('Preflight', 'Preview', 'Execute') -or ($Unified -and $Mode -eq 'Review')) -and -not $StationBinding -and (Test-Path -LiteralPath $sortingStationPointer)) {
    $sortingStationActive = Get-Content -LiteralPath $sortingStationPointer -Raw | ConvertFrom-Json
    $StationBinding = [IO.Path]::GetFullPath((Join-Path (Split-Path $sortingStationPointer -Parent) $sortingStationActive.path))
    if ((Get-FileHash -LiteralPath $StationBinding -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sortingStationActive.sha256) {
        throw 'Station calibration binding changed. Review the saved integration record.'
    }
}
if ($StationBinding -and ((-not $KnownItem -and -not $Unified) -or ($Mode -notin @('Preflight', 'Preview', 'Execute') -and -not ($Unified -and $Mode -eq 'Review')))) {
    throw 'StationBinding supports KnownItem or Unified Preflight, Preview, Execute; Unified also supports Review.'
}
if ($Unified -and -not $StationBinding) { throw 'Unified requires the existing station calibration binding.' }
if ($StationBinding -and -not [IO.Path]::IsPathRooted($StationBinding)) { $StationBinding = Join-Path $sortingRepoRoot $StationBinding }
if ($StationBinding) {
    # Inference stays in the deployed HTTP/GPU service. Use the same OpenCV
    # runtime that measured the marker calibration, including ArUco support.
    $sortingPython = Join-Path $sortingRepoRoot '.venv/Scripts/python.exe'
}
$sortingModule = if ($Unified) { 'src.sorting_unified_run' } elseif ($StationBinding) { 'src.sorting_relocated_run' } elseif ($KnownItem) { 'src.sorting_known_run' } else { 'src.sorting_run' }
$sortingArgs = @('-u', '-m', $sortingModule, '--config', $Config)
if ($StationBinding) { $sortingArgs += @('--binding', $StationBinding) }
if ($Unified) { $sortingArgs += @('--material', $Material) }
if ($KnownItem -and $MotionProfile) { $sortingArgs += @('--motion-profile', $MotionProfile) }
if ($Mode -eq 'Demo') {
    if (-not $DemoOutput) {
        $DemoOutput = Join-Path $sortingRepoRoot ('local-only/sorting-demo-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
    } elseif (-not [IO.Path]::IsPathRooted($DemoOutput)) {
        $DemoOutput = Join-Path $sortingRepoRoot $DemoOutput
    }
    $sortingArgs = @('-u', '-m', 'src.sorting_demo', '--output', $DemoOutput)
} elseif ($Mode -eq 'Preflight') { $sortingArgs += '--preflight' }
elseif ($Mode -eq 'Execute') {
    if ($Unified) { $sortingArgs += @('--execute', '--max-items', $MaxTasks, '--max-seconds', $MaxSeconds) }
    elseif ($KnownItem) { $sortingArgs += '--execute' }
    else { $sortingArgs += @('--execute', '--max-tasks', $MaxTasks, '--max-seconds', $MaxSeconds) }
}
elseif ($Mode -eq 'CaptureEmpty') { $sortingArgs += '--capture-empty-reference' }
elseif ($Mode -eq 'BindVision') { $sortingArgs += '--bind-vision' }
elseif ($Mode -eq 'Review') {
    if ($Unified) {
        if (-not $Observation) { throw 'Unified Review requires Observation; resolve any individual motion task first.' }
        $sortingArgs += @('--resolve-batch', '--observation', $Observation)
    } else {
        if (-not $TaskId -or -not $Observation) { throw 'Review requires TaskId and Observation.' }
        $sortingArgs += @('--resolve-task', $TaskId, '--observation', $Observation)
    }
}
elseif ($Mode -eq 'Preview' -and ($KnownItem -or $Unified)) { $sortingArgs += '--preview' }
if (($KnownItem -or $Unified) -and $Ready) { $sortingArgs += '--ready' }
$env:PYTHONUTF8 = '1'
Push-Location $sortingProject
try {
    & $sortingPython @sortingArgs
    if ($LASTEXITCODE -ne 0) { throw "Sorting stopped; inspect the task ledger. Exit: $LASTEXITCODE" }
} finally { Pop-Location }
