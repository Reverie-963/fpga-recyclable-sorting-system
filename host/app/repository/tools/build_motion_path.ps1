param(
    [Parameter(Mandatory=$true)][string]$ToolRoot,
    [string]$Output,
    [ValidateSet('Csim','Synth','Cosim','Full')][string]$Stage='Cosim',
    [switch]$FullCosim
)
$ErrorActionPreference='Stop'
$motionRepo=Split-Path $PSScriptRoot -Parent
if (!$Output) { $Output=Join-Path $motionRepo ('local-only/motion-path-build/'+(Get-Date -Format 'yyyyMMdd-HHmmss')) }
$Output=[IO.Path]::GetFullPath($Output)
if ($Output -match '\s|[^\x20-\x7e]' -or !( $Output.StartsWith((Join-Path $motionRepo 'local-only')+[IO.Path]::DirectorySeparatorChar))) { throw 'Use a fresh ASCII local-only output directory.' }
if (Test-Path -LiteralPath $Output) { throw 'Output exists; use a fresh directory.' }
$motionVitis=Join-Path $ToolRoot 'Vitis/bin/vitis-run.bat'
$motionVpp=Join-Path $ToolRoot 'Vitis/bin/v++.bat'
$motionVivado=Join-Path $ToolRoot 'Vivado/bin/vivado.bat'
foreach ($motionTool in @($motionVitis,$motionVpp,$motionVivado)) {
    if (!(Test-Path -LiteralPath $motionTool)) { throw "Tool missing: $motionTool" }
}
New-Item -ItemType Directory -Path (Join-Path $Output 'sources') -Force | Out-Null
foreach ($motionName in @('motion_path_accel.cpp','motion_path_accel.hpp','motion_sine_q24.hpp','motion_path_accel_tb.cpp')) {
    Copy-Item -LiteralPath (Join-Path $motionRepo "pynq_km1_vision/src/hls/$motionName") -Destination (Join-Path $Output "sources/$motionName")
}
Copy-Item -LiteralPath (Join-Path $motionRepo 'pynq_km1_vision/build/motion_path_bd.tcl') -Destination (Join-Path $Output 'sources/motion_path_bd.tcl')
$motionConfig=Join-Path $Output 'motion.cfg'
$motionWork=Join-Path $Output 'motion_path_accel'
$motionSlash=$Output.Replace('\','/')
$motionCosimArg=if ($FullCosim) { 'cosim.argv=' } else { 'cosim.argv=--small' }
[IO.File]::WriteAllLines($motionConfig,@('part=xc7z020clg400-1','[hls]','flow_target=vivado','syn.top=motion_path_accel',
    "syn.file=$motionSlash/sources/motion_path_accel.cpp","tb.file=$motionSlash/sources/motion_path_accel_tb.cpp",
    'clock=10ns',$motionCosimArg,'cosim.random_stall=true','cosim.enable_dataflow_profiling=true',
    'package.output.format=ip_catalog'),[Text.Encoding]::ASCII)
function Invoke-MotionTool($Tool,$Arguments,$Log) {
    $motionSaved=$ErrorActionPreference
    try { $ErrorActionPreference='Continue'; & $Tool @Arguments *> $Log; $motionCode=$LASTEXITCODE }
    finally { $ErrorActionPreference=$motionSaved }
    if ($motionCode) { throw "Tool failed ($motionCode): $Log" }
}
Invoke-MotionTool $motionVitis @('--version') (Join-Path $Output 'version.log')
if ((Get-Content (Join-Path $Output 'version.log') -Raw) -notmatch '2026\.1') { throw 'Vitis 2026.1 required.' }
Invoke-MotionTool $motionVitis @('--mode','hls','--config',$motionConfig,'--work_dir',$motionWork,'--csim') (Join-Path $Output 'csim.log')
if ($Stage -eq 'Csim') { Write-Output "C simulation passed: $Output"; return }
Invoke-MotionTool $motionVpp @('--compile','--mode','hls','--config',$motionConfig,'--work_dir',$motionWork) (Join-Path $Output 'synth.log')
if (!(Test-Path (Join-Path $motionWork 'hls/syn'))) { throw 'Missing synthesis output.' }
if ($Stage -eq 'Synth') { Write-Output "Synthesis passed: $Output"; return }
# C simulation always covers 4096 segments. -FullCosim repeats them in RTL;
# default --small is a quick smoke check and is reported separately.
Invoke-MotionTool $motionVitis @('--mode','hls','--config',$motionConfig,'--work_dir',$motionWork,'--cosim') (Join-Path $Output 'cosim.log')
if ($Stage -eq 'Cosim') { Write-Output "RTL co-simulation passed: $Output"; return }
Invoke-MotionTool $motionVitis @('--mode','hls','--config',$motionConfig,'--work_dir',$motionWork,'--package') (Join-Path $Output 'package.log')
$motionPrevious=$env:KM1_MOTION_BUILD;$motionBoardPrevious=$env:PYNQ_BOARD_REPO
try {
    $env:KM1_MOTION_BUILD=$Output;$env:PYNQ_BOARD_REPO=Join-Path $motionRepo 'hardware/board_files'
    Push-Location $Output
    try { Invoke-MotionTool $motionVivado @('-mode','batch','-source',(Join-Path $Output 'sources/motion_path_bd.tcl')) (Join-Path $Output 'vivado.log') }
    finally { Pop-Location }
} finally { $env:KM1_MOTION_BUILD=$motionPrevious;$env:PYNQ_BOARD_REPO=$motionBoardPrevious }
$motionAssets=@{};$motionSources=@{}
foreach ($motionName in @('motion_path.bit','motion_path.hwh')) {
    $motionAssets[$motionName]=(Get-FileHash -LiteralPath (Join-Path $Output "board/$motionName") -Algorithm SHA256).Hash.ToLowerInvariant()
}
Get-ChildItem -LiteralPath (Join-Path $Output 'sources') -File | ForEach-Object {
    $motionSources[$_.Name]=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}
@{toolchain='2026.1';assets=$motionAssets;sources=$motionSources;csim='pass_4096_segments';cosim=$(if ($FullCosim) {'pass_4096_segments'} else {'pass_small_smoke_only'});board_programmed=$false;mechanical_motion=$false} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Output 'build-result.json') -Encoding utf8
Write-Output "Motion-only candidate built; not loaded: $Output"
