param(
    [Parameter(Mandatory=$true)][string]$ToolRoot,
    [string]$Output,
    [ValidateSet('Csim','Synth','Cosim','Full')][string]$Stage='Csim'
)
$ErrorActionPreference='Stop'
$stationRoot=Split-Path $PSScriptRoot -Parent
$stationProject=Join-Path $stationRoot 'pynq_km1_vision'
if (-not $Output) { $Output=Join-Path $stationRoot ('local-only/frame-pipeline-build/'+(Get-Date -Format 'yyyyMMdd-HHmmss')) }
$Output=[IO.Path]::GetFullPath($Output)
if ($Output -match '\s|[^\x20-\x7e]' -or -not $Output.StartsWith((Join-Path $stationRoot 'local-only')+[IO.Path]::DirectorySeparatorChar)) { throw 'Use a fresh ASCII local-only output directory.' }
if (Test-Path -LiteralPath $Output) { throw 'Build output already exists; use a fresh directory.' }
foreach ($entry in @('Vitis/bin/vitis-run.bat','Vitis/bin/v++.bat','Vivado/bin/vivado.bat')) {
    if (-not (Test-Path -LiteralPath (Join-Path $ToolRoot $entry))) { throw "Missing AMD tool: $entry" }
}
New-Item -ItemType Directory -Path (Join-Path $Output 'sources') -Force | Out-Null
foreach ($name in @('station_frame_pipeline.cpp','station_frame_pipeline_tb.cpp')) {
    Copy-Item -LiteralPath (Join-Path $stationProject "src/hls/$name") -Destination (Join-Path $Output "sources/$name")
}
Copy-Item -LiteralPath (Join-Path $stationProject 'build/frame_pipeline_bd.tcl') -Destination (Join-Path $Output 'sources/frame_pipeline_bd.tcl')
$stationConfig=Join-Path $Output 'station.cfg'
$stationWork=Join-Path $Output 'station_frame_pipeline'
$stationLines=@('part=xc7z020clg400-1','[hls]','flow_target=vivado','syn.top=station_frame_pipeline',
    ('syn.file='+($Output.Replace('\','/')+'/sources/station_frame_pipeline.cpp')),
    ('tb.file='+($Output.Replace('\','/')+'/sources/station_frame_pipeline_tb.cpp')),
    'clock=10ns','cosim.argv=--small','package.output.format=ip_catalog')
[IO.File]::WriteAllLines($stationConfig,$stationLines,[Text.Encoding]::ASCII)
$stationVitis=Join-Path $ToolRoot 'Vitis/bin/vitis-run.bat'
$stationVpp=Join-Path $ToolRoot 'Vitis/bin/v++.bat'
$stationVivado=Join-Path $ToolRoot 'Vivado/bin/vivado.bat'
try {
    $ErrorActionPreference='Continue'
    $stationVersion=(& $stationVitis --version 2>&1 | Out-String)
} finally { $ErrorActionPreference='Stop' }
if ($stationVersion -notmatch '2026\.1') { throw 'Vitis 2026.1 required.' }
& $stationVitis --mode hls --config $stationConfig --work_dir $stationWork --csim
if ($LASTEXITCODE) { throw 'Frame pipeline C simulation failed.' }
if ($Stage -eq 'Csim') { Write-Output "C simulation passed: $Output"; return }
& $stationVpp --compile --mode hls --config $stationConfig --work_dir $stationWork
if ($LASTEXITCODE -or -not (Test-Path (Join-Path $stationWork 'hls/syn'))) { throw 'Frame pipeline synthesis failed.' }
if ($Stage -eq 'Synth') { Write-Output "Synthesis passed: $Output"; return }
& $stationVitis --mode hls --config $stationConfig --work_dir $stationWork --cosim
if ($LASTEXITCODE) { throw 'Frame pipeline RTL co-simulation failed.' }
if ($Stage -eq 'Cosim') { Write-Output "RTL co-simulation passed: $Output"; return }
& $stationVitis --mode hls --config $stationConfig --work_dir $stationWork --package
if ($LASTEXITCODE) { throw 'Frame pipeline packaging failed.' }
$stationPreviousBuild=$env:KM1_FRAME_BUILD;$stationPreviousRepo=$env:PYNQ_BOARD_REPO
try {
    $env:KM1_FRAME_BUILD=$Output;$env:PYNQ_BOARD_REPO=Join-Path $stationRoot 'hardware/board_files'
    Push-Location $Output
    try { & $stationVivado -mode batch -source (Join-Path $Output 'sources/frame_pipeline_bd.tcl') }
    finally { Pop-Location }
    if ($LASTEXITCODE) { throw 'Frame pipeline bitstream build failed.' }
} finally { $env:KM1_FRAME_BUILD=$stationPreviousBuild;$env:PYNQ_BOARD_REPO=$stationPreviousRepo }
$stationAssets=@{}
foreach ($name in @('frame_pipeline.bit','frame_pipeline.hwh')) {
    $stationAssets[$name]=(Get-FileHash -LiteralPath (Join-Path $Output "board/$name") -Algorithm SHA256).Hash.ToLowerInvariant()
}
$stationSources=@{}
Get-ChildItem -LiteralPath (Join-Path $Output 'sources') -File | ForEach-Object {
    $stationSources[$_.Name]=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}
@{toolchain='2026.1';assets=$stationAssets;sources=$stationSources;csim='pass_including_640x480';cosim='pass_12_small_transactions_and_invalid_input';mechanical_motion=$false;board_programmed=$false} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Output 'build-result.json') -Encoding utf8
Write-Output "Frame pipeline candidate built; not loaded: $Output"
