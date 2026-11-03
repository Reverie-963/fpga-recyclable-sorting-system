param(
    [Parameter(Mandatory=$true)][string]$ToolRoot,
    [string]$Output,
    [ValidateSet('Csim','Full')][string]$Stage='Csim'
)
$ErrorActionPreference='Stop'
$stationRoot=Split-Path $PSScriptRoot -Parent
$stationProject=Join-Path $stationRoot 'pynq_km1_vision'
if (-not $Output) { $Output=Join-Path $stationRoot ('local-only/station-vision-build/'+(Get-Date -Format 'yyyyMMdd-HHmmss')) }
$Output=[IO.Path]::GetFullPath($Output)
if ($Output -match '\s|[^\x20-\x7e]' -or -not $Output.StartsWith((Join-Path $stationRoot 'local-only')+[IO.Path]::DirectorySeparatorChar)) { throw 'Use a fresh ASCII local-only output directory.' }
if (Test-Path -LiteralPath $Output) { throw 'Build output already exists; use a fresh directory.' }
foreach ($entry in @('Vitis/bin/vitis-run.bat','Vitis/bin/v++.bat','Vivado/bin/vivado.bat')) {
    if (-not (Test-Path -LiteralPath (Join-Path $ToolRoot $entry))) { throw "Missing AMD tool: $entry" }
}
New-Item -ItemType Directory -Path (Join-Path $Output 'sources') -Force | Out-Null
foreach ($name in @('station_vision_accel.cpp','station_vision_accel_tb.cpp')) {
    Copy-Item -LiteralPath (Join-Path $stationProject "src/hls/$name") -Destination (Join-Path $Output "sources/$name")
}
Copy-Item -LiteralPath (Join-Path $stationProject 'build/station_vision_bd.tcl') -Destination (Join-Path $Output 'sources/station_vision_bd.tcl')
$stationConfig=Join-Path $Output 'station.cfg'
$stationWork=Join-Path $Output 'station_vision_accel'
$stationLines=@('part=xc7z020clg400-1','[hls]','flow_target=vivado','syn.top=station_vision_accel',
    ('syn.file='+($Output.Replace('\','/')+'/sources/station_vision_accel.cpp')),
    ('tb.file='+($Output.Replace('\','/')+'/sources/station_vision_accel_tb.cpp')),
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
if ($LASTEXITCODE) { throw 'Station vision C simulation failed.' }
if ($Stage -eq 'Csim') { Write-Output "C simulation passed: $Output"; return }
& $stationVpp --compile --mode hls --config $stationConfig --work_dir $stationWork
if ($LASTEXITCODE -or -not (Test-Path (Join-Path $stationWork 'hls/syn'))) { throw 'Station vision synthesis failed.' }
& $stationVitis --mode hls --config $stationConfig --work_dir $stationWork --cosim
if ($LASTEXITCODE) { throw 'Station vision RTL co-simulation failed.' }
& $stationVitis --mode hls --config $stationConfig --work_dir $stationWork --package
if ($LASTEXITCODE) { throw 'Station vision packaging failed.' }
$stationPreviousBuild=$env:KM1_STATION_BUILD;$stationPreviousRepo=$env:PYNQ_BOARD_REPO
try {
    $env:KM1_STATION_BUILD=$Output;$env:PYNQ_BOARD_REPO=Join-Path $stationRoot 'hardware/board_files'
    Push-Location $Output
    try { & $stationVivado -mode batch -source (Join-Path $Output 'sources/station_vision_bd.tcl') }
    finally { Pop-Location }
    if ($LASTEXITCODE) { throw 'Station vision bitstream build failed.' }
} finally { $env:KM1_STATION_BUILD=$stationPreviousBuild;$env:PYNQ_BOARD_REPO=$stationPreviousRepo }
$stationAssets=@{}
foreach ($name in @('station_vision.bit','station_vision.hwh')) {
    $stationAssets[$name]=(Get-FileHash -LiteralPath (Join-Path $Output "board/$name") -Algorithm SHA256).Hash.ToLowerInvariant()
}
$stationSources=@{}
Get-ChildItem -LiteralPath (Join-Path $Output 'sources') -File | ForEach-Object {
    $stationSources[$_.Name]=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}
@{toolchain='2026.1';assets=$stationAssets;sources=$stationSources;csim='pass_including_640x480';cosim='pass_14_small_transactions';mechanical_motion=$false;board_programmed=$false} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Output 'build-result.json') -Encoding utf8
Write-Output "Station vision candidate built; not loaded: $Output"
