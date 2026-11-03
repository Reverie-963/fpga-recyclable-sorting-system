param(
    [string]$ToolRoot=$env:KM1_TOOL_ROOT,
    [Parameter(Mandatory=$true)][string]$Output,
    [ValidateSet('Csim','Synth','Cosim')][string]$Stage='Synth'
)
$ErrorActionPreference='Stop'
$phaseRoot=Split-Path $PSScriptRoot -Parent
$phaseOutput=[IO.Path]::GetFullPath($Output)
if ($phaseOutput -match '\s|[^\x20-\x7e]' -or -not $phaseOutput.StartsWith((Join-Path $phaseRoot 'local-only')+[IO.Path]::DirectorySeparatorChar)) { throw 'Use a fresh ASCII local-only build directory.' }
if (Test-Path -LiteralPath $phaseOutput) { throw 'Build directory must be new; preserve previous evidence.' }
foreach ($name in @('Vitis/bin/vitis-run.bat','Vitis/bin/v++.bat')) {
    if (-not (Test-Path -LiteralPath (Join-Path $ToolRoot $name))) { throw "Missing tool: $name" }
}
New-Item -ItemType Directory -Path (Join-Path $phaseOutput 'sources') | Out-Null
$phaseHashes=@{}
foreach ($name in @('phase_pixel_accel.cpp','phase_pixel_accel_tb.cpp')) {
    Copy-Item -LiteralPath (Join-Path $phaseRoot "pynq_km1_vision/src/hls/$name") -Destination (Join-Path $phaseOutput "sources/$name")
    $phaseHashes[$name]=(Get-FileHash -LiteralPath (Join-Path $phaseOutput "sources/$name") -Algorithm SHA256).Hash.ToLowerInvariant()
}
$phaseConfig=Join-Path $phaseOutput 'phase.cfg'
$phaseWork=Join-Path $phaseOutput 'phase_pixel_accel'
$phaseLines=@('part=xc7z020clg400-1','[hls]','flow_target=vivado','syn.top=phase_pixel_accel',
    ('syn.file='+$phaseOutput.Replace('\','/')+'/sources/phase_pixel_accel.cpp'),
    ('tb.file='+$phaseOutput.Replace('\','/')+'/sources/phase_pixel_accel_tb.cpp'),
    'clock=10ns','cosim.argv=--small','package.output.format=ip_catalog')
[IO.File]::WriteAllLines($phaseConfig,$phaseLines,[Text.Encoding]::ASCII)
$phaseVitis=Join-Path $ToolRoot 'Vitis/bin/vitis-run.bat';$phaseVpp=Join-Path $ToolRoot 'Vitis/bin/v++.bat'
& $phaseVitis --mode hls --config $phaseConfig --work_dir $phaseWork --csim 2>&1 | Tee-Object -FilePath (Join-Path $phaseOutput 'csim.log')
if ($LASTEXITCODE) { throw 'Phase pixel C simulation failed.' }
$phaseResult=@{top='phase_pixel_accel';toolchain='2026.1';csim='passed';source_sha256=$phaseHashes;board_programmed=$false;mechanical_motion=$false;whole_system_speedup_measured=$false}
if ($Stage -ne 'Csim') {
    & $phaseVpp --compile --mode hls --config $phaseConfig --work_dir $phaseWork 2>&1 | Tee-Object -FilePath (Join-Path $phaseOutput 'synthesis.log')
    if ($LASTEXITCODE -or -not (Test-Path (Join-Path $phaseWork 'hls/syn'))) { throw 'Phase pixel C synthesis failed.' }
    $phaseResult.csynth='passed'
}
if ($Stage -eq 'Cosim') {
    if (-not (Test-Path (Join-Path $phaseWork 'hls/syn'))) { throw 'Synthesis is required before cosimulation.' }
    & $phaseVitis --mode hls --config $phaseConfig --work_dir $phaseWork --cosim 2>&1 | Tee-Object -FilePath (Join-Path $phaseOutput 'cosim.log')
    if ($LASTEXITCODE) { throw 'Phase pixel RTL cosimulation failed.' }
    $phaseResult.cosim='passed_small_transactions'
}
$phaseResult | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $phaseOutput 'result.json') -Encoding utf8
Write-Output "Candidate validated to $Stage only; no Vivado integration or board load: $phaseOutput"
