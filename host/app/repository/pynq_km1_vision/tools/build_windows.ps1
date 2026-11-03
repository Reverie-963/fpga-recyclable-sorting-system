param(
 [string]$BoardRepo,
 [ValidateSet("segment", "segment_open")][string]$Kernel = "segment_open",
 [string]$EncoderXdc,
 [string]$ToolRoot = $env:KM1_TOOL_ROOT,
 [switch]$CheckOnly
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$RequiredVersion = (Get-Content -LiteralPath (Join-Path $ProjectRoot "build/toolchain_version.txt") -Raw).Trim()

function Get-CheckedTool([string]$Name, [string]$Component, [string[]]$VersionArgs) {
 if ($ToolRoot) {
  $ToolPath = Join-Path $ToolRoot "$Component\bin\$Name.bat"
 } else {
  $Command = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($Command) { $ToolPath = $Command.Source }
  else { $ToolPath = "D:\work\AMDDesignTools\$RequiredVersion\$Component\bin\$Name.bat" }
 }
 if (-not (Test-Path -LiteralPath $ToolPath -PathType Leaf)) {
  throw "Missing $Name at $ToolPath. Set -ToolRoot to the version directory containing Vivado and Vitis, or load the AMD environment; see docs/vivado_2026_1.md."
 }
 $ToolPath = (Resolve-Path -LiteralPath $ToolPath).Path
 # AMD launcher .bat files print a harmless stderr line on startup (e.g.
 # "dev is not recognized as an internal or external command"). With
 # $ErrorActionPreference = Stop (set above) that noise aborts the version
 # probe before Vivado/Vitis prints its real version. Relax the preference
 # only for this probe and restore it in finally.
 try {
  $ErrorActionPreference = 'Continue'
  $Output = (& $ToolPath @VersionArgs 2>&1 | Out-String).Trim()
 } finally {
  $ErrorActionPreference = 'Stop'
 }
 $VersionExit = $LASTEXITCODE
 $Match = [regex]::Match($Output, '(?im)^\s*(?:\*+\s*)?' + [regex]::Escape($Name) + '\s+v?(20\d{2}\.\d+(?:\.\d+)?)(?![\d.])')
 if (-not $Match.Success -or $Match.Groups[1].Value -ne $RequiredVersion) {
  throw "$Name must be exactly $RequiredVersion. Found: $Output"
 }
 if ($VersionExit -ne 0) {
  if ($Name -ne 'vivado') { throw "$Name version query failed (exit $VersionExit): $Output" }
  # Some Windows launchers print the version but return 1. Verify the Tcl
  # runtime independently; never waive an actual build failure.
  Write-Host "Vivado -version returned $VersionExit; verifying the batch runtime."
  $Probe = Join-Path $PSScriptRoot 'check_vivado.tcl'
   try {
    $ErrorActionPreference = 'Continue'
    $ProbeOutput = (& $ToolPath -mode batch -nolog -nojournal -notrace -source $Probe 2>&1 | Out-String).Trim()
   } finally {
    $ErrorActionPreference = 'Stop'
   }
  if ($LASTEXITCODE -ne 0 -or $ProbeOutput -notmatch ('(?m)^KM1_VIVADO_VERSION=' + [regex]::Escape($RequiredVersion) + '\s*$')) {
   throw "Vivado batch startup check failed: $ProbeOutput"
  }
 }
 Write-Host "$Name $RequiredVersion`: $ToolPath"
 return $ToolPath
}

# Check all tools before updating an existing project. Do not fall back
# to a legacy vitis_hls executable from a different release on PATH.
$Vivado = Get-CheckedTool "vivado" "Vivado" @("-version")
$VitisRun = Get-CheckedTool "vitis-run" "Vitis" @("--version")
$Vpp = Get-CheckedTool "v++" "Vitis" @("--version")
if ($CheckOnly) {
 if ($ProjectRoot -match '[^\x20-\x7e]|\s') {
  Write-Warning "Tool versions passed, but build from a short ASCII path without spaces (see docs/vivado_2026_1.md)."
 }
 Write-Host "Toolchain version check passed. Synthesis, device support and board files have not been checked."
 return
}
if (-not $BoardRepo -or -not $EncoderXdc) { throw "Build requires -BoardRepo and -EncoderXdc; use -CheckOnly for tool versions only." }
$BoardPath = (Resolve-Path -LiteralPath $BoardRepo).Path
$EncoderPath = (Resolve-Path -LiteralPath $EncoderXdc).Path
if (-not (Test-Path -LiteralPath $BoardPath -PathType Container)) { throw "BoardRepo must be a directory." }
if (-not (Test-Path -LiteralPath $EncoderPath -PathType Leaf)) { throw "EncoderXdc must be a file." }
foreach ($BuildPath in @($ProjectRoot, $BoardPath, $EncoderPath)) {
 if ($BuildPath -match '[^\x20-\x7e]|\s') { throw "Use short ASCII build paths without spaces: $BuildPath. See docs/vivado_2026_1.md." }
}
Push-Location $ProjectRoot
$PreviousBoardRepo = $env:PYNQ_BOARD_REPO
$PreviousKernel = $env:KM1_KERNEL
$PreviousEncoderXdc = $env:KM1_ENCODER_XDC
try {
 $env:PYNQ_BOARD_REPO = $BoardPath
 $env:KM1_KERNEL = $Kernel
 $env:KM1_ENCODER_XDC = $EncoderPath
 # The 2026.1 classic Tcl entry point can omit the design source from csim.
 # Use the supported component/config entry points for every HLS stage.
 $HlsWork = Join-Path $ProjectRoot "build/hls_$Kernel"
 $HlsConfig = Join-Path $ProjectRoot "build/hls_$Kernel.cfg"
 $DesignFile = Join-Path $ProjectRoot "src/hls/$Kernel.cpp"
 $TestbenchName = if ($Kernel -eq 'segment') { 'hls_tb.cpp' } else { 'hls_open_tb.cpp' }
 $TestbenchFile = Join-Path $ProjectRoot "sim/$TestbenchName"
 foreach ($SourceFile in @($DesignFile, $TestbenchFile)) {
  if (-not (Test-Path -LiteralPath $SourceFile -PathType Leaf)) { throw "Missing HLS source: $SourceFile" }
 }
 $ConfigLines = @(
  'part=xc7z020clg400-1', '[hls]', 'flow_target=vivado', "syn.top=$Kernel",
  ('syn.file=' + $DesignFile.Replace('\', '/')),
  ('tb.file=' + $TestbenchFile.Replace('\', '/')),
  'clock=10ns', 'package.output.format=ip_catalog'
 )
 [System.IO.File]::WriteAllLines($HlsConfig, $ConfigLines, [System.Text.Encoding]::ASCII)
 & $VitisRun --mode hls --config $HlsConfig --work_dir $HlsWork --csim
 if ($LASTEXITCODE -ne 0) { throw "HLS C simulation failed" }
 & $Vpp --compile --mode hls --config $HlsConfig --work_dir $HlsWork
 if ($LASTEXITCODE -ne 0) { throw "HLS synthesis failed" }
 if (-not (Test-Path -LiteralPath (Join-Path $HlsWork 'hls/syn'))) { throw "HLS synthesis outputs missing" }
 # Neither csim.argv nor cosim.argv is needed: both testbenches take no arguments.
 & $VitisRun --mode hls --config $HlsConfig --work_dir $HlsWork --cosim
 if ($LASTEXITCODE -ne 0) { throw "HLS C/RTL co-simulation failed" }
 & $VitisRun --mode hls --config $HlsConfig --work_dir $HlsWork --package
 if ($LASTEXITCODE -ne 0) { throw "HLS IP packaging failed" }
 if (-not (Test-Path -LiteralPath (Join-Path $HlsWork 'hls/impl/ip/component.xml'))) { throw "HLS IP component.xml missing" }
 & $Vivado -mode batch -source build/vivado.tcl
 if ($LASTEXITCODE -ne 0) { throw "Vivado build failed" }
 Write-Host "Built $Kernel with AMD $RequiredVersion : board/vision.bit and board/vision.hwh. Copy both together."
} finally {
 $env:PYNQ_BOARD_REPO = $PreviousBoardRepo
 $env:KM1_KERNEL = $PreviousKernel
 $env:KM1_ENCODER_XDC = $PreviousEncoderXdc
 Pop-Location
}
