param(
 [Parameter(Mandatory=$true)][string]$VivadoExe,
 [Parameter(Mandatory=$true)][string]$BuildRoot,
 [switch]$CheckOnly
)
$ErrorActionPreference='Stop'
$packageRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$outputRoot=[IO.Path]::GetFullPath($BuildRoot)
if($outputRoot -match '\s|[^\x20-\x7e]'){throw 'Use an ASCII output path without spaces.'}
if(Test-Path -LiteralPath $outputRoot){throw 'Build output exists. Choose a fresh directory.'}
if($outputRoot.StartsWith($packageRoot+[IO.Path]::DirectorySeparatorChar)){throw 'Build outside the submitted source tree.'}
if(!(Test-Path -LiteralPath $VivadoExe)){throw 'Vivado executable missing.'}
foreach($name in @('vision_motion_bd.tcl','finish_joint.tcl','ip/xilinx_com_hls_station_vision_accel_1_0.zip','ip/xilinx_com_hls_motion_path_accel_1_0.zip')){
 if(!(Test-Path -LiteralPath (Join-Path $PSScriptRoot $name))){throw "Missing input: $name"}
}
if($CheckOnly){Write-Output 'Input files and fresh output path checked. No synthesis or board access.';return}
New-Item -ItemType Directory -Path $outputRoot | Out-Null
foreach($sub in @('ip/vision','ip/motion','evidence','results')){New-Item -ItemType Directory -Path (Join-Path $outputRoot $sub) | Out-Null}
Expand-Archive -LiteralPath (Join-Path $PSScriptRoot 'ip/xilinx_com_hls_station_vision_accel_1_0.zip') -DestinationPath (Join-Path $outputRoot 'ip/vision')
Expand-Archive -LiteralPath (Join-Path $PSScriptRoot 'ip/xilinx_com_hls_motion_path_accel_1_0.zip') -DestinationPath (Join-Path $outputRoot 'ip/motion')
$saved=@{joint=$env:KM1_JOINT_BUILD;board=$env:PYNQ_BOARD_REPO;finish=$env:KM1_FINISH_ROOT;routed=$env:KM1_ROUTED_CHECKPOINT;hwh=$env:KM1_ROUTED_HWH}
try{
 $env:KM1_JOINT_BUILD=$outputRoot
 $env:PYNQ_BOARD_REPO=Join-Path $packageRoot 'host/app/repository/hardware/board_files'
 & $VivadoExe -mode batch -source (Join-Path $PSScriptRoot 'vision_motion_bd.tcl') -log (Join-Path $outputRoot 'logical_build.log') -journal (Join-Path $outputRoot 'logical_build.jou')
 $firstExit=$LASTEXITCODE
 $routed=Join-Path $outputRoot 'v/j.runs/impl_1/design_1_wrapper_routed.dcp'
 $hwh=Join-Path $outputRoot 'v/j.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh'
 if(!(Test-Path -LiteralPath $routed) -or !(Test-Path -LiteralPath $hwh)){throw 'Routed checkpoint/HWH missing; no physical finish attempted.'}
 if($firstExit -ne 0 -and (Get-Content (Join-Path $outputRoot 'logical_build.log') -Raw) -notmatch 'max timing failed'){throw 'Initial build failed for an unrecognized reason. Inspect the retained log.'}
 $env:KM1_FINISH_ROOT=Join-Path $outputRoot 'physical_finish'
 $env:KM1_ROUTED_CHECKPOINT=$routed
 $env:KM1_ROUTED_HWH=$hwh
 & $VivadoExe -mode batch -source (Join-Path $PSScriptRoot 'finish_joint.tcl') -log (Join-Path $outputRoot 'physical_finish.log') -journal (Join-Path $outputRoot 'physical_finish.jou')
 if($LASTEXITCODE){throw 'Physical finish failed. No board programming performed.'}
 Write-Output "Build finished in $outputRoot. Check reports and pair hashes before deployment. No board programming."
}finally{
 $env:KM1_JOINT_BUILD=$saved.joint;$env:PYNQ_BOARD_REPO=$saved.board;$env:KM1_FINISH_ROOT=$saved.finish
 $env:KM1_ROUTED_CHECKPOINT=$saved.routed;$env:KM1_ROUTED_HWH=$saved.hwh
}
