param([Parameter(Mandatory=$true)][string]$ToolRoot,[ValidateSet('Csim','Full')][string]$Stage='Csim')
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$repo=Join-Path $root 'host/app/repository'
# Builds new local output only. Neither helper programs a board.
& (Join-Path $repo 'tools/build_station_vision.ps1') -ToolRoot $ToolRoot -Stage $Stage
& (Join-Path $repo 'tools/build_motion_path.ps1') -ToolRoot $ToolRoot -Stage $Stage
