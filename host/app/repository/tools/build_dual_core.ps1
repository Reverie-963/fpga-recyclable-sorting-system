param(
    [Parameter(Mandatory=$true)][string]$ToolRoot,
    [string]$OutputDir,
    [ValidateSet('all','mask_geometry_accel','visual_evidence_accel')][string]$Kernel='all',
    [ValidateSet('hls','cosim','vivado','all')][string]$Stage='all'
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path $PSScriptRoot -Parent
if (!$OutputDir) { $OutputDir = Join-Path $Repo 'artifacts/dual-core' }
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force $OutputDir | Out-Null
$OutputDir = (Resolve-Path $OutputDir).Path
if ($OutputDir -match '[^\x20-\x7e]|\s') { throw 'Use an ASCII output path without spaces' }
if ($Stage -in @('all','vivado') -and (Test-Path (Join-Path $OutputDir 'vivado_project'))) {
    throw 'Vivado project already exists; use a fresh output directory rather than overwrite evidence'
}
$Vitis = Join-Path $ToolRoot 'Vitis/bin/vitis-run.bat'
$Vpp = Join-Path $ToolRoot 'Vitis/bin/v++.bat'
$Vivado = Join-Path $ToolRoot 'Vivado/bin/vivado.bat'
function Invoke-Logged($Tool, $Arguments, $Log) {
    # Some AMD launchers emit harmless stderr; exit status remains mandatory.
    $saved = $ErrorActionPreference
    try { $ErrorActionPreference='Continue'; & $Tool @Arguments *> $Log; $code=$LASTEXITCODE }
    finally { $ErrorActionPreference=$saved }
    if ($code -ne 0) { throw "Tool failed ($code): $Log" }
}
Push-Location $OutputDir
try {
    if ($Stage -ne 'vivado') {
        $Kernels = if ($Kernel -eq 'all') { @('mask_geometry_accel','visual_evidence_accel') } else { @($Kernel) }
        foreach ($Name in $Kernels) {
            $Work = Join-Path $OutputDir $Name
            $Config = Join-Path $OutputDir "$Name.cfg"
            $Hls = (Join-Path $Repo 'pynq_km1_vision/src/hls').Replace('\','/')
            $ManifestPath = Join-Path $OutputDir "$Name.sources.json"
            $SourcePaths = @("$Hls/$Name.cpp", "$Hls/${Name}_tb.cpp", "$Hls/morph_stream.hpp")
            $Sources = @{}
            foreach ($Source in $SourcePaths) { $Sources[(Split-Path $Source -Leaf)] = (Get-FileHash $Source -Algorithm SHA256).Hash }
            if ($Stage -eq 'cosim') {
                if (!(Test-Path $ManifestPath)) { throw "Missing synthesis source manifest: $ManifestPath" }
                $PreviousSources = Get-Content $ManifestPath -Raw | ConvertFrom-Json
                foreach ($Key in $Sources.Keys) {
                    if ($PreviousSources.$Key -ne $Sources[$Key]) { throw "Source changed since synthesis: $Key" }
                }
            } else {
                # Invalidate any earlier successful build before attempting a
                # rebuild. Publish the new source stamp only after all gates.
                if (Test-Path -LiteralPath $ManifestPath) { Remove-Item -LiteralPath $ManifestPath }
            }
            [IO.File]::WriteAllLines($Config, @(
                'part=xc7z020clg400-1','[hls]','flow_target=vivado',"syn.top=$Name",
                "syn.file=$Hls/$Name.cpp", "tb.file=$Hls/${Name}_tb.cpp",
                'clock=10ns','package.output.format=ip_catalog',
                'cosim.enable_dataflow_profiling=true', 'cosim.random_stall=true'
            ), [Text.Encoding]::ASCII)
            if ($Stage -ne 'cosim') {
                Invoke-Logged $Vitis @('--mode','hls','--config',$Config,'--work_dir',$Work,'--csim') "$Name.csim.log"
                Invoke-Logged $Vpp @('--compile','--mode','hls','--config',$Config,'--work_dir',$Work) "$Name.synth.log"
            }
            if (!(Test-Path "$Work/hls/syn")) { throw 'Missing synthesis outputs' }
            # Both testbenches have no argv; no csim/cosim arguments are needed.
            $CosimLog = if ($Stage -eq 'cosim') { "$Name.stall.cosim.log" } else { "$Name.cosim.log" }
            Invoke-Logged $Vitis @('--mode','hls','--config',$Config,'--work_dir',$Work,'--cosim') $CosimLog
            if ($Stage -ne 'cosim') {
                Invoke-Logged $Vitis @('--mode','hls','--config',$Config,'--work_dir',$Work,'--package') "$Name.package.log"
            }
            if (!(Test-Path "$Work/hls/impl/ip/component.xml")) { throw 'Missing packaged IP' }
            if ($Stage -ne 'cosim') { $Sources | ConvertTo-Json | Set-Content $ManifestPath -Encoding utf8 }
            Write-Host "${Name}: C simulation, synthesis, RTL co-simulation and IP package passed"
        }
    }
    if ($Stage -in @('vivado','all')) {
        foreach ($Name in @('mask_geometry_accel','visual_evidence_accel')) {
            $ManifestPath = Join-Path $OutputDir "$Name.sources.json"
            if (!(Test-Path $ManifestPath)) { throw "Missing synthesis source manifest: $ManifestPath" }
            $Sources = Get-Content $ManifestPath -Raw | ConvertFrom-Json
            foreach ($Property in $Sources.PSObject.Properties) {
                $Source = Join-Path $Repo "pynq_km1_vision/src/hls/$($Property.Name)"
                if ((Get-FileHash $Source -Algorithm SHA256).Hash -ne $Property.Value) { throw "Stale HLS IP: $Source" }
            }
        }
        $env:KM1_DUAL_BUILD = $OutputDir
        $env:PYNQ_BOARD_REPO = Join-Path $Repo 'hardware/board_files'
        Invoke-Logged $Vivado @('-mode','batch','-source',(Join-Path $Repo 'pynq_km1_vision/build/integrate_bd.tcl')) 'vivado.log'
        Write-Host "Built paired outputs in $OutputDir/board"
        $Assets = @{}
        foreach ($Name in @('vision.bit','vision.hwh')) {
            $Assets[$Name] = (Get-FileHash (Join-Path $OutputDir "board/$Name") -Algorithm SHA256).Hash
        }
        $Assets | ConvertTo-Json | Set-Content (Join-Path $OutputDir 'board/sha256.json') -Encoding utf8
    }
} finally { Pop-Location }

