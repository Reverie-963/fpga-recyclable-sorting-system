param(
    [Parameter(Mandatory=$true)][string]$Deployment,
    [ValidateSet('rocm','cpu')][string]$Backend='rocm',
    [int]$DetectorPort=8824,
    [int]$ConsolePort=8810,
    [switch]$DetectorOnly,
    [switch]$ConsoleOnly
)
$ErrorActionPreference='Stop'
$trainedRoot=Split-Path $PSScriptRoot -Parent
$trainedDeployment=(Resolve-Path -LiteralPath $Deployment).Path
$trainedRecord=Get-Content -LiteralPath (Join-Path $trainedDeployment 'deployment.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$env:PYTHONUTF8='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:YOLOE_RECYCLABLE_POLICY=Join-Path $trainedDeployment 'policy.json'
$env:YOLOE_RECYCLABLE_SETTINGS=Join-Path $trainedDeployment 'settings.json'
$env:YOLOE_RECYCLABLE_CALIBRATION=Join-Path $trainedDeployment 'calibration/deployment.json'
$env:YOLOE_RECYCLABLE_URL='http://127.0.0.1:'+ $DetectorPort
function Quote-TrainedPath([string]$Value) { '"'+$Value+'"' }
if (-not $ConsoleOnly) {
    $trainedPython=Join-Path $trainedRoot $(if($Backend -eq 'rocm'){'.venv-rocm/Scripts/python.exe'}else{'local-only/yoloe-deployment/venv/Scripts/python.exe'})
    $trainedDevice=if($Backend -eq 'rocm'){'cuda:0'}else{'cpu'}
    $trainedArguments=@('-u','-m','src.yoloe_trained26_monitor','--port',$DetectorPort,'--device',$trainedDevice,
        '--weights',(Quote-TrainedPath (Join-Path $trainedRoot $trainedRecord.weights)),
        '--policy',(Quote-TrainedPath $env:YOLOE_RECYCLABLE_POLICY),
        '--filter',(Quote-TrainedPath (Join-Path $trainedDeployment 'station-filter.json')),
        '--settings',(Quote-TrainedPath $env:YOLOE_RECYCLABLE_SETTINGS))
    $trainedProcess=Start-Process -FilePath $trainedPython -ArgumentList $trainedArguments `
        -WorkingDirectory (Join-Path $trainedRoot 'pynq_km1_vision') -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $trainedDeployment 'detector.stdout.log') `
        -RedirectStandardError (Join-Path $trainedDeployment 'detector.stderr.log')
    Write-Output ('Detector launcher PID: '+$trainedProcess.Id)
}
if (-not $DetectorOnly) {
    $consolePython=Join-Path $trainedRoot '.venv/Scripts/python.exe'
    $consoleProcess=Start-Process -FilePath $consolePython -ArgumentList @('-u','-m','src.sorting_prompted_console',
        '--port',$ConsolePort,'--upstream',$env:YOLOE_RECYCLABLE_URL) `
        -WorkingDirectory (Join-Path $trainedRoot 'pynq_km1_vision') -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $trainedDeployment 'console.stdout.log') `
        -RedirectStandardError (Join-Path $trainedDeployment 'console.stderr.log')
    Write-Output ('Console launcher PID: '+$consoleProcess.Id)
}
