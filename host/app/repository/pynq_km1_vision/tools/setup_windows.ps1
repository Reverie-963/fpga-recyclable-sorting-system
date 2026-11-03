param([ValidateSet("cpu", "cuda")][string]$Backend = "cuda")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
py -3.11 -m venv .venv-yolo
if ($LASTEXITCODE -ne 0) { throw "Install Python 3.11 x64 first" }
$Python = ".\.venv-yolo\Scripts\python.exe"
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip failed" }
# CUDA 12.6 build matches the current RTX 4070 Laptop / 560.94 driver.
$Index = if ($Backend -eq "cuda") { "https://download.pytorch.org/whl/cu126" } else { "https://download.pytorch.org/whl/cpu" }
& $Python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url $Index
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }
& $Python -m pip install -r requirements_yolo.txt
if ($LASTEXITCODE -ne 0) { throw "YOLO dependencies failed" }
& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency conflict" }
if ($Backend -eq "cuda") {
    & $Python -c "import torch,torchvision; assert torch.cuda.is_available(), 'CUDA unavailable'; x=torch.randn(1,3,32,32,device='cuda',requires_grad=True); net=torch.nn.Conv2d(3,8,3).cuda(); net(x).square().mean().backward(); assert torch.isfinite(x.grad).all().item(); boxes=torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]],device='cuda'); scores=torch.tensor([.9,.8],device='cuda'); keep=torchvision.ops.nms(boxes,scores,.5); assert keep.tolist()==[0]; torch.cuda.synchronize(); print('GPU forward/backward and NMS: PASS'); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw "GPU execution check failed" }
}
& $Python -m src.doctor
if ($LASTEXITCODE -ne 0) { throw "Diagnostics failed" }
Write-Host "Dependencies checked. Prepare labeled data, train best.pt, then validate the model and detection service."
