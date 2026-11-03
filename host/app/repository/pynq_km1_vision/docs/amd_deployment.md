# Ryzen AI 9 HX 370：Windows 11正式上位机部署

**当前任务入口：**正常/变形瓶盖两类，模型目录为 `data/models/cap_deformation`；六状态保留为后续扩展。下文通用 `host_rocm.json` 指向 `data/models/best.pt`，使用前须核对本轮模型路径和标签，不能直接当作两类专用配置。

更新：2026-10-16。用户确认 AMD AI PC 已到位并接替原拯救者 Y9000P。实机为 ASUS Zenbook S 16 UM5606WA、Ryzen AI 9 HX 370、Radeon 890M、32 GB、Windows 11 25H2，已有独立 ROCm 环境及基础 GPU/随机权重 YOLO 检查，详见[安装记录](../../docs/team/reviews/2026-10-16-hx370-environment.md)。驱动已更新为 32.0.31021.5001 并通过 GPU 复验，本队权重、真实图片推理与板端通信尚待验收。以下安装步骤供复现，已有环境优先复用，不重复安装。

当前仓库位于 `C:\FPGA`，新环境在仓库根目录 `.venv-rocm`。运行项目模块时进入 `pynq_km1_vision/` 并调用 `..\.venv-rocm\Scripts\python.exe`；下文旧独立工程路径及 `.\.venv-rocm` 命令按实际布局调整。

## 1. 平台选择和验收条件

用户提供的[AMD原始指南](<../../选题指南/全国大学生嵌入式芯片与系统设计竞赛'2026FPGA赛道选题指南-AMD.pdf>)物理第13～15页要求AMD Ryzen AI PC、真实FPGA功能和ROCm 7.14.0及以上；核对的平台限制条款没有指定必须Ubuntu。选择Windows 11后仍须提供真实ROCm推理和设备记录。发布包若不携带原始指南，README应另记录赛方公开来源。

AMD [ROCm 7.14.0兼容矩阵](https://rocm.docs.amd.com/en/docs-7.14.0/compatibility/compatibility-matrix.html)的Ryzen表列出Windows 11 25H2、Adrenalin 26.6.4及OEM驱动26.10.28。先在页面选实际APU，核对整机/OEM适用驱动，两种驱动不是要求同时安装。该组合是官方部署起点，尚非本机验证结果。

旧radeon-ryzen文档只覆盖到7.2.1，不能据此写成满足比赛要求。最终服务为 **YOLO11n + PyTorch ROCm GPU推理，FP32，config/host_rocm.json**。代码使用half=False，没有启用NPU或FP16。

如果借机已带赛方验证环境，先保存原环境并做第5节检查，通过则复用；不为了和示例版本完全相同而重装。Ubuntu只在Windows路线实际受阻、确认解决方案后考虑，不默认换系统。

## 2. 到手先记录，暂不改驱动

在HX 370打开PowerShell：

```powershell
Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,TotalPhysicalMemory
Get-CimInstance Win32_Processor | Select-Object Name
Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion
Get-ComputerInfo | Select-Object WindowsProductName,WindowsVersion,OsName,OsVersion
winver
py -0p
```

1. CPU名称须核实为Ryzen AI 9 HX 370；型号不同不继续套用gfx1150命令。
2. 计划机型对应Radeon 890M/gfx1150路线，实机GPU和官方选择器再次核对。设备管理器内部驱动号与Adrenalin发行号不同，两者分别记录。
3. 用winver确认系统版本，按当前支持组合准备25H2；已有赛方环境先核对其说明并保留恢复资料。
4. 当前 AI PC 已有协作仓库和独立环境，继续使用现有工作目录。补齐本队模型、数据及配对位流时核对来源和哈希，不复制原 Y9000P 虚拟环境。
5. 建立 metrics/evidence/hx370_environment/，保存输出、截图及安装来源，不记录登录口令。

## 3. 驱动、Python和独立环境

1. 打开[兼容矩阵](https://rocm.docs.amd.com/en/docs-7.14.0/compatibility/compatibility-matrix.html)，选择AMD Ryzen、准确APU、Windows，核对系统/驱动。
2. 本机已按安装记录完成驱动更新和 GPU 复验。首次配置或恢复时通过该页或整机厂商支持页取得对应驱动；若安装器提示 Windows 待重启，先完成更新并重启。
3. 打开[ROCm 7.14.0安装选择器](https://rocm.docs.amd.com/en/docs-7.14.0/install/rocm.html)，选Ryzen、实际APU、Windows、pip，检查前置条件。页面有多个硬件分支，只执行所选分支。
4. 打开[PyTorch安装选择器](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)，选ROCm 7.14.0、Ryzen、HX 370、Windows、pip，确认Python有对应wheel。以下使用Python 3.12 x64与官方列出的PyTorch 2.12.0组合。
5. 用Python官方安装器准备3.12 x64，先执行 py -3.12 --version，再在HX 370运行：

```powershell
Set-Location C:\fpga\pynq_km1_vision
py -3.12 -m venv .venv-rocm
.\.venv-rocm\Scripts\python.exe -m pip install --upgrade pip
New-Item -ItemType Directory -Force metrics\evidence\hx370_environment
```

若已有.venv-rocm，先检查用途/版本，不覆盖已验收环境。下文显式使用该解释器，无须依赖环境激活。

## 4. 安装ROCm与项目依赖

下列gfx1150命令于2026-10-07核对AMD页面，整理为PowerShell单行；**仅在第2～3节确认实机对应组合后执行**。项目无音频任务，不安装torchaudio。

```powershell
.\.venv-rocm\Scripts\python.exe -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "rocm[libraries,device-gfx1150]==7.14.0"
.\.venv-rocm\Scripts\python.exe -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[device-gfx1150]==2.12.0+rocm7.14.0" "torchvision[device-gfx1150]==0.27.0+rocm7.14.0"
.\.venv-rocm\Scripts\python.exe -m pip install -r requirements_yolo.txt
.\.venv-rocm\Scripts\python.exe -m pip check
```

安装源：[ROCm pip步骤](https://rocm.docs.amd.com/en/docs-7.14.0/install/rocm.html)、[PyTorch步骤](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)。找不到发行版时查Python位数/版本、Windows版本、包源和硬件选择，不去掉版本号装普通PyPI torch，不执行CUDA安装脚本。

项目固定Ultralytics与上述PyTorch组合仍需实机验证。pip check只核对依赖声明，不能证明算子和本队模型可运行。保留异常栈和安装记录，针对错误修复，不无差别升级所有包。

## 5. 验证真实GPU运算

```powershell
.\.venv-rocm\Scripts\python.exe -c "import torch; print('torch',torch.__version__,'hip',torch.version.hip,'cuda_build',torch.version.cuda); assert torch.version.hip and torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); x=torch.randn(256,256,device='cuda'); y=x@x; torch.cuda.synchronize(); assert torch.isfinite(y).all(); print('GPU tensor pass',y.shape)"
.\.venv-rocm\Scripts\python.exe -c "import torch,torchvision; from torchvision.ops import nms; b=torch.tensor([[0,0,10,10],[1,1,9,9]],dtype=torch.float32,device='cuda'); s=torch.tensor([0.9,0.8],device='cuda'); print('GPU NMS',nms(b,s,0.5).cpu().tolist()); torch.cuda.synchronize()"
.\.venv-rocm\Scripts\python.exe -m pip show rocm torch torchvision
```

预期HIP非空、GPU名称正确、矩阵运算通过、NMS返回[0]。ROCm沿用torch.cuda API，名称不表示NVIDIA。仅is_available=True不够，第6节还须通过真实模型。

| 失败 | 先检查 |
|---|---|
| HIP为空 | 解释器错误，或装了CPU/CUDA torch |
| GPU不可用 | 系统构建、驱动、重启、APU型号 |
| 矩阵/NMS报错 | torch与torchvision配套版本、安装完整性 |
| FP32模型失败 | 当前代码确实使用FP32；只测FP16成功不算本项目通过，针对性修复并复测 |

## 6. 复制模型与同数据评估

复制 data/models/best.pt、best.onnx、model.json、data/defect_detection/和defect_dataset.yaml。复制前后比对best.pt的SHA256。把defect_dataset.yaml改为HX 370实际数据路径。

```powershell
Get-FileHash -Algorithm SHA256 data\models\best.pt
.\.venv-rocm\Scripts\python.exe -m src.evaluate_yolo --config config/host_rocm.json --data data/defect_dataset.yaml --images data/defect_detection/images/test --repeat 5 --output metrics/logs/yolo_hx370_run1.json
```

应生成评估JSON，六状态正确、后端为ROCm。文件名使用新run编号。与Y9000P比较时输入、模型、尺寸相同，保存预处理/推理/后处理/总耗时。模型质量以独立图像和实物联调确认，不预先承诺速度和精度。

## 7. 启动服务并连接板卡

1. 当前上位机为 AI PC。确认其物理网口（或 USB 以太网适配器）与 PYNQ 直连，按开工手册3.2核对192.168.2.1/24、网关/DNS留空；板保持192.168.2.99，Wi-Fi仅用于下载。同一条板卡链路只使用一台上位机。
2. 复用本队KM1_TOKEN，或重新生成随机口令并同步两端；不写入项目文件/提交包。
3. 按开工手册9.0建立限制真实网卡、本机192.168.2.1、远端192.168.2.99的TCP 8765规则；修改示例中的网卡名以匹配HX 370。直连可能被识别为公用网络，规则须覆盖当前网络类型。
4. 确认HX 370直连网口已为192.168.2.1，使用ROCm环境启动，保留终端：

```powershell
$env:KM1_TOKEN = "替换为本队保存的至少24字符随机口令"
.\.venv-rocm\Scripts\python.exe -m src.detect_server --config config/host_rocm.json --host 192.168.2.1 --port 8765
```

5. 板端SSH执行：

```bash
cd /home/xilinx/pynq_km1_vision
export KM1_TOKEN='替换为与HX370完全相同的口令'
curl -H "Authorization: Bearer $KM1_TOKEN" http://192.168.2.1:8765/health
```

预期protocol=2、task=detect、六状态标签、providers=["rocm_pytorch"]。板端config.json的inference.url沿用http://192.168.2.1:8765，换机不用改为Wi-Fi地址。若固定expected_model_id/runtime_id，按此次验收响应更新，不能保留CUDA运行配置的runtime_id。再按开工手册第9节发送真实图片，health不能代替图片通信验收。

## 8. 正式闭环、离线与存档

1. 按开工手册第14节先只规划，核对迁移后的通信延迟、目标关联和窗口。
2. 已验收路线进行单件及批次，重测正式成绩；不把Y9000P结果改名为HX 370成绩。
3. 保留直连网线，关闭Internet连接共享并断开HX 370的Wi-Fi，确认健康检查、真实图像请求和冷启动任务均不在线下载模型；正式测量在HX 370重做。
4. 记录交流供电、性能/电源模式，演示期间避免睡眠；赛前不临时升级驱动。
5. 保存环境清单，Python写UTF-8：

```powershell
.\.venv-rocm\Scripts\python.exe -c "from pathlib import Path; import subprocess,sys; p=Path('metrics/evidence/hx370_environment'); p.mkdir(parents=True,exist_ok=True); p.joinpath('pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')"
```

同时保存官方安装命令、驱动文件名/来源、整机信息、模型摘要、评估JSON和批次证据。freeze不能独自重建AMD自定义包，还需对应安装源和匹配安装文件。

完成标准：实机型号/环境、HIP/GPU/NMS、本队YOLO、真实网络请求、物理闭环、离线冷启动均通过。未通过项如实记录。

## 9. Windows实际受阻时

保留最小复现、驱动/包版本和错误，核对赛方或设备提供方环境。本文不自动切换Ubuntu。确认需要调整后才按AMD选择器选择HX 370支持的Ubuntu组合，并重做相同测试；不以NPU provider或CPU回退代替正式ROCm验收。

