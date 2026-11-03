# Windows操作手册  轻质零件外观缺陷检测与动态剔除

**当前任务入口：**正常/变形瓶盖两类，按[开工手册8.0](commissioning_roadmap.md)与AMD部署手册推进。六状态为后续扩展，扩展时再按第8.2节迁移配置；末端直行仍需复核。

方案v6更新，2026-10-07。控制器代码仍复用v5。本文是最终参赛系统的开发与复现流程。架构、目标和完整任务见 [最终参赛方案](final_design.md)。主程序复用YOLO多目标检测与输送带预测拦截；新任务类别必须先迁移，旧分类器仅作对照。

**状态更新 2026-10-16：**AI PC 已到位并成为当前上位机，原拯救者 Y9000P 已退出使用。当前先做正常/变形瓶盖两类；HX 370 已完成驱动更新和 ROCm 基础运算复验，本队模型及板端通信待完成。新机状态以[团队状态](../../docs/team/STATUS.md)和[AMD部署手册](amd_deployment.md)为准。

**旧机历史环境记录：**2026-10-07 Y9000P 的 Python/CUDA、GPU/NMS 和随机初始化 YOLO 检查见[历史记录](../metrics/evidence/y9000p_environment/gpu_smoke_check.json)，不代表当前 AI PC 验收。

## 1. 设备职责与文件入口

当前 AMD AI PC 通过 ROCm/PyTorch 运行 YOLO 服务，本队模型仍待迁入验收；NVIDIA/CUDA 训练和服务仅保留历史复现步骤。PYNQ-Z2连接USB相机和KM1 USB控制器，PL负责掩码和编码器，PS负责跟踪、标定与动作调度；PC不暴露运动接口。

| 入口 | 在哪里运行 |
|---|---|
| tools/setup_windows.ps1 | Windows安装YOLO依赖 |
| src.train_yolo / src.evaluate_yolo | Windows训练与模型评估，后者也用于AMD实机 |
| src.detect_server | 上位机YOLO检测服务，协议v2 |
| src.run / src.recycling_run | PYNQ动态拦截任务控制 |
| src.capture / src.calibrate / src.arm_cli | PYNQ采图、标定与机械臂点动 |
| src.benchmark | PYNQ上CPU与PL逐像素/耗时对照 |
| tools/build_windows.ps1 | Windows x86-64 AMD工具环境 |

将工程解压至 `C:\fpga\pynq_km1_vision`。所有PowerShell命令从此目录运行。Bash命令在PYNQ SSH中运行。JSON使用UTF-8，允许BOM；不要用PowerShell 5默认UTF-16输出写入配置文件。

## 2. Windows NVIDIA环境（历史复现）

当前 AI PC 使用 [AMD部署手册](amd_deployment.md)及[已有安装记录](../../docs/team/reviews/2026-10-16-hx370-environment.md)，跳过本节 CUDA 安装。下文 `.venv-yolo`、`host_cuda.json` 等为旧机示例，不能直接用于当前 AI PC；须使用根目录 `.venv-rocm` 和匹配本队两类权重的 ROCm 配置。

安装Python 3.11 x64、NVIDIA驱动与OpenSSH客户端。2026-10-07已实测为RTX 4070 Laptop GPU、8188MiB显存、驱动560.94。采用PyTorch 2.7.1/torchvision 0.22.1的cu126发行包；GPU矩阵运算、卷积反向、CUDA NMS及YOLO11n前向已实测通过；六状态缺陷模型仍待训练。

```powershell
cd C:\fpga\pynq_km1_vision
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
py -3.11 -m venv .venv-yolo
.\.venv-yolo\Scripts\python.exe -m pip install --upgrade pip
.\.venv-yolo\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv-yolo\Scripts\python.exe -m pip install -r requirements_yolo.txt
.\.venv-yolo\Scripts\python.exe -m pip check
.\.venv-yolo\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))"
.\.venv-yolo\Scripts\python.exe -m src.doctor
```

等效入口：`powershell -File tools/setup_windows.ps1 -Backend cuda`。策略不允许脚本时执行上述逐条命令，不必改全局策略。这个环境采用opencv-python，不与旧headless或旧ORT CUDA环境混装。最终服务直接加载PyTorch `.pt`；ONNX作为模型交付和后续部署产物，当前服务不将其误当分类模型加载。

## 3. 硬件与接线

按 [硬件手册](hardware_setup.md) 完成供电、PYNQ镜像和相机固定。KM1使用原厂8V 5A适配器，按手册先上适配器电源再接USB；PYNQ独立合规供电。USB Hub连接相机和KM1，PYNQ网口直连当前 AI PC 的物理以太网：PC为192.168.2.1/24，PYNQ为192.168.2.99/24；PC有线网关/DNS留空，Wi-Fi负责下载。设置和排障见开工手册3.2；板端缺依赖时按4.3.1处理，电脑能上网不代表板端能上网。不要同时保留OpenMV发送运动命令。

相机保持俯视固定，照明和背景固定；待抓区与分类盒图像ROI互不重叠，分类盒保留可见空位。现场确认断电措施。软件无法撤销已发到ESP32的运动。

PYNQ镜像使用其自带PYNQ/NumPy/OpenCV，缺少serial时按镜像方式补齐。不在32位ARM板上安装YOLO或PC依赖。

```powershell
ssh xilinx@192.168.2.99 "mkdir -p /home/xilinx/pynq_km1_vision"
scp -r src config tools sim build docs data board report metrics skill config.json requirements.txt README.md xilinx@192.168.2.99:/home/xilinx/pynq_km1_vision/
```

IP为本工程固定直连配置，账号以实际镜像为准；首次上传不覆盖已有现场标定配置，不上传Windows虚拟环境。PYNQ终端：

```bash
cd /home/xilinx/pynq_km1_vision
python3 -m src.doctor
python3 -m src.capture --count 20 --interval 2
```

`config.json` 是板端唯一工作配置，`config/contest_board.json`是旧v5样例；新缺陷方案使用`config/defect_board.json`，先按开工手册8.2备份迁移，再以根config.json作为唯一工作配置。相机分辨率与标定一致。根配置默认PL融合核，动态主程序即使设置 `vision.backend="cpu"` 也需要PL编码器位流；该设置仅将图像路径切换为CPU对照。没有位流时可运行软件测试，不能运行动态硬件控制入口。

## 4. 数据集与YOLO11n训练

采集完整工作区图像，标出每件物体的检测框与类别。多物体、不同位置/朝向、邻近、遮挡和空场景均应覆盖。按不同采集场次/物体实例划分train/val/test，避免相邻视频帧泄漏。旧64×64裁剪分类数据不能直接当检测数据集。

```text
data/defect_detection/
  images/train/  images/val/  images/test/
  labels/train/  labels/val/  labels/test/
```

每张图片有同名TXT，每行 `class_id center_x center_y width height`，四个坐标按原图宽高归一化。真正的背景图片可用空标签；不能把漏标物体当背景。编辑 `data/defect_dataset.yaml` 的绝对path和names；六状态固定为cap_ok、cap_notch、cap_deformed、plate_ok、plate_corner_missing、plate_hole_missing。训练与模型包含全部6类；arm.bins和verification.bin_rois仅含4种缺陷键，正常类不抓。未识别/漏剔仍可能直行，不能将末端直接称为已验证合格出口。样件标准、标签ID和迁移步骤见开工手册第8节。

将PYNQ完整图片取回Windows：

```powershell
scp -r xilinx@192.168.2.99:/home/xilinx/pynq_km1_vision/data/frames .\data\
.\.venv-yolo\Scripts\python.exe -m src.train_yolo --data data/defect_dataset.yaml --weights yolo11n.pt --device 0 --epochs 100 --batch 8 --imgsz 640
```

官方预训练权重在首次训练时下载，训练产物为 `data/models/best.pt`、`best.onnx`、`model.json`。数据来自本队工位，COCO预训练不能证明已认识本队回收物。batch=8是配置起点，按实机显存调整；改变输入尺寸要同步host配置，并记录性能比较条件。

脚本用val选择best，随后test独立评估并导出ONNX。test不得再用于反复调参数。训练输出、随机种子与原始标注保存。

```powershell
.\.venv-yolo\Scripts\python.exe -m src.evaluate_yolo --config config/host_cuda.json --data data/defect_dataset.yaml --images data/defect_detection/images/test --repeat 5 --output metrics/logs/yolo_cuda.json
```

输出mAP/Precision/Recall、batch1预处理/推理/后处理/总耗时分位数，以及包含文件读取的持续吞吐。首次预热不计入稳定测量；同名报告拒绝覆盖。最终AMD机器重跑相同数据与模型。

## 5. 检测服务与通信

```powershell
.\.venv-yolo\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
$env:KM1_TOKEN = "复制生成的口令"
.\.venv-yolo\Scripts\python.exe -m src.detect_server --config config/host_cuda.json --host 192.168.2.1 --port 8765
```

绑定直连网卡192.168.2.1，防火墙按开工手册9.0只允许板卡192.168.2.99访问TCP 8765；直连常被识别为公用网络，不能仅启用专用网络规则。服务在开放端口前预热模型，配置cuda却不能使用NVIDIA GPU时直接失败。正式AMD用host_rocm.json并检查torch.version.hip，不能把CUDA成功当ROCm成功。

PYNQ设置同一口令，根config.json及板端样例的inference.url已统一为http://192.168.2.1:8765；现有板端副本只更新该字段，不覆盖标定。执行：

```bash
export KM1_TOKEN='与PC完全相同的口令'
curl -H "Authorization: Bearer $KM1_TOKEN" http://192.168.2.1:8765/health
```

health必须返回protocol=2、task=detect和实际后端。端点为 `/detect`，整帧PNG上传，返回原图坐标xyxy检测框。不是旧 `/infer` 裁剪分类服务。可把model_id/runtime_id固定到板端expected字段；为空则首个响应锁定，批次内变更会停止。

## 6. PL构建与掩码检查

统一使用 **Vivado 2026.1 + Vitis HLS 2026.1**。本机程序已在 `D:\work\AMDDesignTools\2026.1`，普通 PowerShell 可执行 `./tools/build_windows.ps1 -CheckOnly` 自动定位并检查；其他目录通过 `-ToolRoot` 指定。后续确认Zynq-7000器件支持、准备TUL PYNQ-Z2 board files（芯片xc7z020clg400-1）及纯英文短路径工程，再执行构建，见 [2026.1 环境说明](vivado_2026_1.md)。脚本会拒绝其他工具版本，实际综合与上板验证仍需单独完成。

```powershell
.\tools\build_windows.ps1 -BoardRepo C:\fpga\board_files -Kernel segment_open -EncoderXdc C:\fpga\encoder_pins.xdc
scp board/vision.bit board/vision.hwh xilinx@192.168.2.99:/home/xilinx/pynq_km1_vision/board/
```

必须同次构建bit/hwh。HLS执行C仿真、综合和协同仿真，Vivado实现后读取timing/utilization；目标100MHz、II=1不是已经实现的成绩。

```bash
python3 -m src.benchmark --config config.json --pl --images data/frames --repeat 20
```

CPU参考与PL融合核采用一致零边界，逐像素必须一致。计时包含打包、DMA、等待和输出转换。分别保存报告，不删掉DMA制造加速。`vision.backend="pl"` 与 `pl_kernel="open"` 同时配置。

PL掩码用于YOLO框内的独立连通域确认、内部候选抓取点、动作前复查及放置后验证。调好background_bgr、threshold、min_area/max_area；颜色接近底板或连成一团的回收物可被拒绝，不能靠调低全部门槛强行执行。

## 7. 输送带、编码器与动态标定

完整接线、RTL寄存器、时间标定和配置说明见 [输送带部署](conveyor_setup.md)。固定相机覆盖观察区、抓取窗口与分类盒，外接正交编码器经正确电平转换接PYNQ；电机独立调速供电。v5构建必须提供EncoderXdc，旧位流不含编码器IP。

使用tools/point_picker.html取得像素点；采至少6个拟合点和3个独立检查点，填写data/calibration_pairs.json：

```bash
python3 -m src.calibrate data/calibration_pairs.json
```

按真实样品高度标定，配置机械臂行程、观察凸包、输送方向、抓取窗口、四种缺陷状态的放置坐标和盒ROI。标定凸包可覆盖上游观察区，动态抓取窗口必须处于机械臂可达区。必须测得编码器毫米/计数、UVC采集时延及误差、夹爪接触时间和波动、带速与加速度范围。

`arm.commissioned`与`dynamic.commissioned`默认false，现场验收后再启用。默认带速1～8mm/s是配置占位，不是已测速度范围。图像融合不再要求物体静止，编号由编码器补偿后的一对一关联维持。

## 8. 最终动态任务运行

```bash
# 真实采图、编码器与推理，生成窗口内的预测计划，不开机械臂串口
python3 -m src.run --config config.json --frames 3000
# 现场验收后，执行有上限的连续输送分拣批次
python3 -m src.run --config config.json --execute --max-tasks 20 --frames 3000
```

查看board/last_plan.json中的track_id、预测接触点、接触时刻与误差预算。类别由异步YOLO返回并绑定原帧编号，位置由板端掩码和编码器持续更新；无可行窗口则继续观察。机械臂忙碌时允许来料流向末端收集区，这些物体必须计入整机回收率分母。

下降前检查目标预测位置，闭爪前检查位移漂移和时限；超时、歧义、编码器异常或串口错误停止后续指令。分类盒前景增加作为程序完成证据；物体随带离开原位置不作为抓取成功证据。证据不足停止批次，不自动重抓。真实成功率另用录像核验。

## 9. 指标、原始证据与AMD部署

`board/events.jsonl` 保存跟踪编号、编码器、异步检测、预测接触点、误差预算与停止原因；每项任务保留 `board/evidence/<task_id>/before.png` 和after.png。task_completed仅表示通过分类盒图像证据，人工录像真值另统计。

```bash
python3 -m src.summarize --output report/session_summary.json
python3 metrics/scripts/measure_rtt.py --url http://192.168.2.1:8765 --count 100 --output metrics/logs/health_rtt_run1.csv
```

五类必报和两类专项见 [指标说明](../metrics/README.md)。总物理响应由同一外部时基测量，不用软件frame_age或RPC替代。最终赛机安装与验收见 [AMD部署](amd_deployment.md)。

## 10. 问题定位

| 现象 | 检查 |
|---|---|
| 有框却没拦截计划 | 前景掩码、框覆盖、第二框争用、邻近间隙、标定凸包、类别有效期、编码器速度、预测窗口与误差预算 |
| 缺少best.pt | 完成自有检测数据训练，部署相同模型与model.json |
| protocol错误 | 必须运行detect_server及host配置，不能混用旧pc_server |
| Changed model/runtime | 固定配置后重启整个批次，不热切换模型继续动作 |
| bin ROI错误 | 每类都有实际测量ROI，ROI在画面内且不与待抓凸包重叠 |
| needs_review | 检查原物体是否移走、是否放入正确盒、遮挡/盒满；人工检查后重启 |
| DMA超时 | bit/hwh配对、open核、TLAST、23位长度寄存器、时钟复位与缓存 |
| Windows安装冲突 | 使用独立.venv-yolo，不混旧headless/ORT环境，检查pip check |

旧静态src.contest_run配config/static_board.json，以及岭回归/CNN仅作历史对照，入口src.baseline_run及config/baseline_local.json；不再作为参赛系统默认流程。
