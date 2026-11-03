# 识别链路说明（队员 A：模型训练 · 识别评估 · 推理部署）

> 原稿：队员 A；2026-10-17 按当前项目状态修订。本文保留识别侧交接用途，不新增永久分工。
> 目标：说明正常/变形瓶盖两类识别的训练、评估、部署与 PYNQ 接口。六状态保留为后续扩展。
> 证据边界：原稿记载的 Windows / CUDA 链路验证属于历史开发记录，本次未复测。当前 HX 370 / ROCm 基础检查已通过，本队模型迁入、真实图片推理和板端通信仍待验收，见[环境记录](../../docs/team/reviews/2026-10-16-hx370-environment.md)。通用预训练权重的运行不能证明本队缺陷识别效果。

---

## 1. 识别链路概览

本系统的识别职责由 **AMD AI PC（正式：PyTorch + ROCm）**承担，产出整件零件的缺陷状态/框/置信度。FPGA(PL)只做像素级分割与编码器计数，不做 YOLO 推理。

```
数据采集标注
   │
   ▼
训练(train_yolo) ──► best.pt + ONNX + model.json
   │
   ▼
评估(evaluate_yolo) ──► mAP / Precision / Recall / 分阶段计时
   │
   ▼
推理核心(detection.YOLODetector) ──► predict(frame) → detections + timing
   │
   ▼
服务部署(detect_server) ──► HTTP 协议 v2 ──► 板端(PYNQ)按帧 POST 调用
```

识别侧向 PYNQ 返回 `YOLODetector.predict()` 的结果：`detections`（label + confidence + xyxy）、`timing`、模型/运行指纹、后端及图像尺寸。HTTP 服务再添加协议和请求标识；AI PC 不直接接收或发送机械动作命令。PS 负责几何、标定、跟踪和动作调度。

## 2. 当前环境与运行目录

复用仓库根目录 `.venv-rocm`；当前 HX 370 / Radeon 890M 的 Python 3.12.10、ROCm 7.14.0、PyTorch 2.12.0 环境和验证边界见上述环境记录。首次安装或恢复按 [AMD 部署手册](amd_deployment.md)，不要向当前环境混装 CUDA 版 PyTorch。根目录 Python 3.11 轻量测试环境与 GPU 环境分开。

以下命令均为 **PowerShell**。先从仓库根目录进入工程目录；后续模块命令均在 `pynq_km1_vision/` 执行：

```powershell
Set-Location pynq_km1_vision
& ..\.venv-rocm\Scripts\python.exe -c "import torch; print(torch.version.hip, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU unavailable')"
```

ROCm 版 PyTorch 同样使用 `torch.cuda` 接口；需同时核对 HIP 版本及实际设备。GPU 基础推理检查不代表本队模型训练或正式性能已验收。

## 3. 数据集格式

识别数据为 YOLO 检测格式，配置见 `data/cap_deformation.yaml`：

```yaml
path: <本机绝对路径>/data/cap_deformation   # 训练前改为本机实际路径
train: images/train
val:   images/val
test:  images/test
names:
  0: cap_ok        # 正常瓶盖（不抓）
  1: cap_deformed  # 变形瓶盖（抓取候选）
```

- 标注为 YOLO `.txt`：每行 `类别 x_center y_center w h`（归一化），与 `data/cap_deformation/labels/` 结构一致
- 划分原则：**按样件/采集场次划分**，同一瓶盖的照片不得同时出现在训练与测试（避免泄漏导致指标虚高）
- 规范提醒：正常与缺陷样件应覆盖多种光照、位置、变形程度，避免模型只学颜色或批次

## 4. 训练

入口：`src/train_yolo.py`，微调 YOLO11n，产出 `best.pt` + ONNX + `model.json`（含标签、SHA256 指纹、test 指标）。

```powershell
& ..\.venv-rocm\Scripts\python.exe -m src.train_yolo --data data/cap_deformation.yaml --weights yolo11n.pt --device 0 --epochs 100 --batch 8 --imgsz 640 --output-dir data/models/cap_deformation
```

产出：
- `data/models/cap_deformation/best.pt`（部署用权重）
- `data/models/cap_deformation/best.onnx`（导出 ONNX）
- `data/models/cap_deformation/model.json`（架构、标签、test 指标、权重 SHA256）

> [团队状态](../../docs/team/STATUS.md)已有历史样本、模型目录和训练记录。克隆中缺少 `data/models/` 不代表团队从未训练；先按[资产规则](../../docs/team/ASSETS.md)领取本队权重并核对标签、来源和 SHA256，没有共享地址的资产仍只在持有人本机。需要重训时先备份已有产物。`yolo11n.pt` 是初始化权重，不是本队两类成品模型；当前 GPU 训练能力须另验。

调参使用验证集。训练入口会评估 test 划分，不能反复据该结果调参后仍把它当作独立验收成绩。

## 5. 评估

入口：`src/evaluate_yolo.py`，对**未参与训练的 test 集**评估检测质量与单帧分阶段计时。

当前从 [host_rocm.json](../config/host_rocm.json) 生成独立的本地两类配置并修正模型路径；原 `host_cap_deformation_cuda.json` 仅适用于 CUDA。以下文件由 `.gitignore` 排除，不提交 Git：

```powershell
$recognitionConfigPath = 'config/host_cap_deformation_rocm.local.json'
if (Test-Path $recognitionConfigPath) { throw '本地配置已存在，请先核对现有内容，不要覆盖。' }
$recognitionConfig = Get-Content config/host_rocm.json -Raw | ConvertFrom-Json
$recognitionConfig.model = 'data/models/cap_deformation/best.pt'
$recognitionConfig | ConvertTo-Json | Set-Content -Encoding utf8 $recognitionConfigPath
```

核对配置中 `accelerator=rocm`、权重标签为 `cap_ok` / `cap_deformed`，并准备独立测试数据后执行。报告文件名每次新建，避免覆盖已有结果：

```powershell
$recognitionReportPath = 'metrics/logs/yolo_evaluation_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.json'
& ..\.venv-rocm\Scripts\python.exe -m src.evaluate_yolo --config config/host_cap_deformation_rocm.local.json --data data/cap_deformation.yaml --images data/cap_deformation/images/test --output $recognitionReportPath
```

产出：mAP、Precision、Recall + `preprocess/inference/postprocess/total` 各阶段 P50/P95/P99/max 耗时。
验收关注：检测报告 **mAP@0.5:0.95**；计时按单张图分别报预处理/模型/后处理/总时间（对应指南要求）。

## 6. 推理核心

`src/detection.py` 的 `YOLODetector` 是对外的唯一推理接口，做了严格的输入/环境校验。

关键配置（上一节生成的 `config/host_cap_deformation_rocm.local.json`）：
```json
{ "model": "data/models/cap_deformation/best.pt",
  "accelerator": "rocm", "device": 0,
  "imgsz": 640, "confidence": 0.25, "iou": 0.45, "max_det": 30 }
```

调用（输入 BGR 图，输出结构化结果）：
```python
from src.detection import YOLODetector
det = YOLODetector(cfg)           # 加载权重，若 best.pt 缺失会报错提示先训练
result = det.predict(frame)       # frame: (h, w, 3) uint8 BGR
```

`predict()` 返回结构（**虚构的格式示例，数值不是实测成绩**）。`cap_deformed` 必须来自具有本队对应标签的模型，不能用通用预训练权重的运行结果声称已识别缺陷：
```json
{
  "detections": [
    { "label": "cap_deformed", "confidence": 0.71, "xyxy": [17.4, 0.08, 209.9, 409.1] }
  ],
  "timing": { "preprocess_ms": 1.28, "inference_ms": 19.4, "postprocess_ms": 2.09, "total_ms": 23.1 },
  "model_id": "<best.pt 的 SHA256>",
  "runtime_id": "<配置的 SHA256>",
  "providers": ["rocm_pytorch"],
  "image_size": [640, 480]
}
```

字段含义：
- `detections[].label`：类别（正常/缺陷），下游据此决定**是否抓取**、放入哪个不良盒
- `detections[].confidence`：置信度；检出侧阈值 `confidence`、抓取侧门槛 `grasp.confidence`（见板端配置）
- `detections[].xyxy`：原图像素框 `[x1,y1,x2,y2]`，不是 letterbox 坐标或夹爪落点。PS 结合掩码内部点、标定与运动预测生成毫米坐标，不能直接把框中心当抓取点
- `model_id`：权重文件 SHA256 指纹，板端用它校验"模型未换"，防旧框配新模型
- `providers`：推理后端，当前 AMD 平台为 `rocm_pytorch`；`cuda_pytorch` 属于 CUDA 环境
- `runtime_id`：推理配置摘要；板端同时检查模型与运行指纹，预设或在首次响应锁定，拒绝批次内变化
- `image_size`：`[width,height]`，必须匹配请求帧；推理计时不包含板端采集、网络和机械动作

## 7. 服务部署（供板端调用）

`src/detect_server.py` 把 `YOLODetector` 封装成认证 HTTP 服务（协议 v2），板端(PYNQ)按帧 POST 调用。

先按 [AMD 部署手册](amd_deployment.md)核对实际网卡地址及防火墙，再在 AI PC 工程目录启动。口令与板端一致且至少 24 字符，不写入文件或提交：

```powershell
$env:KM1_TOKEN = '<替换为与板端一致且至少24字符的口令>'
& ..\.venv-rocm\Scripts\python.exe -m src.detect_server --config config/host_cap_deformation_rocm.local.json --host 192.168.2.1 --port 8765
```

监听地址必须是实际已配置的 AI PC 直连网卡地址。`127.0.0.1` 仅供电脑本机自测，PYNQ 无法访问该监听地址。项目目标为 AI PC `192.168.2.1`、PYNQ `192.168.2.99`，板端完整配置中的 `inference.url` 为 `http://192.168.2.1:8765`。首次联调先验证只读检测通信，不启用机械动作。

协议 v2：
- 认证：请求头 `Authorization: Bearer <KM1_TOKEN>`
- `GET /health`：返回协议/模型指纹/标签列表
- `POST /detect`：body 为 `{protocol:2, request_id, frame_id, png:<base64>}`；请求体最多 8,500,000 字节，图像宽高各不超过 1920、总像素不超过 `1920×1080`；返回 `{protocol, request_id, frame_id, ...detector.predict()}`

要点：
- `request_id`/`frame_id` 原样回显，供板端"对账"——确保结果对应正确的帧，避免旧框配新画面（异步语义核心）
- 最近 4096 个已接收请求 ID 去重，重复返回 409；客户端需生成唯一 ID
- 认证失败返回 401

## 8. 与 PYNQ、机械臂的交接

实时查看送检画面可按[AI PC 工位视觉监视](vision_monitor.md)给服务增加 `--monitor-port 8766`，在本机浏览器查看同帧检测框和类别。独立观察使用板端 `src.camera_observe`，无需运行机械控制器；检测协议与响应保持 v2。

当前已有代码契约如下，不再把“只给中心点还是完整框”作为未决定的接口。

| 层次 | 输入与职责 | 交接和待验收项 |
|---|---|---|
| AI PC 识别 | 整帧 → 原图框、类别、分数、指纹 | 两类权重/标签/哈希、独立评估、真实图片与通信日志 |
| PYNQ PS 几何 | PL 掩码 → 连通域/内部候选点 → 标定后的毫米坐标 | 固定相机、ROI、平面标定和误差；检测框中心不直接用作抓取点 |
| PYNQ PS 关联/预测 | 检测绑定送检帧几何与 track_id，结合编码器更新位置 | 帧龄、歧义拒绝、位移与时间标定、可达窗口与接触误差 |
| PYNQ PS 动作计划 | 类别筛选、逆运动学、逐步动作及期限检查 | 正常瓶盖不抓，变形瓶盖满足条件后才成为候选；对齐 `arm.bins` 和盒区 ROI |
| KM1 / OpenMV | 接收校验后的动作帧 | 关节限位、闭合与路径、桥延迟及实物结果；串口回执不证明抓取成功 |

实现入口：[grasp_geometry.py](../src/grasp_geometry.py)、[moving_targets.py](../src/moving_targets.py)、[recycling_run.py](../src/recycling_run.py)。PS 保留送检快照，不能将旧框直接套在新画面上；帧龄用 PS 的单调时钟，不直接比较两台机器的时钟。

PR #8 已合入 OpenMV 无视觉桥接和固定目标预演，见 [PYNQ 接口设计](../../docs/team/ARM_PYNQ_INTERFACE_DESIGN.md)与[机械臂审核入口](../../docs/team/MEMBER_B_ARM_REVIEW.md)。动态 OpenMV 执行仍关闭；V3、整臂、静态抓放和动态时序仍待实物验收。人员按当次任务协作，识别服务不直接驱动机械臂。

## 9. 当前状态与下一步

| 环节 | 状态 |
|---|---|
| AI PC / ROCm 基础环境 | 2026-10-16 GPU、NMS、随机权重 YOLO 检查通过；不代表本队模型效果或训练通过 |
| 历史 CUDA 链路 | 原稿记载已跑通，本次未复测；不能替代当前主机验收 |
| 本队样本与模型 | 历史本机有目录与训练记录，需领取并核对来源、标签和哈希 |
| 当前 AI PC 真实模型及独立评估 | 待迁入本队模型后验证，不能用通用权重或示例数值代替 |
| 当前主机与板端通信 | 待验收，模拟 HTTP 测试不等于实际链路通过 |
| 整机动态剔除 | 标定、编码器、机械时序与人工真值验证仍待完成 |

下一步先领取并核验两类模型和数据，再评估真实图片及板端通信；确需重训时再补数据、训练。统一工位标定和编码器链路后，按授权的分阶段机械实验推进。六状态、新对象和 NPU 不作为本轮前置条件。

## 10. 复现步骤（供他人照做）

1. 复用当前 ROCm 环境，在工程目录调用正确解释器。
2. 领取数据和模型，核对两类标签、样件划分、来源和哈希；确认 YAML 数据路径。
3. 需要时才训练；生成并核对本地 ROCm 配置，独立评估并保存失败样例及计时。
4. 核对网卡地址、认证和模型/运行指纹，验证真实图片与板端 HTTP 通信。
5. 交接模型和配置哈希、数据说明、评估报告与待测事项；机械实验另按项目授权流程执行。

本次修订仅改文档，没有运行训练、GPU 性能、板端/串口、时序、综合或机械动作，不增加新的模型或整机实测成绩。
