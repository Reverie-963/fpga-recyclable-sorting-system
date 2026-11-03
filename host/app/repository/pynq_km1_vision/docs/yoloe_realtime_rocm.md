# YOLOE ROCm GPU 部署与实时监控

2026-10-29，用户要求先部署 YOLOE 到 ROCm GPU，再改为实时监控。本轮按该顺序完成 GPU 单图验证、CPU/GPU 对照、相机接入及浏览器控制验证。当前入口仍为 `http://127.0.0.1:8778/`，默认塑料/金属/纸质三类参考图模式。

## 已运行的加速路径

- HX 370 的 Radeon 890M，Windows 本机 ROCm；复用根目录 `.venv-rocm`。
- PyTorch `2.12.0+rocm7.14.0`、HIP `7.14.60850`、Ultralytics `8.3.230`，FP16。
- 仍使用 YOLOE-11m 原先缓存的文字和视觉提示权重。配置、官方权重、派生权重及参考图哈希未变；没有重新训练或量化模型。
- `execution.device=cuda:0` 与 `tensor_device=cuda:0` 已在真实结果中验证。这是 PyTorch 对 AMD ROCm GPU 的接口名称，不表示使用 NVIDIA；`execution.hip` 和 `device_name` 同时报告真实后端。
- GPU 模式严格检查 ROCm 及固定版本，不允许静默回退到 CPU。`runtime_id` 现在包含设备、精度和后端，区别于此前 CPU 部署；模型哈希单独保持不变。

首次原生 GPU 推理遇到本机 MIOpen BatchNorm 动态编译缺 `type_traits`。沿用既有训练进程已经采用的兼容方式，仅在本服务进程设置 `torch.backends.cudnn.enabled=False`，使用原生 GPU 算子；没有修改驱动、PyTorch 源码或系统配置。该设置不等于切回 CPU，真实张量位置已检查。GPU 首次加载/预热可耗数秒，服务在完成两种模式的预热后才监听端口。

本轮没有启用 NPU、重写 FPGA 核、综合、烧写、连接串口或发送机械动作。FPGA/PS 相机链路只提供已经存在的只读图像源，本轮不声称 PL 在执行 YOLOE 网络。

## 启动与回退

先完成[原 YOLOE 资产准备](yoloe_materials_deployment.md)。复用已经通过检查的 ROCm 环境，不运行升级或安装命令。在根目录前台启动，Ctrl+C 可停止：

```powershell
.\tools\start_yoloe_preview.ps1 -Backend rocm
```

等效命令，适合在工程目录运行：

```powershell
..\.venv-rocm\Scripts\python.exe -u -m src.yoloe_preview_server --device cuda:0 --half --camera-url http://127.0.0.1:8780/frame.jpg
```

相机源是本机8780只读 JPEG 桥接，当前复用已连接 PYNQ `/dev/video0` 的既有进程。不会再次打开同一 USB 相机或改动板端脚本。8780 不可用时，页面明确显示断线并等待恢复；本启动脚本不会索取凭据或隐式连接另一台设备。桥接的私有连接配置继续留在本机，不进入 Git。

CPU 回退和纯图片检查必须显式选择，先停止正在占用8778的本服务再启动：

```powershell
.\tools\start_yoloe_preview.ps1 -Backend cpu
.\tools\start_yoloe_preview.ps1 -Backend rocm -ImagesOnly
```

旧 `setup_yoloe.ps1 -StartPreview` 仍是独立 CPU 图片入口。基础配置中的 cpu 是其历史默认值，实际运行后端由当前启动参数和响应中的 `execution` 字段确定，不应为切换 GPU 修改已封存的资产配置指纹。

## 实时页面及接口

页面默认显示实时相机、**对应被识别那一帧**的覆盖图、三类计数、识别更新帧率、推理用时及接收至结果帧龄。它们是不同指标：相机刷新率、网络推理吞吐和页面观察到的新结果率不能混称一个 FPS。

- 暂停/恢复识别：暂停时相机继续刷新，旧识别图、计数及下载按钮清除；恢复从最新帧重新识别。
- 文字/参考图模式：切换时清除原模式结果，运行中的旧模式推理不能重新发布。
- 图片试用：切到此页时暂停实时推理，可继续使用上传、样例及显式结果下载。切回实时监控恢复最新帧处理。
- 实时帧仅保留有界最新帧和结果，不排队追赶旧视频、不逐帧落盘。上传图片仍按原规则保存独立证据；用户点击下载才导出实时当前帧 JSON。

新增接口：

| 接口 | 行为 |
|---|---|
| `GET /health` | 增加 execution、camera_connected、live_enabled、live_prompt_mode、recognition_fps、inference_error |
| `GET /camera.jpg` | 当前新鲜相机 JPEG；X-Frame-Id、X-Frame-Age-Ms 标明桥接接收身份与年龄 |
| `GET /live-result` | 对应帧的原观察 JSON及overlay_base64；无新鲜结果时返回503 |
| `POST /live` | JSON `{"enabled":true,"prompt_mode":"visual"}`；仅控制识别显示，不能触发动作 |

结果保留原 schema 和三类编号：0=plastic、1=metal、2=paper，不套用旧0=can编号。新增 `source_stream_id`、`source_frame_id`、`camera_received_monotonic_ms`、`upstream_age_ms`、`result_age_ms` 关联对应 JPEG。帧龄是桥接/本机接收年龄，**不是相机曝光或机械动作的精密硬件时间戳**。

上游年龄与本机等待时间共同计入2秒有效期；重复轮询同一帧不刷新年龄。断线、流重启、暂停或模式切换会使运行中的旧结果失效；即使相机继续更新，超过有效期的识别结果也不显示。浏览器另有过期画面清理。HTTP并发处理与单一推理锁避免浏览器空闲连接堵住整个页面；端口使用排他绑定，防止新旧服务同时抢占同一地址。

始终保持 `automatic_motion_allowed=false`、`quality_gate_passed=false`、`sorting_bin=null`。GPU 加速没有修复原先透明瓶错分、箱体误检或箱内漏检问题。

## 实际验证与测量

1. 先运行 GPU 单图入口，实际设备为 Radeon 890M，张量在 cuda:0；完成后才接相机。
2. 同一冻结模型和8张既有图片的 CPU FP32 / GPU FP16 对照：9个实例类别及数量一致，匹配掩码 IoU **0.976～1.000**，最大得分差约0.0028。这是数值一致性检查，不是新质量验收。
3. 第一轮暖机后 GPU FP16 中位约97.5 ms；GPU FP32更慢且波动，选择FP16。旧CPU记录约0.3～0.7秒，不是同期公平性能对照。
4. 正式8778页面连续20秒采样：56个新结果，观察更新率 **2.77帧/秒**；推理中位 **111.45 ms**，范围74.59～1264.73 ms；接收至结果年龄中位273.5 ms、最大1453 ms；采样中无503轮询。另一组YOLO训练同时使用GPU，数据包含资源争用，不能宣传为稳定30FPS或无负载上限。
5. 实际浏览器显示ROCm GPU/FP16、现场相机与对应轮廓，并验证暂停隐藏旧结果、相机继续刷新及恢复更新。没有将历史截图当作实时输入。

专项检查（工程目录）：

```powershell
..\.venv-rocm\Scripts\python.exe -m unittest sim.test_yoloe_inference sim.test_yoloe_live_preview -v
```

**21项通过、0失败、0跳过**，覆盖材质映射、非正方形掩码坐标、空闲HTTP连接、端口重复绑定、重复帧过期、上游年龄、只取最新帧、旧结果过期、暂停/模式切换/重连期间丢弃在途结果，以及非法源/控制拒绝。

原轻量环境工程全量模拟 `..\.venv\Scripts\python.exe -m unittest discover -s sim -p "test*.py" -q`：284项，276通过、8跳过。6项HTTP检查因轻量环境缺可选Pillow跳过，已在ROCm环境全部通过；另2项为既有ONNX/CNN可选依赖。仓库检查、差异检查及启动脚本语法另行核对。

证据仅在本机 `local-only/yoloe-live-acceleration/`：`gpu-benchmark.json`、`gpu-deployment-first/`、`gpu-cpu-equivalence.json`、`live-20s-measurement.json` 和实时截图。现场图、参考图、派生权重、环境均无共享下载地址，不随Git同步。当前暂无本轮PR，保留其他既有未提交工作。
