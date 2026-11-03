# 基于FPGA的可回收物智能分拣系统

**当前现役链路：**v8/ROCm实例分割、电脑CPU抓点/语义统计、PYNQ采图与新FPGA RGB/六区连续统计、KM1/ESP32执行。KnownItem快档已[直接使用新视觉接口](docs/sorting_station_interface.md)；旧双核不再用于现役执行。原始自动结果、人工复核和补证分别保存，生产和无人值守门槛未放行。

**统一监视器（8779）：**[四视图监视器](docs/sorting_monitor.md)按实时画面、任务原图/命令记录、GPU/FPGA保存实测和图片识别组织，撤下旧双核面板；抓点明确来自CPU。FPGA相对板端ARM参考2.41倍，同输入HX370参考15.1ms，不宣称整机提速或快于本机。

## 历史部署与阶段记录

**2026-10-30整帧处理候选：**新增灰度、3×3平滑、Sobel边缘、变化去噪、直方图与六区摘要，保留原RGB核验规则。C/RTL仿真和640×480真实画面离线逐像素核对通过；本轮按队长要求仅完成构建和离线验证，不切换现役服务。见[流水线接口](docs/frame_pipeline.md)及[本轮证据](../docs/team/reviews/2026-10-30-frame-pipeline.md)，加速比待上板实测。

以下旧候选、旧端口和实物记录用于追溯，不表示同时处于运行状态；当前状态以上文及[STATUS](../docs/team/STATUS.md)为准。

**FPGA主线调整候选：**保留ESP32，新增板端RGB/六区域/多帧持续变化流水处理；C/RTL验证、配对位流构建及100MHz时序已通过，文件已隔离暂存但未启用，详见[板端视觉核验](docs/station_board_vision.md)。旧双核仍供现役联调兼容，不能把候选写成已部署或已获得性能收益。

**综合监视器（8779）：**复用当前GPU识别，显示FPGA统计、抓点、执行指令与核验记录，以及CPU/PL包含搬运通信的真实对照。当前PL往返慢于本机CPU，页面明确显示，不能宣称FPGA端到端加速。见[监视器说明](docs/sorting_monitor.md)。

**2026-10-29 三类统一单件自动入口已实物跑通：**当前v8按识别类别自动选择高度、夹持量与箱位，纸/金属/塑料各一轮16/18/17条指令，耗时88.6/99.9/94.3秒；终态三帧核验与现场反馈均通过，工程回归485项通过。使用根目录脚本的`-KnownItem`模式，见[单件自动操作](docs/sorting_known_items.md)与[实物记录](../docs/team/reviews/2026-10-29-physical-loop-commissioning.md)。范围为固定工位、同一已测样件、一次一件、目标箱为空，三类无人值守尚未验收。

**当前已按队长要求启动YOLOv8s并迁移闭环识别适配：**显式绑定模型族、supervised模式与新权重/运行指纹，真实三帧及PL对照已核对。见[v8操作说明](docs/yolov8s_realtime_rocm.md)、[实物记录](../docs/team/reviews/2026-10-29-physical-loop-commissioning.md)和[CPU/GPU实测](../docs/team/reviews/2026-10-29-rocm-cpu-benchmark.md)。原11s与YOLOE资料保留。

**2026-10-29 三类自动闭环代码入口：**[统一实现与现场步骤](docs/yoloe_physical_loop.md)。新增分类、逐件重规划、双核统计服务、串口执行、放置后视觉核验及持久异常账本；根目录 `tools/start_sorting_loop.ps1` 支持 Demo / Preflight / Preview / Execute。三类合成闭环已跑通，真实三类标定、识别质量和路径验收仍待完成，本轮未驱动设备。

**2026-10-29 YOLOE GPU实时监控已运行：**见[ROCm部署及实时监控](docs/yoloe_realtime_rocm.md)。8778当前使用Radeon 890M/FP16，显示相机及对应帧三类轮廓，支持暂停/恢复。并行训练下实测约2.77次更新/秒，识别质量仍待改善，未接管机械分拣。

**2026-10-29 YOLOE 三类预览已运行：**见[塑料/金属/纸质识别说明](docs/yoloe_materials_deployment.md)。本机8778支持图片上传、逐件轮廓及文字/参考图两种提示；已有现场非参考图试跑，仍有错分、漏检和背景误检，尚未接管自动分拣。

**2026-10-29 TACO 原模型已独立跑通：**见[本机部署与实测](docs/taco_original_deployment.md)。实时相机与连续识别入口为本机8781，原8777保留图片上传。按用户规则将作者十类归并为瓶子类（Bottle/Bottle cap/Cup/Straw/Lid）、罐子类（Can/Pop tab）、其他类（剩余三类），保留原始输出；当前本机 CPU 预览不接管自动分拣。样例中有背景误检与塑料瓶错分，不能认定质量验收通过。

**三类模型最终训练：[完整方案与一键脚本](docs/recyclables_final_training.md)。** 使用完整分组数据运行正式训练；此前小批训练记录保留用于追溯。

**2026-10-28 当前路线：**静态三类可回收物识别、抓取与视觉核验，见[总方案](../docs/design/static_waste_sorting_plan.md)和[从零开始实施手册](../docs/design/recyclable_sorting_from_zero.md)。GPU 实例分割和新增 PL 图像核均按新方案单独验收。当前移出 NPU、自动持物/脱落判断和第二视角，保留抓取前识别与退离后的单相机复查；结果不明时人工复核，不自动重抓。下方为已有动态剔除工程的说明和命令，不是新系统的启动命令；保留[静态分拣离线模拟](docs/static_sorting_simulation.md)，2026-10-29新增的[统一设备入口](docs/yoloe_physical_loop.md)仍须完成三类现场标定与实物验收。

## 历史动态路线与可复用工程基础

当前实施：**正常/变形瓶盖两类 YOLO11n 识别 + PL 掩码与编码器采集 + PS 目标关联及窗口预测 + KM1 动态剔除**。六状态是后续扩展；正式上位机为已到位的 AMD Ryzen AI PC，原拯救者 Y9000P 已退出当前使用。

**2026-10-16 当前状态：**AI PC 已接替原上位机，实机为 ASUS Zenbook S 16 UM5606WA、Ryzen AI 9 HX 370、Radeon 890M、32 GB、Windows 11 25H2。AMD 驱动已更新为 32.0.31021.5001，新驱动下 ROCm 基础运算与随机权重 YOLO 检查已通过；本队模型迁入、真实图片推理和板端通信尚待验证，详见[环境记录](../docs/team/reviews/2026-10-16-hx370-environment.md)。Vivado/Vitis 2026.1 由用户安装，新机工具环境与历史构建分别验收。板卡、相机和 PL 图像核的历史证据见[团队状态](../docs/team/STATUS.md)，不将旧机成绩视为新机成绩。

- **[历史动态路线开工手册](docs/commissioning_roadmap.md)**：旧路线的操作命令、参数填写、故障定位与证据保存。
- **[历史动态剔除方案](docs/final_design.md)**：旧路线架构、算法分工与验收指标；当前方案使用上方新入口。
- **[Windows最终操作手册](docs/windows_final.md)**：训练、部署、上板与批次执行。
- [Mac协作](docs/macos.md) · **[HX 370 / Windows 11正式部署](docs/amd_deployment.md)**
- [硬件与标定](docs/hardware_setup.md) · [通信协议](docs/protocol_zh.md)
- [识别链路交接](docs/recognition_pipeline.md)：当前两类模型的训练、评估、ROCm 部署与 PYNQ 接口。
- [AI PC 工位视觉监视](docs/vision_monitor.md)：查看 PYNQ 送检画面、类别、置信度与检测框；提供独立的板端连续观察入口。
- [指标与证据](metrics/README.md) · [验证记录](report/validation.md)

## 全项目网络：网线直连

当前 AI PC 沿用有线目标地址192.168.2.1/24，一根网线直连PYNQ-Z2（192.168.2.99/24）；电脑Wi-Fi保留上网。启动登录、SCP上传/取图/取日志、板端整帧YOLO请求、动态任务使用这条链路，正式闭环无需路由器或公网。根配置及板端样例的inference.url保持为http://192.168.2.1:8765。新机接线和地址仍需核验。

先按[开工手册3.2](docs/commissioning_roadmap.md#32-上电查ip和登录)确认 AI PC 实际物理网卡和地址，按第9.0节核验仅限板卡的检测服务端口；缺板端依赖见4.3.1。原 Y9000P 的网络及 CUDA 服务记录仅作历史证据。当前按[AMD部署手册](docs/amd_deployment.md)完成驱动、模型和通信复验。

**当前阶段已收窄：**先只检测瓶盖变形这一种缺陷，保留正常瓶盖对照，按开工手册8.0走正常/变形两类软件链路。两类YAML为data/cap_deformation.yaml，模型独立保存到data/models/cap_deformation；原config/host_cap_deformation_cuda.json保留作历史对照，当前按AMD部署手册配置ROCm并核对两类权重路径与标签。完整六状态是后续扩展，当前不要求全部备齐。

## 新任务范围与迁移

六状态为cap_ok、cap_notch、cap_deformed、plate_ok、plate_corner_missing、plate_hole_missing。只对4种缺陷状态设置arm.bins，正常类不抓；未抓取的未知/漏检/漏剔件也可能到末端，因此末端为待复核区。当前没有自动合格放行账本，需录像人工真值。现有FPGA负责几何处理和编码器，尚无PL缺陷特征核；新增场景的性能收益要做CPU/PL及整机实测。

[样件与完整操作步骤](docs/commissioning_roadmap.md#8-第六步两种零件六种状态的外观缺陷检测) · [六状态训练配置](data/defect_dataset.yaml) · [缺陷剔除配置样例](config/defect_board.json)

## 主入口

FPGA 工具链统一为 **Vivado 2026.1 + Vitis HLS 2026.1**，版本来源为 `build/toolchain_version.txt`。本机普通 PowerShell 可直接运行 `./tools/build_windows.ps1 -CheckOnly`，自动寻找现有安装。版本检查与 Windows 构建路径见 [2026.1 环境说明](docs/vivado_2026_1.md)。

```text
PC:    python -m src.train_yolo / src.evaluate_yolo / src.detect_server
PYNQ:  python3 -m src.run --config config.json --execute --max-tasks 20
FPGA:  tools/build_windows.ps1 -Kernel segment_open -BoardRepo <实际目录> -EncoderXdc <实际引脚约束>
```

运行前按手册完成实物标定、参数验证和模型训练。默认不执行机械动作，arm.commissioned=false且dynamic.commissioned=false；手动添加--execute才允许串口动作。根config.json是板端工作配置；现有根配置及config/contest_board.json仍是v5旧四类，未自动覆盖。新样例是config/defect_board.json，训练YAML是data/defect_dataset.yaml。按开工手册8.2备份迁移后再运行；新样例两个commissioned均为false。

## 与指南推荐目录的对照

| 指南模块 | 本工程 |
|---|---|
| docs架构、接口、硬件 | docs/final_design.md、protocol_zh.md、hardware_setup.md |
| FPGA源码与构建 | src/hls/、src/rtl/、build/、tools/build_windows.ps1 |
| 位流与工具报告 | board/vision.bit、vision.hwh、report/timing.rpt、utilization.rpt，构建后生成 |
| AI PC应用 | src/detection.py、detect_server.py、train_yolo.py、evaluate_yolo.py |
| 板端应用 | src/recycling_run.py、moving_targets.py、conveyor.py、detect_remote.py、pl.py、arm.py |
| 模型与部署 | data/models/，训练后生成；config/host_*.json、requirements_yolo.txt |
| 指标与证据 | metrics/，board/events.jsonl与board/evidence运行后生成 |
| 工作流与设计报告 | skill/、report/ |

[输送带与编码器配置](docs/conveyor_setup.md)是v5新增的必读部署文档。

旧src.contest_run仅保留为静态对照/故障定位，config/static_board.json含旧类别，不直接用于当前六状态缺陷样品；必要时按开工手册从当前配置生成独立对照配置。旧src.baseline_run、train/train_cnn、pc_server、inference与config/baseline_local.json仅用于历史/消融对照，不作为最终方案入口。其余旧入口文档已指向统一手册。已测结果与尚待设备执行的项目分开记录，不将设计目标写成实测成绩。
