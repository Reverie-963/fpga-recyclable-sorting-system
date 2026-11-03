# YOLOE / YOLO11s / YOLOv8s 三类自动物理闭环

当前现场执行入口是[三类已测样件的统一单件自动模式](sorting_known_items.md)：`tools/start_sorting_loop.ps1 -KnownItem`。它复用本页控制状态机、PL/模型适配与持久账本，增加按现场记录绑定的局部抓点及关节路径。原生产模式和正式质量/机械门槛保留，不为运行已测样件伪造生产验收。

当前监控使用队长选择的[YOLOv8s-seg / ROCm](yolov8s_realtime_rocm.md)。HTTP适配支持显式 `model_family=yolov8s` 或 `yolo11s`、`prompt_mode=supervised` 与各自权重/运行指纹，并严格拒绝模型族混接。下述纸盒单件实物试验使用11s；旧计划及停止记录保留。

更新：2026-10-29。统一入口是 `src.sorting_run`，Windows 启动脚本是根目录 `tools/start_sorting_loop.ps1`。这轮完成可执行控制代码与模拟验证；三类实物自动分拣尚未验收。现有 YOLOE 错分、背景误检和箱内漏检仍会影响闭环，GPU 部署成功不等于可开放自动动作。

后续现场阶段已获队长明确授权并开始：真实奶盒识别、双核统计CPU对照以及65毫米卡面空爪悬停核对已执行，最新进度见[团队状态](../../docs/team/STATUS.md)。下文“本轮未运行”描述初次代码交付时的边界，不覆盖后续已授权实物记录；三类自动验收仍未完成。

此前11s奶盒17条、当前v8空罐20条/透明塑料件19条分阶段抓放均通过。当前v8统一入口另已完成纸/金属/塑料各一轮16/18/17条完整自动计划，约88.6/99.9/94.3秒，三帧语义/RGB/PL核验及现场反馈全部通过，见[完整实物记录](../../docs/team/reviews/2026-10-29-physical-loop-commissioning.md)。这些指定样件单次记录不代替三类识别质量或无人值守验收。

## 运行拓扑与分类

首版沿用现场已使用的连接方式：PYNQ USB 相机 → 8780 只读 JPEG 桥接 → AI PC YOLOE / ROCm → PYNQ 双核统计服务 → AI PC 统一规划及账本 → AI PC USB 直连 KM1。这是静态路线的当前集成实现，区别于早期设计里由 PS 集中调度的目标拓扑，也区别于历史输送带入口。OpenMV V3 可由同一 Motion 适配器显式选择，但当前整臂路径仍须按实际连接单独验收。

| YOLOE 标签 | 固定箱 ID | 需要单独标定的物体配置 |
|---|---|---|
| plastic | bin_plastic | 首批指定空塑料瓶 |
| metal | bin_metal | 首批指定空易拉罐 |
| paper | bin_paper | 首批指定硬纸盒/纸板 |

不把 YOLOE 的 0/1/2 套成旧模型 can/plastic_bottle/cardboard 的类别顺序。动作路由直接使用上述字符串。材料分类不能确定物体高度和夹持受力；每类先绑定有明确高度、尺寸和姿态范围的样件。当前固定腕角与主轴窗口配合使用，超出已测方向直接拒绝，不承诺任意垃圾、任意姿态或堆叠抓取。

## 当前即可运行

在仓库根目录运行：

```powershell
.\tools\start_sorting_loop.ps1 -Mode Demo
.\tools\start_sorting_loop.ps1 -Mode Preflight
```

Demo 使用三件合成物体、同一真实控制状态机、CPU 整数统计和内存执行器，依次分类投放并检查结果，最后源区为空。日志中的 `simulation=true`、`simulated_visual_pass` 不代表设备成功。它不访问相机、GPU、串口或 FPGA。

Preflight 只读取配置、参考图和验收记录，不加载模型、不访问设备。示例配置会明确列出未完成项，退出正常不代表允许动作；必须查看 `status` 和 `execution_blockers`。

## 一个完整周期

1. 采集至少三张相互独立的新帧；核对相机、图像尺寸、模型、提示模式、运行环境指纹、帧号和年龄，与已标定工位的静态纹理块比较。
2. 源区允许 1～3 件分离物体；不确定类别、相接、跨源区边界和场景变化均拒绝。稳定帧窗口中的所有帧在决策时仍须新鲜。
3. 确定性选择一件，检查掩码连通性、全部轮廓是否在实测区域内、该类的面积与物理宽度、主轴方向、内部抓点；分别用抓取和悬停高度的映射转换坐标。
4. 一次预检完整 12 步计划：回已确认准备位、腕角、张爪、接近、下降、闭爪、抬升、到箱上、下降、释放、退离箱、回准备位。每条指令都检查真实关节限位，后续不可达也会阻止第一条指令。
5. 再拍一帧核对；先持久写入动作意图再打开串口。串口打开后、进入目标上方前、下降前再取新帧核对。下降前若机械臂遮挡造成分割或画面不一致，会停止交人工复核，不能放宽为默认通过。
6. 每条指令发送前落盘，按动作时间和额外停稳时间等待。直连要求完整回显；回显只是通信证据，不当成舵机到位或抓住物体的反馈。V3 的 SENT_UNVERIFIED 同样只证明桥接发送。
7. 退离后丢弃缓存帧，重新采三帧。选中物体源位置消失、其余源物体保持、目标箱内出现一个同类新实例、新增语义掩码与 RGB 变化持续重合、原箱内物体仍独立可见、其他箱无异常变化，才结为 `completed_visual_evidence`。
8. 每件结束后重新观察和规划下一件。动作或核验异常为 `needs_review`，持久锁存，不自动重抓、回滚或重新发送。低置信、源区规划失败等未动作任务记录为 `rejected` 并结束本次批次，拒绝件不会悄悄计为成功。

不做途中持物/脱落识别；`physical_success_verified=false` 表示没有把软件视觉结论冒充独立人工真值。后续应通过独立正负实物试验验证自动视觉判定。

帧号来自8780桥接，年龄使用桥接年龄加控制端等待/推理耗时估算，不是相机曝光硬件时间戳。默认还要求连续证据的JPEG内容不同，避免只靠重复轮询证明稳定；相机冻结或桥接重启会结束本次会话。

## 实测配置与验收记录

将 `config/sorting_station.example.json` 复制到 **同一 config 目录**下的本机配置 `sorting_station.local.json`；本机文件不提交。若放在其他目录，须相应调整所有相对路径。相对路径总是相对配置文件，不依赖启动目录。示例 ROI、阈值、边界只是格式说明；`synthetic_fixture=true` 会阻止实物执行。不要仅修改许可开关尝试动作。

必须补齐：

- 固定工位：真实 source/bin/static ROI、640×480 图像、相机固定姿态、两块分离且有足够角点的静态纹理、参考 PNG 及 SHA256、标定版本。
- 识别：冻结 `model_version`、`runtime_id`、`prompt_mode`、`device` 和 `half`；默认与当前 `cuda:0` / FP16 / visual 对齐。运行中的 8778 `/health` 可查看默认模式的模型及运行指纹。不能把 text 的哈希填给 visual，也不能自行把当前 `quality_gate_passed=false` 改成已验收。
- 每类 `object_profile_id`、实测物体高度、该高度的 `image_to_grip_mm` / `image_to_hover_mm` 3×3 单应矩阵、覆盖轮廓与抓点的凸多边形、抓取/悬停模型 Z、alpha、腕角、夹爪开闭脉宽、面积/宽度范围、主轴角度与允许偏差。
- `bin_xyz_mm`、实际准备位、所有六关节的实测限位、速度与停稳时间，以及整条路径的扫掠净空验证。逆解通过和 PWM 在范围内不构成碰撞证明。每个周期起始必须确实位于现场确认的准备姿态；串口并不读取关节位置。
- 串口设备名与 `usb_identity`（整数 vid/pid，可加 serial_number）、双核 bit/hwh 同次构建哈希、PL 服务地址。CPU 基线可预览；真实执行必须用 `pl_http`，不会故障时偷偷回退 CPU。

每类配置的 `acceptance_record` 为 `{"path":"现场记录.json","sha256":"文件SHA256"}`。记录至少包含下列字段，并附独立实测过程、样件和结论：

```json
{
  "kind": "sorting-profile-acceptance-v1",
  "station_id": "实际工位ID",
  "label": "paper",
  "configuration_sha256": "预检返回的acceptance_configuration_sha256",
  "automatic_trial_approved": true
}
```

三类各一份，标签分别为 plastic / metal / paper。只有实测完成后才填写批准结论。模型、阈值、ROI、映射、箱位、关节限位或动作时间改变都会使配置指纹变化，旧验收记录不能继续解锁；只切换已验收的许可声明或填入验收记录位置不改变该指纹。本实现不生成虚构的验收记录。

当前旧纸盒两高度映射存在名义高度复用，旧投放后曾三帧漏检；不能将其直接提升为三类自动验收。现场应先补箱内识别，再逐类核对标定、空爪路径、带物抓放与核验正负例。

## PL 统计服务与接口

板端在项目目录运行。**下方会加载 FPGA，因此只供另行授权的现场阶段使用；本轮未运行。** 令牌在两端各自通过环境变量 `KM1_STATS_TOKEN` 提供，至少 32 字符，不写入 JSON 或 Git。沿用已验收同次构建的 `vision.bit/.hwh`，不要为了启动服务重新综合或拿不配对文件替换。

```bash
python3 -m src.sorting_statistics_server --bitfile board/vision.bit --host 192.168.2.99 --port 8782 --load-overlay
```

服务复用 `board/km1_dma_runtime.py` 的单一 DualCoreSession，串行调用两个已有驱动，保持 dual-core-v1.1 约定；没有相机或机械动作接口。几何固定 erode=0 / dilate=0，视觉固定 mode=1，输出像素仍与现有驱动一致，不修改 HLS、寄存器、引脚或位流。发生驱动异常后服务锁存故障，不在后台自动重载 overlay。

| 请求 | 返回及约束 |
|---|---|
| GET /health | contract=sorting-statistics-v1、session_id、bit/hwh 哈希、faulted |
| POST /run | role=geometry/visual、session_id、request_id、job_id、image_size、1或2个掩码；返回相同身份、完整统计及输出像素 SHA256 |

请求带 Bearer 令牌，最大 120000 字节；掩码按原图逐行排列，以 NumPy packbits 的 little bit order 打包、base64 传输，末字节未使用位必须为零。geometry 返回 m00/m10/m01/m20/m02/m11、行列边界、job_id/status；visual 返回 inter/erased/beyond/union、变化边界、job_id/status。边界最大值包含端点，空结果边界为零。PC 对每次统计和输出像素哈希执行 CPU 精确对照，任何不符均停止；不能因此宣称已经测得 PL 加速收益。

## 现场预览与有限自动执行

先关闭或暂停独立 8778 的连续识别，避免两个进程重复推理争用 GPU；保留8780相机桥接。统一入口自身只在决策与复查时推理，整批只加载一次模型，不积压视频、不逐帧落盘。证据仅保存动作前、复核点与退离后的 PNG/掩码/指纹及动作日志。

也可在实测本机配置使用 `inference_backend="http"`、`detector_url="http://127.0.0.1:8778"` 复用已部署GPU模型，控制器本身不用再加载网络。适配器向既有 `/detect` 发送当前原图解码后的无损PNG，逐次核对输入字节哈希、模型/运行指纹、提示模式及尺寸；相机原JPEG身份由控制端单独保留。此方式只在检查点请求识别，既有服务会按原规则保存这些有限请求的图片证据。默认省略该字段仍为原in_process模式。

训练模型必须额外显式设置 `model_family="yolo11s"` 或 `model_family="yolov8s"`、`prompt_mode="supervised"`，并从当前服务绑定对应权重和运行指纹；旧配置不会自动接受另一模型族。可选duplicate_instance_policy包含min_mask_iou（至少0.90）与min_score_margin，仅解析同一高重合轮廓的明显类别领先，接近分数仍拒绝，独立低分物体保留。`sorting_scene.check_motion_layout`单独检查运动中的固定标记位置/可见性，不用姿态引起的RGB变化证明机位移动或抓取成功；最终颜色与语义判据保持独立。

```powershell
.\tools\start_sorting_loop.ps1 -Mode Preflight -Config pynq_km1_vision/config/sorting_station.local.json
.\tools\start_sorting_loop.ps1 -Mode Preview -Config pynq_km1_vision/config/sorting_station.local.json
```

Preview 走真实相机、模型、配置选定的统计后端和规划，输出完整指令，但不打开串口。需要实测标定后才能输出可行计划；示例空矩阵不会生成虚假计划。

所有三类记录、质量检查和工位许可齐全，并取得本次实物实验授权后，才运行：

```powershell
.\tools\start_sorting_loop.ps1 -Mode Execute -Config pynq_km1_vision/config/sorting_station.local.json -MaxTasks 1 -MaxSeconds 300
```

先完成单件，后续授权批次可改 MaxTasks，最多100件、总预算最多3600秒。剩余批次时间不足一个 max_cycle_s 时不再启动新动作。空源区会等待到批次观察预算结束，其他异常终止批次。

创建配置 `stop_file` 指定文件或按 Ctrl+C 会停止后续指令。软件停止不保证舵机立刻停止，已发指令可能继续运动；现场断电方式按设备验证。发生异常后保留夹持与位姿事实，不自动张爪、回原点或补发动作。

## 账本与人工复核

同一工位固定使用同一 `journal_dir`。controller.lock 阻止并行运行；pending.json 在串口打开前保留任务，并在每条指令发送前刷新。崩溃后即使换进程，pending 也会阻止新任务。不得换日志目录绕开未解决任务。

操作者确认现场状态、移走异常物品、恢复已验证准备姿态后，记录事实再解除该任务：

```powershell
Set-Location pynq_km1_vision
..\.venv\Scripts\python.exe -m src.sorting_run --config config/sorting_station.local.json --resolve-task 实际task_id --observation "现场复核事实和恢复情况"
```

此操作记为 `operator_resolved_not_automatic_success`，不会补记自动成功或重新发送原任务。异常退出遗留 controller.lock 时，先确认记录中的进程确已退出，再由操作者移除锁文件；保留 pending 和证据，再执行上方复核。不得在原控制进程仍运行时移除锁。

## 验证范围

专项命令：工程目录 `..\.venv\Scripts\python.exe -m unittest sim.test_sorting_loop sim.test_sorting_io -q`。验证三类重规划、图像掩码绑定、稳定与过期帧、主轴/宽度/标定/后段限位拒绝、动作前落盘、USB打开后的复查、下降前场景变化、错箱、源物体未消失、旧箱物体复用、异常锁存和人工复核；本地模拟 HTTP 验证双核契约、令牌、配对资产、响应身份及故障后禁止重试。

详细本轮数量及跳过项见[本轮记录](../../docs/team/reviews/2026-10-29-yoloe-physical-loop.md)。尚未验证本轮真实GPU/板端联调、串口回显、三类机械动作、PL服务上板、时序、闭环性能或长期成功率。已有模型和参考资产仅本机，无共享下载地址；新代码不会把旧成绩提升为本次验收。
