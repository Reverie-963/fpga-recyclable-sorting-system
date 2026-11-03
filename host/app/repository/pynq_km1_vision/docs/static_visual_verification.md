# 静态抓放视觉检查点：录制回放版

**2026-10-28 范围调整：**当前方案移出 NPU、自动持物/滑移/脱落判断，不增加侧视相机。下文 `lift`/`hold` 是早期只读回放接口，保留用于历史实验，不再作为当前流程的必经检查，也不接入执行器。当前仅推进抓取前基线与 `placed` 放置退离后的源区/箱内复查；箱内奶盒漏检和机位门控尚未通过实物验收，结果不明确时交人工复核，不自动重抓。手机侧照可作爪口和净空的人工补充证据。

当前下一步：固定现有工位 → 采集源区与箱内前后正负例 → 解决漏检与机位检查 → 冻结阈值 → 独立验证放置后复查。途中不安排抬起或持物检查采样，人工确认与程序结果分别记录。

2026-10-28：新增 `src.static_visual_verification`，用于 CPU 离线核验连续图像和实例掩码。没有相机、串口、PL、NPU 或运动入口，没有接入自动执行器。继续使用 240 张轮次 best，模型 SHA-256 为 `db23bd893c32bf706ae51a5e7b7c51270eef1c96369a1961130dcc7830770a1e`。

这是下一轮现场采集的可测试基础，**尚未解决自动核验的实物验收**。旧两轮的 `needs_review` 原样保留；不能用新方法追认旧试验通过。

本轮用户确认工位不变、奶盒放回、右箱清空后，通过已有本机SSH/V4L2脚本只读采得3张新图。三图仅检出奶盒，纸类置信度0.982～0.990、轮廓面积4411～4413像素。分离静态区域的模板匹配诊断位移为0，但旧整体机位门控仍拒绝；不据新诊断放行动作。原图和完整掩码只在本机，摘要见[真实基线记录](../report/static_visual_baseline_20261028.json)。随后手动移走奶盒及手动放入右箱各采3张，共9张。空场景右上静态背景有2次低置信度罐误检，但源区/箱区无检出；箱内奶盒3帧均漏检。静态模板新增角点支撑和匹配唯一性检查后，原候选左侧直边区域均因歧义拒绝；早期位移诊断原样保留。已知物体SIFT外观探索在空箱无匹配、箱内仅4～6个拟合内点，尚不能作为可靠替代核验。尚无本轮机械臂抬起/持物/放置样本，手动摆放不计机械臂成功。

## 行为

| 检查点 | 需要观察的证据 | 拒绝情形示例 |
| --- | --- | --- |
| `lift` 抬起后停住 | 源区清空，唯一同类实例位于固定持物观察区，多帧稳定 | 空抓、源区还有物体、看不见目标 |
| `hold` 持物停留 | 在同一观察姿态重新采集连续帧，目标位置和掩码稳定 | 掉落、滑移、遮挡、重复旧帧 |
| `placed` 放下并退离 | 源区清空，目标位于正确箱内，多帧新增外观与语义掩码同时存在 | 仅阴影变化、旧箱内物体无新增外观、放错箱 |

每处恰好接收预先冻结的 `stable_frames` 张图（至少3张），帧间距、置信度、可见面积和稳定度等均为显式策略参数。此版本一次评估一个检查点，**不实现跨检查点状态机、连续夹持身份跟踪或整轮成功账本**。

源区 RGB 变化比例继续输出作为诊断，不再作为通用物体消失门槛。源区清空仍要求所有返回实例的联合掩码在源区没有前景，且原目标轮廓消失。检测漏检仍是风险；不能单凭源区空白证明成功。箱区同时要求同类实例、正确区域和跨帧持续变化，不能单靠变化像素放行。

机位检查只用冻结的两个以上静态纹理区，区域彼此分离且避开源区、持物区和箱区；检查模板相关性、局部位移及颜色差，不自动配准补偿。平坦无纹理的桌面、角点不足或支撑太局部的区域、直边或重复纹理产生的歧义匹配均拒绝作为稳定证据。角点至少4个且凸包覆盖模板面积至少10%；匹配主峰还需与允许位移半径外的候选拉开冻结的分差。重复纹理、局部形变、单目深度和模型误识别仍须用实物正负例检验。

三个动作区域在这个保守版本中必须互不重叠。若抬起后物体投影仍与源区重叠，或机械臂遮挡物体，就无法使用该检查点；需另行确认可见的受监督观察姿态，或调整传感方案，不能仅修改 ROI 声称离桌成功。`hold` 只能检查录制的停留时间内的可见稳定性，不能证明全程夹持力或转移途中没有掉落。

所有结果保持 `automatic_motion_allowed=false`、`physical_success_verified=false`、`independently_validated=false` 和 `automatic_retry=false`。`visual_evidence_pass` 仅表示本检查点证据符合暂定规则，不授权下一段动作。

## 输入和运行

在主工程目录、根目录虚拟环境已安装 `requirements.txt` 后运行：

```powershell
..\.venv\Scripts\python.exe -m src.static_visual_verification --help
..\.venv\Scripts\python.exe -m src.static_visual_verification --recording ../local-only/visual-checkpoints/task-001/lift.json
```

第二条命令要求先形成真实录制清单；仓库不附带可冒充现场数据的演示清单。退出码0表示暂定视觉证据通过，2表示待复核或输入错误。输出为标准输出 JSON，不改输入或历史报告。即使退出码0也不得接到现有执行器自动放行动作。

顶层清单使用 `schema_version=static-visual-checkpoint-v1`：

| 字段 | 内容 |
| --- | --- |
| `reference_image` | `{path, sha256}`，动作前图像，路径相对清单目录 |
| `before_observation` | 与动作前图绑定的身份、帧号、采集时间、图像哈希和唯一目标实例；`stage=before` |
| `before_mask_rle` | 上述唯一目标的完整图像行优先 RLE，必须与实例逐项一致 |
| `policy` | 下表所列冻结策略，按实际检查点和机位生成；不能用测试参数当标定 |
| `stage` | `lift`、`hold` 或 `placed` |
| `recorded_review_ms` | 当时审核时刻，使用与采集时间相同的 PC 单调时钟；离线回放不拿今天时间计算旧帧年龄 |
| `frames` | 有序数组，每项含 `image={path,sha256}` 和 `observation` |

每条 observation 都包含 `task_id`、`camera_id`、`model_version`、`calibration_id`、`bin_id`、`image_size=[宽,高]`、`stage`、整数 `frame_id`、`capture_started_ms`、`image_sha256` 和 `instances`。每个实例包含 `label`、0到1的 `confidence` 和完整图像 `mask_rle`。实例掩码必须来自该图和冻结模型；不要只导出最高置信目标而丢弃源区中其他实例。哈希绑定防止误混图，不能证明输入生产者正确或可信。

策略所有字段均需显式提供：

| 类别 | 字段 |
| --- | --- |
| 身份 | `task_id, camera_id, model_version, calibration_id, bin_id, label, image_size` |
| 动作前帧 | `reference_frame_id, reference_capture_started_ms` |
| 区域 | `source_roi, grip_roi, bin_roi, static_rois`，矩形 `[x0,y0,x1,y1]`，右下界不含 |
| 时序 | `stable_frames, min_frame_gap_ms, max_age_ms` |
| 静态图检查 | `search_margin_px, max_layout_shift_px, min_patch_std, min_patch_corners, min_patch_peak_margin, min_patch_correlation, max_patch_mean_difference` |
| 目标门槛 | `min_confidence, min_target_area_px, max_target_area_px, min_source_erased_fraction, min_target_roi_fraction, max_center_jitter_px, min_persistent_fraction` |
| 箱内新增 | `rgb_difference_threshold`（BGR三通道绝对差之和）、`min_bin_gain_px`（与语义掩码重合且跨帧持续的像素数） |

单帧图像和掩码由调用方生成；本模块不加载 YOLO 权重，不采图，不转换旧现场脚本清单，也不跨机器比较不同单调时钟。现场采集/模型推理适配和跨阶段身份绑定仍待实现。第一次采集先检查相机、标定和可见性，再生成清单及冻结策略。

## 早期三检查点实验步骤（历史参考，已移出当前必做范围）

1. 保持当前相机、底座、纸箱固定。仍用同一奶盒和已观察的小范围位置；先人工清空右箱。不要为了增加视野临时移动相机。
2. 在不触碰工位和运动路径的前提下，检查画面能否提供两个分散的固定纹理区；不足时在静态背景加标记，重采动作前参考图。标记不能进入源区、持物区或箱区。若机位变化，先重标定。
3. 先采空爪、源区有奶盒、右箱空白的参考图。检查提起后是否能看清目标，并与源区投影分开；否则暂停设计观察姿态。
4. 经本次实物授权后，沿已观察路径受监督操作，在抬起、停留、放置退离三个检查点分别记录至少3张新图、模型完整掩码和现场真值。操作者反馈与程序判断分别保存。
5. 另采安全的空爪、未抓起、源区仍有物体、目标在箱外、箱内只有阴影、持物不可见等负例。优先手工摆放/空爪静态采集，不故意带载跌落或碰撞制造失败。
6. 开发样本上拟定阈值，冻结后再采独立验证样本；统计误放行和误拒绝。旧两轮只能做已知案例诊断。通过后再考虑接入状态机和受监督重复抓放，不自动重抓。

## 当前技术和证据

| 技术 | 当前实际用途与边界 |
| --- | --- |
| AMD Ryzen AI 9 HX 370 / Radeon 890M | AI PC；PyTorch/ROCm GPU训练与推理。AMD是硬件和工具平台，AI是运行其上的识别方法 |
| YOLO11n-seg 实例分割、迁移学习 | 当前金属/塑料/纸类模型；240张原照片中230张实际入集（训练147、验证83），同场次诊断，非独立测试成绩 |
| AMD/Xilinx Zynq-7020、PYNQ-Z2 FPGA | PL几何统计和视觉变化双核已独立上板；本轮奶盒抓放尚未调用双核，YOLO也没有在FPGA上运行 |
| HLS C++、RTL、Vivado/Vitis HLS 2026.1、AXI DMA | FPGA核开发、构建与PS/PL数据搬运；本次只改CPU Python，不重建位流 |
| Python、NumPy、OpenCV | 掩码几何、内部抓点、机位检查、平面映射、分高度标定及视觉核验 |
| PYNQ/Linux、UVC/V4L2、SSH/以太网 | 板端USB相机采集及PC取图；当前试验并不等同于已接通正式HTTP自动状态机 |
| 逆运动学、串口、舵机指令、逐段日志 | KM1受监督抓放，PC直连机械臂；回显不能证明夹稳或到位，没有力反馈验收 |
| Ryzen AI NPU、夹持状态小模型 | 2026-10-28 已移出当前范围，不作为部署、运行或验收依赖；保留历史记录 |

最新依据：[240张训练记录 PR #38](https://github.com/Reverie-963/FPGA/pull/38)、[两轮现场记录 PR #40](https://github.com/Reverie-963/FPGA/pull/40)、[FPGA双核独立验收](../../docs/team/reviews/2026-10-27-dual-core-local-repair.md)。原图、权重和完整现场包仅在本机/板端，尚无团队共享下载地址。
