# 静态分拣离线模拟（Issue #29）

本入口只使用 Python 标准库和 JSON，不导入机械臂、串口、PYNQ 或模型模块。它独立于旧动态入口及生产配置。当前接口为 `static-sim-v1` 提案，供 @Porter-Ma 与队长评审；尚未与真实识别服务冻结或联调。

## 运行

从仓库根目录进入 `pynq_km1_vision`，执行：

```powershell
..\.venv\Scripts\python.exe -m src.static_sorting --replay data/static_sorting/demo.json --output ../local-only/static-sorting-demo.json
..\.venv\Scripts\python.exe -m unittest discover -s sim -p "test_static_sorting.py" -v
```

示例使用 8×6 人工掩码、虚构平面映射与箱位，预期完成 1 个模拟任务。输出保存逐任务状态、拒绝原因、规划和逐步模拟结果。相同输入回放产生相同记录；重新回放是模拟复现，不是重发真实动作。目录中的 JSON 可复制修改以构造异常。

完整验收回放使用 `--replay data/static_sorting/acceptance.json`，输出可改为 `../local-only/static-sorting-acceptance.json`。该文件包含 15 个场景：3 类成功及 12 项异常/重复场景。每项附 `expected_status` 和 `expected_reason` 作为验收期望，专项测试会逐项核对；CLI 只执行并保存记录，不把异常案例误当作程序崩溃。缺标定、超范围与无路径等配置异常另由专项测试覆盖。

## 接口提案

顶层包含 `config` 与有序 `requests`。每项请求包装为 `{ "now_ms": 1010, "request": {...} }`，时间均为控制端本地单调时钟的回放值。

| 字段 | 约定 |
| --- | --- |
| schema_version | 固定 `static-sim-v1` |
| request_id / task_id | 非空字符串；同一模拟会话均不可重复，失败后也不可自动重试 |
| camera_id / frame_id | 相机 ID 与非负递增整数帧号；相同相机不接受已检查帧及更早帧 |
| model_version | 必须匹配配置；更换模型需重新建立明确的模拟会话 |
| image_size | `[width,height]`，必须与配置相符 |
| received_at_ms | 结果进入控制端时的本地单调时钟值 |
| received_age_ms | 进入控制端时已知的采集至结果年龄，包含采图/推理/通信耗时；不能直接相减两台机器的时钟 |
| observation_error | 可选；非 null 表示断网、超时或采集失败，立即拒绝 |
| instances | 当前只接受一个实例；空场景及多实例拒绝，不随意选择目标 |
| instance_id / label / confidence | 实例 ID；`can`、`plastic_bottle`、`cardboard`；有限数值且达到阈值 |
| bbox_xyxy | 原图坐标，右/下边界不包含；必须与解码掩码的包围框一致 |
| mask_rle | 原图完整无损二值掩码；逐行从左至右，首段是背景长度，前景/背景交替；所有长度是非负整数，总和等于像素数 |
| simulated_outcomes | 可选的动作名→结果映射；省略的动作默认为模拟 `ok`；`unknown` 或其他非 `ok` 值中止任务 |

例如示例掩码 `[18,2,6,2,20]` 在 8×6 原图的第 2、3 行各有两个前景像素（坐标从零开始）。不使用网络输入坐标、ROI 局部坐标或有损 JPEG 掩码；将来服务端适配必须先还原原图并核对尺寸。当前没有网络客户端或真实服务适配。

处理时年龄为 `received_age_ms + now_ms - received_at_ms`，未来时间、负年龄、超期结果均拒绝。接收年龄必须由将来的采集适配保证可信；离线模拟只验证给定数字。

## 状态与停止行为

顺序：received → validated → planned → picked → lifted → transferred → released → retreated → source_verified → bin_verified → completed。

模拟动作依次为 `pick`、`lift`、`transfer`、`release`、`retreat`、`verify_source`、`verify_bin`。任一步失败都不产生后续动作，状态进入 stopped；已有动作时结论为 needs_review，其余为 rejected。执行未知不自动重发。记录保留完整原因，供人工复核。2026-10-28 起移除自动夹持检查；记录显式包含 `automatic_grip_check=false` 和 `automatic_drop_detection=false`，不再产生 `grip_verified`。旧 `simulated_outcomes.check_grip` 输入以 `unsupported_automatic_grip_check` 在模拟动作前拒绝，调用方须移除此字段；不静默忽略旧失败信号。`completed` 仍只表示模拟完成。

CPU 从掩码求质心并选取距其最近的前景像素，采用显式二维仿射映射得到 mm 坐标；整个掩码映射后均须在取物边界内。抓取点不是实际夹持可行性证明。配置须显式提供相机对应的模拟标定、类别箱位和 `path_available=true`。

`path_available` 仅注入“有/无可行路径”这一模拟条件。生成的是包含取物点、箱位和阶段的符号计划，不计算机械臂逆解、碰撞、腕角、高度或真实轨迹。示例标定和箱位禁止用于实物执行；完整路径的物理验证由后续集成实现。

## 会话边界与后续联调

同一实例同步处理任务，busy 时拒绝新请求；已接收的请求和任务 ID 即使失败仍保留。去重状态仅在进程内，重启后不具备真实系统所需的持久幂等性。本入口无设备输出；正式执行前必须另行实现持久任务账本与崩溃恢复策略，不能直接把此模拟器接上真实机械臂。

下一步与识别队友确认原图掩码编码、帧号作用域、年龄来源、模型版本和错误回包约定；与队长确认真实标定、路径约束与视觉核验接口。现场核验目前为注入的模拟结果，并未分析真实抓取图片。模拟成功不代表设备或三类闭环验收。

## 双核接口离线对接增量

参考 Porter-Ma PR #19 的 `f92dc121`、PR #22 的 `47f96866`。新增 `src/static_mask_contract.py`，仅按源码字段对齐 CPU 参考和录制统计输入，没有合入驱动或加载位流，不代表双核通过板测。

运行前后掩码核验：在工程目录执行

```powershell
..\.venv\Scripts\python.exe -m src.static_sorting --replay data/static_sorting/mask_evidence.json --output ../local-only/static-mask-evidence.json
```

示例配置 `require_mask_evidence=true`，缺少掩码证据直接拒绝。旧 demo 仍为动作注入模式。

`verification_masks` 包含 task_id、request_id、camera_id、model_version、image_size、before_frame_id、after_frame_id、bin_id，以及 source/bin 两组 before_rle、after_rle、job_id。掩码均使用原图尺寸，区域外补零；源区前态须等于选中实例掩码，箱 ID 须匹配计划，后帧号大于输入帧号，两个 job_id 不同。source 为所选物体源区掩码，bin 为固定目标箱区域前景掩码。

CPU 计算 `inter/erased/beyond/union`。默认源物体至少 75% 像素消失且源区无新增前景，箱内净增至少 1 像素；可用 `verification_policy.min_source_erased_fraction` 和 `min_bin_gain_px` 修改。阈值仅用于演示，须由队长用独立现场数据确定。核验失败后进入 needs_review，不自动重试。人工注入的失败/unknown 仍优先阻止成功。

可选请求字段 `pl_geometry` + `geometry_job_id` 接收录制的 `m00/m10/m01/m20/m02/m11` 及行列边界；source/bin 可选 `pl_stats` 接收录制的视觉统计。status 必须为整数 1（DONE 且无错误位），job_id 及每个统计字段必须与 CPU 参考相同，否则在模拟动作前拒绝。核 bbox 的 max_row/max_col 为包含边界，检测框 xyxy 右下为不包含边界；空掩码/无变化 bbox 为零。几何仅支持 erode=0/dilate=0 的旁路比较，视觉仅支持二值 mode=1；不比较形态学输出或灰度 mode=0。job_id 校验只是录制关联，不是设备执行证明。

示例输入和测试统计均为人工合成，不是现场识别或 PL 输出。回放提前拿到前后掩码，先验证输入再模拟动作；正式在线流程须在各阶段动作后重新采图、验证实时年龄、工位和 ROI 标定版本。当前没有在线采图和设备许可。箱内变化也不等于类别或抓取成功证明。

队长 PR #31 新增 20 张未标注照片（两只塑料瓶、两只易拉罐各五姿态），应保持相同实物分组；尚不能直接作为实例掩码输入。本轮没有虚构照片标签或模型精度。真实识别服务仍需队友交付后共同冻结接口。

## PR #33 逐次驱动结果导入

新增离线入口 `src.static_driver_replay`，依据 PR #33 `df362250` 的接口 v1.1：驱动共享 DualCoreSession，并在完成后返回独立的一维 uint32 输出数组。此入口仅消费保存后的数据，不导入驱动、PYNQ 或 NumPy，不加载位流。

```powershell
..\.venv\Scripts\python.exe -m src.static_driver_replay --capture data/static_sorting/driver_capture_fixture.json --reference-report data/static_sorting/pr33_asset_reference.json --output ../local-only/static-driver-replay.json
```

该例是 `provenance=synthetic_fixture`，预期 1 个模拟任务完成。参考文件中的 bit/hwh 哈希抄自 PR #33 验收报告；示例的统计和掩码为 CPU 生成，哈希匹配不使它变成板测记录。报告始终带 simulation=true，并保存 provenance、输入文件及参考文件 SHA-256。

新封装 `static-driver-capture-v1` 包含：

- contract_version=`dual-core-v1.1`，provenance=`synthetic_fixture` 或 `recorded_driver_return`；后者只是导出者的来源声明，不是签名或实物证明。
- assets：`vision.bit`、`vision.hwh` 的 SHA-256；须与操作者选定的可信 `--reference-report` 匹配。参考版本变更须显式换文件，不静默使用其他位流。
- config/requests 沿用原回放格式。每个 requests 项额外包含 captures，其键必须恰为 geometry/source/bin。
- 每个 capture 带 role、request_id、task_id、camera_id、frame_id（本次任务源帧）、model_version、image_size、job_id、stats、output_rle。三次 job_id 不能重复；source/bin 与 verification_masks 的 job_id 匹配。
- geometry 的 erode/dilate 均为整数 0；source/bin 的 mode 为整数 1。输入是已有 request/verification_masks 的原图二值掩码，核输出也按原图完整 RLE 保存；输入与输出不得只保存 bbox 内裁片。

导入会比较全部 CPU 参考统计和输出像素，之后才交给静态模拟器；配置、证据身份和帧关联继续由模拟器验证。ctrl 中 AP_DONE 可能已被会话读取清除，不能要求其为 1，也不能把 status=1 当作完成新硬件任务的证据；本入口只检查录制快照的一致性。

队长已有 `dual_core_acceptance.py` 的 acceptance.json 只保存逐例尺寸、选项和耗时，没有逐例 stats/output，不能直接导入；报告中的 passed=true 或 2560 个成功任务也不能替代这些数据。本机没有这份逐次输出，未完成真实板测结果回放。

队长后续在已授权实验中拿到每次 `(stats, out)` 后，可用以下纯序列化代码保存返回值（代码本身不调用设备）：

```python
from src.static_driver_replay import export_result
capture = export_result(stats, out.tolist(), [width, height], job_id,
                        erode=0, dilate=0)  # geometry；visual 改为 mode=1
capture.update(role="geometry", request_id=request_id, task_id=task_id,
               camera_id=camera_id, frame_id=source_frame_id,
               model_version=model_version)
```

分别收集 geometry/source/bin 三份返回值并放进同一任务的 captures。导出者还需保留对应原图二值输入、前后帧/目标箱、时间与标定元数据，按示例完成 request/config。收到真实文件后只在离线入口回放，不会自动连接设备。PR #33 驱动输出已经复制并释放内部 DMA 缓冲，调用方不应再次释放 out。
