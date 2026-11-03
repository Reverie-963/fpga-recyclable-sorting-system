# 按当前位置预览抓取点

当前优先解决“每次必须把物体摆回旧位置”：识别当前物体，选择轮廓内有夹持余量的材料点，换算接近位和抓取位。先完成可用抓取区域及多位置对准，再继续整轮搬运测试。

`src.static_grasp_preview` 从本机已使用的 `auto_point_planner.py` 整理而来。它不使用旧参考中心距离，不调用 `static_live_pilot.check_observations` 的固定点门控；后者仍是旧示教路径的独立检查。新增入口仅计算坐标和画图，未替换现场执行器，没有相机采集、串口、动作计划或执行开关。

## 当前实际边界

- 仅同一65 mm表面高度奶盒、原朝向，固定相机和640×480图像；抓取/接近的名义模型Z分别为50/70，不能直接用于瓶罐。
- 连续帧须符合模型、相机、分辨率、置信度、时效、实例唯一性和稳定性要求；掩码必须完整落在源区。
- 抓点在同一实体组件内，避开孔洞，并同时满足最低边距和组件最大边距的80%（沿用现场策略）。在这些候选中选择最靠近质心、同时位于两高度已核对区域内的点。
- 不要求整个物体都进入小三角；**小三角只表示已核对的点域**，不证明整爪、连杆和物体的碰撞安全。
- 当前本机标定的共同点域在纸卡坐标为 `(-20,0)、(20,0)、(0,20)` mm，底40 mm、高20 mm。这解释了为什么现有试验仍对摆放位置敏感。它是定性核对范围，未验收毫米级精度。
- 标定相机身份或分辨率不符、任一高度缺少区域、空/退化区域均拒绝。相同相机ID不能证明机位没动，结果明确记录 `camera_pose_verified=false`；实时应用仍须独立复核机位。

## 接口与回放

库函数 `plan_target(rows, policy, calibration, now_ms)` 的观测和策略沿用本机现场JSON：

- `rows`：至少2帧，且数量等于 `stable_frames`；每帧包括 `camera_id`、`image_size`、递增 `frame_id`、同一单调时钟域的 `capture_started_ms`、`model_version` 和完整实例 `instances`。每实例含 `label`、`confidence`、`bbox_xyxy`、`mask_rle`。若有 `all_full_frame_instances`，以其检查源区歧义。
- `policy`：冻结模型/相机/图像尺寸、源区、类别、面积、置信度、内部边距、稳定帧数、最大帧龄及帧间抓点偏移。
- `calibration`：`checked_for_supervised_preview`、`surface_height_mm`、`camera_id`、`image_size`、`surface_image_to_card`、`grip_card_to_control`、`hover_card_to_control_left/right` 和 `verified_point_hulls_card_xy_mm`。至少提供抓取及接近两个高度的凸多边形点域，求交使用；本机左右比例纠正结果来自文件，不在代码中重新猜测。
- 输出 `pick_pixel`、`card_xy_mm`、`grip_xyz_nominal`、`hover_xyz_nominal` 和实际像素边距。控制坐标单位mm，仅为运动学模型名义量；无路径及物理精度结论。

在工程目录进行录制数据回放：

```powershell
..\.venv\Scripts\python.exe -m src.static_grasp_preview --observations <录制观测JSON> --calibration <对应高度标定JSON> --replay-now-ms <录制时的判定时刻> --output <预览JSON>
```

`--replay-now-ms` 必须显式提供，不能拿当前电脑运行时间与历史单调时钟混用。回放结果固定标记 `recorded_replay_only=true`，不是新鲜观测或动作放行。

可成对增加 `--image <最后一帧图片> --overlay <输出PNG>`。图片哈希须与最后一帧的 `image_sha256` 一致。绿色为两高度共同核对点域，红十字为本次选点。没有有效抓点时仍可显示区域，但没有红十字；拒绝原因写入JSON。

## 下一步验收

1. 固定当前相机、底座和箱位，确认当前样件及高度。目标是覆盖实际想摆放的区域；不要继续要求物体对准历史抓取点。
2. 在目标区域中心及边缘采集多点对应关系，分别核对接近位和抓取位。范围外先做受监督标定，不直接试抓。
3. 预留未参与拟合的点，记录实际对准误差和测量方法；据夹持余量决定可接受区域，不能把目测“对准”录成0 mm误差。
4. 在区域内至少选择左、中、右、近、远五种不同摆放位置；先看自动红十字，再核对张爪接近，随后分别验证下降和整段路径。失败位置保留原因。
5. 同一物体换位置后自动更新抓点、接近和抓取坐标，边界外明确拒绝；通过后才接入整轮试验入口。工位变化则重新核对映射。

## 本轮验证（2026-10-28）

运行 `..\.venv\Scripts\python.exe -m unittest sim.test_static_grasp_preview sim.test_contest sim.test_system -q`，44项通过、0失败、0跳过，其中20项为抓点专项，覆盖改变摆放位置会更新坐标、细微轮廓变化不沿长盒跳点、带孔物体、双高度区域交集、越界及空标定拒绝。根目录 `py -3.11 tools/check_repository.py` 和 `git diff --check` 通过。历史三帧回放复现 `[290,141]` 像素及原高度换算，并生成区域图；这不是当前工位或新一次机械实验。

本机输入仍在 `local-only/static-240-live-loop-20261028/`，本轮预览在 `local-only/grasp-point-preview-20261028/`，没有团队共享地址。代码可随Git交接，现场图片和标定文件不能假定随仓库同步。本轮未连接设备、采集新图、执行串口/机械动作、板端/时序/性能测试、综合或位流加载。
