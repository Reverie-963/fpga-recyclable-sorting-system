# 受监督夹爪间距标定

2026-10-31最新状态：用户在1600 / 9 mm处要求暂停，未批准1650扩展；随后“启用”仅授权接入已测9–61 mm间距表，不恢复机械标定。1500原14 mm读数已排除，复测22 mm；1525采用最新21 mm，原19 mm及2 mm差异保留。生产策略以独立启用记录绑定原测量/回执，不修改历史位置和路线资格，夹持力与吸管抓放仍待验收。接入接口见[轮廓测宽](contour_grasp.md)，实际部署结果见[记录](../../docs/team/reviews/2026-10-31-jaw-gap-calibration.md)。

`POST /operation` 的 `action=jaw_calibrate` 是空载中位的单步测量动作。沿用页面 token、owner、唯一 request_id、真实客户端 heartbeat、ready、模型绑定、板端所有权、STOP、pending、相机固定标记与串口回执检查。请求不重发；未知结果先查账本。

`jaw_calibration` 必须包含 `target_pwm`、`source_task_id`、`source_sha256`、`gap_mm`、`hands_clear=true`。间距是用户在实际夹持位置测出的两侧软垫内表面净间距，单位 mm；来源任务和哈希绑定上一条完整动作报告。测量后移开尺和手才能请求下一步。

初始历史必须证明全部关节回到已用中位和张开值。后续按已完成标定报告链恢复命令状态；回显不代表位置反馈。只发送 joint 5，单步变化不超过 100，时长 3000 ms，范围 1000–1500 且不超过现有最大闭合值。当前间距不大于 2 mm 时拒绝继续闭合。其他关节、机械限位和抓放配置不变。

2026-10-31测得1500时仍有14 mm，用户另行明确同意“继续小步标定”。新增可选绑定字段`jaw_calibration_extension`，kind为`operator-approved-jaw-extension-v1`，包含授权原文、`max_pwm=1600`以及1500实测来源任务和报告SHA-256。无此封存绑定仍不能越过1500。扩展只允许沿该来源历史继续，闭合每步最多25；间距≤8 mm时最多10，≤4 mm时最多5，≤2 mm禁止继续闭合。每步仍3秒，必须重新读数和移开手/尺；张开方向保留100步长并允许真实0 mm读数。扩展授权不等于实测行程、夹持力或吸管抓放验收，生产闭合表不自动启用。

完成状态 `jaw_calibration_command_completed` 只表示命令回显一致并完成等待，随后仍需用户测距。报告保留 `gap_mm=null`、`measurement_required=true`、`physical_grip_verified=false`；失败保留待复核记录，不自动重试。间距表和吸管实抓验收是后续独立步骤，不能由单步执行成功推导。

连续标定通过已验证的来源哈希链找到最初的工位参考图，各点均对它做固定标记检查，避免逐步更换参考产生累计漂移。每条新图像记录从实际像素提取`image_size=[宽,高]`，须与配置尺寸一致；旧记录原样保留，不补写历史字段或跳过图像/像素哈希检查。

本地模拟验证：`python -m unittest sim.test_jaw_calibration sim.test_sorting_neutral_return sim.test_board_station_host sim.test_monitor_operation sim.test_voice_height_registration -q`。
