# 模型选择与机械执行联动

中控启用 `model_motion_follow` 后，模型菜单一次选择同时更新识别和 PS 执行配置。支持目录中登记的三类、24类、26类及其调参/参考图配置；切换不会发送机械指令。

## 切换与任务语义

1. 验证登记权重、配置、GPU身份和实际现场识别结果，为候选创建不可变视觉路由。
2. 在 PS 空闲且不存在设备占用/未决阻断时，按绑定版本提交一次切换。
3. PS 原子持久化新绑定并返回对应请求号，之后页面才发布新选择。失败保留原选择；回执丢失只读核对，不重复提交。
4. 无法确认提交结果时保留候选服务并阻止新动作。已开始的任务使用冻结的模型、阈值、类别、路由与版本，不能中途换模型。

每个配置独立保存模型哈希、上游配置哈希、策略及设置哈希、类别到材质的映射、阈值和参考图库身份。候选和原有服务返回的检测实例统一通过所选策略补全 `material`，保留原始 `official_label`、`model_material` 和参考辅助证据。不能将新模型的结果重标为旧模型身份。

三类使用登记的三类映射，24类使用其现役策略，26类使用各自固定类别表。当前登记阈值分别为三类/常规26类0.30、24类0.60、新版调参及参考辅助0.10。界面显示与执行使用同一配置，不能仅改变显示阈值。

当前各材质高度、箱位、加速运动、首条指令前检查和简易箱内核验保持现役配置。新模型的准入属于用户授权的受监督试用，不继承原模型的实物验证成绩。相机定位、停止、控制权、串口回显及未决保护继续生效。

## 接口

- `GET /execution-model`：读取 `profile_id`、`profile_sha256`、模型/配置/设置/策略哈希、`revision`、`route_id` 和上次 `request_id`。
- `POST /execution-model/bind`：需要 `profile_id`、`profile_sha256`、`expected_revision`、唯一 `request_id`、已验证的 `route_id`。使用已有控制链路认证；不取得执行租约、不开始动作、不清理历史记录。
- `/operation` 和 `/prompted/plan` 接受 `execution_profile_sha256` 与 `execution_binding_revision`；中控提交预演/开始时携带它们。PS拒绝过期的页面选择。
- 中控 `/model-switch/status` 返回 `bound_model`、`motion_model_matches`、`execution_binding` 和 `execution_binding_unknown`。绑定不一致时阻止新动作，停止/保活仍可用。

PS配置来自冻结 `execution-models.json`，绑定保存在权威状态目录 `execution-model-binding.json`。PC视觉路由只接受登记的本机模型端口和图像请求，具有独立随机路由身份，不提供机械接口。代理重启时仅按PID、进程创建时间和完整模型参数接管自己登记的工作进程。

ASRPRO口令也可按相同绑定发起一次单件start，详见[语音启动及拒绝处理](voice_single_start.md)；升级不自动清除已有主动停止或待复核。

实际任务写入所选模型的登记置信度，规划器在相同登记上下文中接受该精确值，包括0.10。加载基础机械策略时仍允许原范围内的基础阈值；没有登记上下文或不等于登记值的阈值继续受旧范围约束，其他决策及运动阈值保留。页面操作补入模型绑定后必须同步更新请求正文长度，避免板端只收到截断JSON。

首写入画面时间戳从StationCamera已验证的 `_raw()` 结果绑定，语义next和无识别raw共用该绑定。失败或控制权失效不刷新，原帧龄限制及首写入检查保留；before_sequence期间不为后续每条指令新增取图。

## 配置与验证

使用 `tools/prepare_execution_models.py` 从模型目录及原策略/设置生成配置。通过 `tools/serve_board_station.py` 的验证阶段部署；阶段可显式指定已经校验的 `station_config`，以保留现役联合FPGA配置。电脑专有路径、权重、图库、令牌和现场状态均不提交Git。

相关测试：

```powershell
# 仓库根目录
python -m unittest discover -s tools/tests -p test_model_motion_follow.py -v
python -m unittest discover -s tools/tests -p test_switchable_console.py -v
# 主工程目录
python -m unittest sim.test_execution_models -v
```

真实部署与验证边界见[本轮记录](../../docs/team/reviews/2026-10-31-model-motion-follow.md)。
