# 新类别准入与控制保活

## 邻物张爪范围改为提示（2026-11-01）

用户明确要求移除`unified_neighbor_in_open_gripper_footprint`限制。统一候选规划仍计算每个邻物与名义张爪矩形的关系及净距，但不因该矩形重叠拒绝候选。`choice.nominal_tool_footprint_check_enforced=false`明确记录该检查不作为门槛；`nominal_tool_footprint_clear`仍表示实际二维计算结果，重叠时为false，不能因放行而记为true。`nominal_neighbor_clearance_mm`保留，用于既有候选排序，重叠时可为0。

本次仅移除这一项拒绝。既有轮廓最小间距开关/配置、有效抓点、夹爪宽度/标定范围、完整路线编译、关节限位、同帧/模型身份、STOP、所有权、真实客户端保活和未知回执处理继续各自生效。当前现场轮廓最小间距检查此前已由用户关闭；不将它重新开启为替代门槛。接近邻物的碰撞风险增加，`full_link_collision_check=false`及物理未验收标记保留，不宣称无碰撞或无人值守可用。部署与无动作验证见[本轮记录](../../docs/team/reviews/2026-11-01-open-jaw-advisory.md)。

本次为已有原生 PYNQ 运行环境的修复，不是独立安装包。完整实验控制基座、机器绑定、模型和启动加速库尚未全部进入 main；不能只检出本 PR 就声称已重建现场。与 PR #60、#61、#62、#63 的运行环境组合使用。

## 控制接口

### 2026-11-01 启机恢复与历史照片尺寸

重启后USB端口编号可能交换。机械臂仍须按已绑定的VID/PID/序列号唯一识别，并由封存的机器配置声明当前设备节点；不能按ttyUSB编号猜设备。语音启动工具按ASRPRO既有1a86:7522身份选择唯一端口，不与机械臂1a86:7523混用。已有日志不可写时新建本次独立日志，不修改旧日志归属或权限；预检进程不算现役控制器。

早期动作守卫照片可能缺少observation.image_size。recorded_frame(pixels=True)仍先核验原文件SHA及解码BGR SHA；仅尺寸字段缺失、原帧号/流号存在且图像不超过640×480时，从这些已核验的原像素推导尺寸，返回副本标记image_size_source=hash_verified_decoded_legacy_record。帧号接受非负整数或原HTTP头中的规范十进制字符串；存在station_vision.frame_id时必须相符。显式错误尺寸、空值、哈希不符、帧身份缺失或冲突继续拒绝，不改写历史元数据/照片/报告，不构成舵机位置反馈或历史物理结果补验。

### 已知空爪标定状态的回初始

home_confirmed确认机械臂停稳、夹爪无物及现场清空后，home可处理完整哈希链及回显/等待证明的supervised_jaw_calibration状态：其余五关节必须仍在原初始指令值，夹爪仅接受既有1000–1600范围，先按每步最多100 PWM、3000ms只张爪至1000，再执行原5000ms回初始段。无需把夹爪已张开作为重置前提，不直接发送闭爪姿态的全关节回程。

张爪与回程属于同一个任务/独占权/账本事务，每步均保留新鲜场景、停止、真实客户端保活、原串口回显及完整等待；中途失败保持pending和已有运动事实，不自动重试。未知姿态、其他关节变化、来源哈希/回执不明、缺空爪现场确认继续拒绝。预演返回command_count、empty_jaw_opening_steps，duration_ms为整条恢复计划的指令时长之和；它不是含采图/检查的实际总用时。报告增加empty_jaw_recovery，既有neutral_return_command_completed成功状态不变，不代表位置传感反馈。

`Operations(..., defer_preflight=True)` 用于原生板端入口；默认 False 保持旧调用行为。start/preview/home 先建立唯一任务，再在工作线程完成原有预检。预检不持有状态/保活/停止共用锁；物理执行仍须通过新鲜预检、取消状态、真实客户端租约及板端独占权检查。`preflight_checks_deferred` 表示这种调度方式，不表示省略检查。

页面独立每 2 秒提交当前 owner/operation_id 的 heartbeat，确认 `heartbeat_accepted`；页面退出停止，不发 start，不生成服务端假保活。状态 GET 不再控制保活节奏。客户端状态超时 8 秒，代理控制通道建立 3 秒、响应 4 秒，原有 12 秒租约、图像时限及机械限制保持。代理不重连或重放动作。完成后的 `/board-preview-report` GET 使用独立控制连接，避免被监控画面容量占满。

## 连续执行已确认路线

2026-10-31 用户要求取消逐条检查。`motion_scene_check_mode: before_sequence` 使正式 prompted 单件入口先完成准备、目标/路线检查，并在串口打开后、首条 home 写入前取得新鲜场景；此后按完整冻结路线连续执行，不再为每个舵机小步骤取图、跑标记/几何检查或扫描全部源文件。结束后仍执行现役简易箱内核验。没有该字段时默认 `per_command`，显式启用连续模式必须有 operator_statement。

首条指令仍必须带原控制器提供的新鲜检查，超时或缺失时禁止首次写入。每条写入和等待中仍检查停止信号、独占权/真实客户端租约，并保留原数据包限制、串口回显、完整运动等待、未决记录及禁止重放；不会把整段指令一次性排入设备而失去停止能力。途中不重新观察目标或机位变化，这是用户明确选择的行为改变，不是视觉保护仍逐条存在的模式。初始定位/规划等准备仍有耗时。

### 一次发令处理当前登记批次

2026-10-31用户确认“当前画面整批分完，最后统一核验”。提示模型入口支持`automatic_batch_enabled`，须有`batch_final_operator_statement`，默认界限`batch_max_items=6`、`batch_max_seconds=1200`，不允许超过这些上限。执行入口要求`height_plan`逐件登记；空旧配置仍保留单件行为。材质口令只处理登记池中符合该筛选的物体，AUTO处理所有已准入材料。

`PromptedBatchController`首帧对选中目标全部规划，任一初始候选不可规划则在动作前拒绝；每件复用同一实例账本和整批设备租约，后续目标仍须新鲜观测、独立高度、完整路线、动作前场景及STOP/真实心跳检查。已尝试目标记入账本，掉在源区也不自动重复抓取；新出现物体不扩充本批目标。逐件回显与完整等待确认后才继续下一件，逐件结果核验关闭，最后调用`verify_batch_bin_occupied`取得一张新鲜结果图，对所有已访问箱核对原简易占用条件。严格旧批末语义核验入口保持。

汇总报告保留各子任务及其哈希、全部plan/steps/回执、`motion_completed_count`、`completed_count`和`final_verification`；整批动作和最终核验全部完成时，外层使用兼容的`completed_visual_evidence`，原始`batch_status`及降级核验scope分别留存。最终失败或部分执行中断记needs_review并留batch-pending；不据此声称物理逐件成功。状态返回`batch_mode=true`、`batch_verification_mode=end_of_batch`、`skip_per_item_visual_verification=true`。进度命令数跨件累计，初次连接/准备后保持整批动作阶段，只有最终核验进入verifying；统计钩子不发动作。

状态新增 `motion_scene_check_mode` 和 `per_command_visual_checks`。任务报告 `motion_sequence` 记录实际入口模式、首条视觉检查调用数及末端核验要求。不会因为切换模式自动解除已有 pending/blocked，也不追认旧失败通过；此前 163 秒实测不能作为新模式速度证明。

## 显式试用范围

`operator_trial_scope` 仍限定当前模型的 26 类及既有纸/塑料/金属映射，且单次人工监督；原 Drink can 实物记录不扩写成 26 类验收。

`live_object_admission: true` 只在显式 operator_supervised_trial 范围中生效，转换为内部 `operator-live-geometry-v1`。候选及放置核验不再要求符合原样件面积/长宽模板。候选仍需非空有效轮廓、主连通分量、内部抓点；实测宽度必须小于夹爪开口减两侧余量，完整路线仍须编译成功。近圆轮廓使用零轴，并按该轴计算实际宽度。不会自动修改高度、舵机范围、测过的释放路线或资格标志。

禁用全场实例匹配时，动作阶段只跟踪冻结的被选物体材料和轮廓 IoU；按用户2026-10-31要求，同材料物体的具体分类名称变化不再导致拒绝。新增背景类别的数量波动不再重新启用全场匹配。目标材料变化、缺失、移动或匹配歧义仍拒绝，最新其他掩码仍用于已有障碍检查。当前用户先前关闭的箱空检查等选项继续按原配置执行，本次不宣称完整机械碰撞证明。

报告 `choice.specimen_admission` 记录实测像素面积、毫米宽/长、使用的轴、模板是否必需及物理未验收。操作摘要保留串口打开/写入尝试字段。新类别准入不等于新物体整轮抓放已验证。

## 与启动优化组合

协调器可读取 stage.bootstrap 的相对路径及 SHA，校验后运行候选入口。`board_runtime_bootstrap.py` 先验证冻结清单，再装入既有启动加速扩展，最后启动原生主机；保留原持久状态与独占数据库。禁止用新数据库清空未决状态。CPU native forward projection 的库文件与机器配置不进入此 PR，也不能计为 PL 加速。

## 人工复核（2026-11-01）

### 已完成源区下降后的空爪恢复

2026-11-01后续修复：最新物理意图报告若主机明确记录board_uart_write_attempted=false且steps为空，不会覆盖更早的完整回位姿态；子任务必须由同batch_id整批清单的精确报告SHA证明，不能仅据空steps或无回执跳过。回位报告记录被跳过记录及SHA，原报告不改。单件RGB源区域重绑定必须保留独立pickup_source_roi，避免轮廓跨过裁剪边缘误报整批数量变化；首动仍按完整工作区验证全部原登记物体。

冻结批次后继件的无语义身份/布局检查使用原站点source.raw的无损PNG，同一取帧仍绑定原哈希、流号、帧龄和几何守卫，前后检查执行身份。不得使用供轻量标记守卫的JPEG motion.camera假作RGB校验帧；JPEG保持拒绝，不通过重编码伪造原PNG身份。该路径不作新的类别推理、选件或路线重算。

2026-11-01提示目标的粗箱区边缘判定：仅重心位于不可变取物区、位于箱区外、原掩码不侵入箱区5%保护内核且带原padding覆盖不超过箱区5%的目标，可在边缘进入后续原路线准入。此为独立固定正常数值的几何分支，不继承旧扩容factor3，也不修改箱区占用、障碍掩码或其他机械检查。choice.source_bin_edge_reviews记录每次证据，source_bin_overlap_fraction仍保留真实重叠。箱内中心、内核侵入、大范围或非取物区目标仍拒绝，不能将此判定当作实测物理避碰。

`home`新增严格受限的源区回程：仅接受known_physical/needs_review、board_control_blocked，前六段home/source_high/align_wrist/source_hover/source_mid/source_grip均有完整且匹配原计划的回显与等待，末段只能是无回执的夹爪通道close。其他关节未知、路线改变、更多未决指令或其他失败原因均拒绝。保留闭爪结果未知，start_pwm的夹爪项为null，不声明未写入或实际张开。

操作员须重新确认机械臂停稳、未断电/手动挪动、空爪、双手工具撤出。先按显式恢复流程归档原pending/STOP，再发新的home请求；首段仅将空爪打开至原1000目标，随后复用本次计划的反向mid/hover/high、front_high、恢复腕向、准备位及原5秒回初始段。反向前三段必须逐值等于原下降段，回程段各3秒，原目标PWM和限位不变。原失败记录、未知闭爪回执均不追认成功；每段仍要求原新鲜场景/标记、当前所有权、STOP、串口回显和完整等待。完成仅说明指令回显/等待及现场核对，不宣称舵机位置反馈。重置审计措辞与页面的空爪确认一致，不推断夹爪已张开或已经在初始位置。

状态`manual_review`提供`supported/required/available/owner_epoch/request_id/recovery_required`。页面动作`review`携带独立请求号、`expected_operation_id`、`ready=true`、`review_confirmed=true`、`review_epoch`及`review_request_id`，经原会话、所有权、重复及忙碌检查。操作员必须确认机械臂在初始位置、夹爪张开且无物、双手撤出；软件不提供位置反馈证明。

该入口只归档已结束的`blocked/needs_review`任务：持有所有权事务及Journal锁，先刷盘复核清单及原pending文件SHA，再原样归档pending/batch-pending、写入复核审计并释放该任务所有权；失败恢复本次原文件。原报告不追认成功，STOP不清除，无串口、home/reset或自动发令。活动/归属不明所有权、设备租约、controller.lock以及记录中的未知动作结果/回显失败/未完成等待均拒绝。页面确认时任务身份变化也拒绝，不自动重试。详见[记录](../../docs/team/reviews/2026-11-01-manual-review-button.md)。

## 验证环境

`sim.test_runtime_admission` 的 3 项控制并发测试可独立运行；另外 6 项使用现有原生分拣基座和 `sim.test_sorting_unified` 夹具，缺失时明确跳过。完整现场集成环境 9 项均运行；跳过不得计作通过。详见本次[验收记录](../../docs/team/reviews/2026-10-31-runtime-admission-and-heartbeat.md)。

## 2026-10-31 简易终态检查

### 8mm薄物受监督试用

用户在确认吸管实测高度10mm后，明确要求塑料夹取平面及最小夹持间距均降到8mm，并在未测8mm开口的情况下授权按比例外推。`prompted_grasp_override.requested_grasp_planes`仅在已有`operator_supervised_trial`范围及独立操作说明下生效；原绑定准入先通过，再派生塑料`grasp_z_mm=8`及源路线末段8mm，其他机械路径不改、不继承新高度实物验收。

`requested_jaw_gap_extension`以`operator-supervised-linear-extension-v1`描述允许下限8mm；运行策略`jaw_gap_extension`仅允许在原实测最低点下方最多1mm线性延伸。原表仍为9mm@1600、13mm@1575等10个实测点，不添加假的8mm实测样本。用相邻最低两点线性外推，8mm目标约1606 PWM，仍校验500–2500协议范围；低于8mm拒绝。范围内继续原测量插值，20%收紧公式保持。外推报告`mode=estimated_width_extrapolation`、`linear_extension.measured=false`及`physical_validation_completed=false`，不能写成已测抓取精度或接触反馈。

8mm是允许下限，闭合目标仍为本次轮廓实测宽度×0.8。例如这根吸管原图的投影宽度约15.26mm，闭合目标约12.21mm/PWM1580，走原测量插值；不会因物高填10mm就把横向宽度替换成10mm或强制闭合到8mm。实际夹持效果未通过原图预演获得验收，须按现场实物核对。

当前 prompted 单件入口默认使用[简易箱内检查](simple_bin_verification.md)，替代上文历史同类新增实例与源区消失等终态要求。任意类别箱内检测或本轮动作前后箱内明显变化即可通过，报告单独标明此模式；动作保护与历史失败记录保留。
