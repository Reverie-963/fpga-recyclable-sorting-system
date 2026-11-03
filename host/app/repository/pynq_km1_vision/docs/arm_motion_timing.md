# 机械臂分阶段时长与连贯动作预演

2026-10-29当前实际三类单件入口已接入[三类闭环快档](sorting_motion_speed.md)，包含一次奶盒14条/29.703秒真实执行和末端核验未通过的记录。下文旧12步例程及离线工具不代表当前入口已自动使用这些旧时间表。

当前冻结实物计划的离线动作合并入口见[动作合并与提速预演](arm_route_preview.md)。该工具直接读取已保存的实际关节指令，支持保留避让点、最少指令子序列和可选的运动/回显重叠等待；不执行设备动作。2026-10-29的17步→14步结果及时间预算见[离线记录](../../docs/team/reviews/2026-10-29-arm-route-preview.md)。下文12步示例与实际17步路径属于不同入口，预算不能混用。

新增 arm.step_move_ms（逐步骤毫秒）和 arm.step_settle_s（逐步骤额外秒数），字段缺省时沿用原 move_ms/settle_s。规划时将每步新时长写入实际协议 T 字段，执行和 inspect_plan 使用同一等待规则，不能只缩短主机等待而保留旧 T 时长。

快速候选配置：config/arm_pc_fast.example.json。默认12步运动及停稳合计21.6秒，候选14.4秒；直连开串口仍需2秒，故不含通信/计算的名义整轮约23.6→16.4秒。此为软件预演，不是实物成功率、速度或连续轨迹验收。

在工程目录只预演：

```powershell
..\.venv\Scripts\python.exe -m src.arm_pick --config config/arm_pc_fast.example.json --xy 0 160 --label cap_deformed
```

候选普通移动段额外停稳为0，descend保留0.1秒、close/release保留0.2秒；每步完整运动等待仍保留。短移动和夹爪可分别配置，home和长距离搬运不统一套最短时长。逐步停车和控制器插补特性仍可能影响平滑度；这不是连续轨迹混合算法。

字段名必须来自home/wrist/open/above_target/descend/close/lift/above_bin/lower_bin/release/leave_bin/return；时长整数300..5000，停稳有限数值0..10，布尔值和未知步骤名拒绝。整轮依然只打开一次串口，传输故障不重发、不发后续动作；保留原直连上电等待与机械限位。

快速文件为示例：commissioned=false、端口占位、既有示例箱位。原本机配置、默认配置、生产速度和commissioned均未改动。队长须把两个时间映射合入实际已标定且已验证的本地配置副本，先预演检查每步T字段与wait，再按其现场授权范围逐步验证，不能直接把示例坐标作为生产标定。没有现场启动命令前，不声称这个入口就是造成现场慢动作的实际入口。

当前仅作用于静态 src.arm.plan_pick/execute；不改旧动态调度、固件、串口协议、关节位置或运动步骤顺序。模拟串口与sleep验证了整段运动等待、一次连接、异常停止和旧配置兼容，未连接真实设备。

## 更快候选：分级现场试验

工具 `src.arm_speed_trial` 从实际已标定配置复制生成预演配置，只覆盖时间映射及关闭commissioned，不覆盖坐标、箱位、串口、姿态和限位。输出目录必须新建，防止覆盖源配置/已有试验。

| 步骤 | 原快速方案 ms | brisk ms | candidate ms |
|---|---:|---:|---:|
| home | 1500 | 1500 | 1500 |
| wrist | 800 | 600 | 500 |
| open | 800 | 600 | 500 |
| above_target | 1200 | 1000 | 800 |
| descend | 1000 | 900 | 800 |
| close | 1200 | 1000 | 800 |
| lift | 1200 | 1000 | 900 |
| above_bin | 1500 | 1300 | 1200 |
| lower_bin | 1000 | 900 | 800 |
| release | 1000 | 900 | 700 |
| leave_bin | 1200 | 1000 | 800 |
| return | 1500 | 1300 | 1200 |

三档均保留下降0.1秒、夹紧/释放各0.2秒停稳，其余额外停稳为0。运动+停稳预算分别14.4 / 12.5 / 11.0秒；冷启动直连串口另加2秒及实际传输/处理时间。home保留1.5秒，因为初始姿态未知。candidate只是较快候选，不是测得的物理最快值，不用于任意距离/负载/姿态。

从工程目录先预演（以下明确使用示例配置，仅验证工具）：

```powershell
..\.venv\Scripts\python.exe -m src.arm_speed_trial --config config/arm_pc_fast.example.json --profile brisk --xy 0 160 --label cap_deformed --output-dir ../local-only/arm-trial-brisk
..\.venv\Scripts\python.exe -m src.arm_speed_trial --config config/arm_pc_fast.example.json --profile candidate --xy 0 160 --label cap_deformed --output-dir ../local-only/arm-trial-candidate
```

现场应替换为队长自己的已验证配置、真实取物坐标及类别；不得用示例坐标代替实测。生成 config.preview.json 和 report.json，保留实际计划和时间对比；没有--execute参数，不会导入串口或发送动作。输出 commissioned=false 是预演状态，程序不会替队长启用设备。

队长试验顺序：

1. 保持实际坐标/姿态/夹爪力度不变，先确认当前14.4秒档可用；记录实际启动命令与配置版本。
2. 预演brisk，按现场既有授权先空载分段检查，再用同一样件完整试验；观察下降、夹紧、抬升、转向箱沿、释放和退离。
3. brisk可用后再分段试candidate，初次不直接连续自动循环；某段出现过冲、抖动、碰边或物体滑动，就恢复该段上一档的运动时间和停稳，不把全部步骤一起加速。
4. 最终候选至少逐次记录10轮：配置哈希、物品、总耗时、每段开始/结束与等待、是否抓空/滑落/碰边/未释放。10轮通过仅作本轮筛选，不能当长期可靠性结论。
5. 队长根据记录选出最快可靠组合；若需要更快，再讨论基于已知姿态跳过重复初始化和缩短实际路径，不能在未知起点直接删除home，也不能发后续指令覆盖未完成动作。

11秒档仍不是连续轨迹插补。逐段指定T只改变到达时长，是否停顿、加减速曲线如何受底座固件影响；不得据预演承诺实物11秒完成或完全无停顿。现场若使用固定实验脚本而非src.arm.plan_pick/execute，需要把同一时间表移到那个真实入口，单独更新本工具不会让其他脚本自动提速。
