# 输送带与动态拦截配置

**v6任务入口：**两种零件、六种状态，详见[开工手册第8节](commissioning_roadmap.md#8-第六步两种零件六种状态的外观缺陷检测)。根配置尚未自动迁移，先完成第8.2节；4种缺陷可抓取，2种正常不抓，末端直行仍需复核。

版本v5。本文件与 [Windows手册](windows_final.md) 配合使用。固定标定相机、低速直线输送带、正交编码器、PYNQ-Z2与KM1构成动态回收工位；输送电机由独立调速驱动器供电，当前软件不控制电机启停。

## 1. 机械与电气

- 输送带平整无明显打滑；编码器测实际带轮或测量轮位移，不能仅把电机转速当作物体速度。
- 编码器A/B须经适配到PYNQ允许的3.3V逻辑；5V/12V/24V信号不可直接接板。按实际器件设计电平转换、共地/隔离和抗干扰。记录每圈脉冲、四倍频计数和实际毫米/计数。
- 仅使用核对过的PYNQ-Z2接口引脚。项目不猜测PMOD封装管脚，也不提供虚假的即插即用引脚表。
- 独立测定相机观察凸包、机械臂可抓区域和其中更窄的动态抓取窗口；分类盒ROI位于输送带区域外。检查全轨迹与带边、支架、分类盒的间隙。
- 收集区承接机械臂忙碌或错过窗口的物体。通过上料间隔和限速控制密度，不能承诺KM1赶上任意带速。

## 2. 构建编码器接口

提供现场文件，例如 `C:\fpga\encoder_pins.xdc`，为顶层`encoder_a`和`encoder_b`指定实际PACKAGE_PIN及IOSTANDARD，按器件/板卡资料核对。不要把示意符号填作真实封装脚。构建：

```powershell
.\tools\build_windows.ps1 -BoardRepo C:\fpga\board_files -Kernel segment_open -EncoderXdc C:\fpga\encoder_pins.xdc
```

Vivado同时加入HLS图像核、AXI GPIO和`conveyor_encoder.v`。保持PL时钟100MHz。使用同次生成的bit/hwh，旧v4位流没有编码器接口，不能用于v5。

AXI GPIO通道1为32位输出命令，通道2为32位读数据。GPIO寄存器偏移0/4/8/12对应DATA/TRI/DATA2/TRI2。命令bit8翻转触发快照，bit3:0选寄存器；等ack后读出的计数、时间、错误量属于同一快照。

| 索引 | 返回值 |
|---|---|
| 0 | 固定协议标识0x4b4d3501 |
| 1 | 已确认的快照toggle，0或1 |
| 2 | 32位有符号位移计数，Python按模2³²累积增量 |
| 3、4 | 快照100MHz tick低/高32位 |
| 5 | 非法正交跃迁累计数 |

RTL做两级同步，不内置适配所有编码器的去抖。信号跳变需远慢于采样时钟、满足电气质量；每次软件读取间计数变化须小于2³¹。实物转动正反向、已知距离、故障注入后再验收。

## 3. 根config.json需现场填写的项目

| 配置 | 含义 |
|---|---|
| encoder.mm_per_count / direction | 四倍频后的实际位移尺度、正向符号 |
| dynamic.axis_xy | 标定后的机械坐标系内输送方向单位向量 |
| dynamic.pick_window_mm | 预计接触允许区[xmin,xmax,ymin,ymax]，在arm.pick_bounds_mm内部 |
| capture_delay_s / capture_uncertainty_s | 图像曝光至PS读取完成的实测平均修正与保守误差 |
| min_speed_mm_s / max_speed_mm_s | 实际验收的正向带速范围；默认1～8mm/s仅为占位 |
| max_accel_mm_s2 | 工作状态允许加速度上界，必须实测验证 |
| grip_ms / grip_contact_s | 闭爪命令时长及从发令到夹持接触的实测时间 |
| actuation_uncertainty_s / schedule_slack_s | 舵机接触时序波动与软件调度允许迟到量 |
| position_uncertainty_mm / max_error_mm | 标定等位置误差与实际夹取容差 |
| association_mm / semantic_max_age_s | 编号关联门限、类别有效时间 |
| arm.bins / verification.bin_rois | 每个类别的放置坐标与对应图像盒区；同一回收流可共用 |
| dynamic.commissioned / arm.commissioned | 两项现场验收完成才设true；默认false |

预测提前量约为两次机械移动的时间及等待，加上夹爪接触时间。即使算法很快，机械臂仍有数秒提前量；必须保证观察距离大于带速乘提前量，再加完整物体尺寸与误差余量。若窗口无可行拦截，程序继续观察，不强行抓取。

## 4. 程序运行与证据

按Windows手册训练六状态缺陷检测模型并运行detect_server；在PYNQ完成上述配置与PL部署后：

```bash
python3 -m src.run --config config.json --frames 3000
python3 -m src.run --config config.json --execute --max-tasks 20 --frames 3000
```

不带execute只输出预测计划，但仍要求真实PL编码器和匹配位流。CPU图像对照设vision.backend=cpu，保留同一编码器与其他参数，避免同时改变感知和时基。Mac/Windows只跑YOLO服务；硬件控制入口在PYNQ。

日志包含tracking、semantic_result、intercept_plan、verification和stopped，含track_id、编码器快照、接触时刻、预测点及误差预算。把程序观察到的编号丢失与外部录像的漏抓真值区分；闭爪及放置阶段不保证连续视觉跟踪。

动态主程序的安全停止是停止发送后续指令，不等于硬件急停。程序不能撤销ESP32已经接收的运动，也不自动停止输送带。
