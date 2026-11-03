# KM1 纯夹爪控制 V2

新增整臂V3源码与静态PYNQ适配器，见[队友B整臂说明](../../docs/team/MEMBER_B_ARM_V3.md)。V3尚未部署，现有设备仍运行张开实测通过的V2。V3安装入口为main_arm.py改名main.py，不能把V2客户端当作V3客户端混用；当前main.py文件仍为V2，便于保留已通过版本。

不加载相机、模型或HerbSort。程序启动和导入不发动作，必须收到单次许可及动作请求才转发。当前只开放5号夹爪，尚未实现整臂坐标抓放。

## 部署文件

复制同目录的 main.py、km1_gripper_bridge.py、arm_parameters.py、gripper_manual.py、kinematics.py 到OpenMV磁盘根目录。最后复制main.py，更新后停止旧程序并启动新入口，再核验V2握手；仅复制文件不证明新代码正在运行。

kinematics.py原样保存自2026-10-16实物备份，SHA256为a87c75f8d799b86dc809fbe5d3d3a8ccce9a577cbde1c502c5c29e73240c4d8b，来源于队友基于厂商KM1 V3.1改编的驱动；没有重写运动学。厂商参考原件不变。设备E盘不是ESP32固件磁盘。

arm_parameters.py复制原实物配置的开合参数：OPEN=1000、CLOSE=1700、MOVE=800ms。张开曾实物通过；闭合与整臂路径仍待观察确认。gripper_manual.py不再依赖herbConfig.py。备份原文件后，herbConfig.py可以移出设备，视觉脚本和模型均非依赖。

## 电脑操作

从仓库根目录进入pynq_km1_vision，然后使用根虚拟环境：

```powershell
..\.venv\Scripts\python.exe -m src.gripper_client status
..\.venv\Scripts\python.exe -m src.gripper_client open
..\.venv\Scripts\python.exe -m src.gripper_client close
```

status只查询状态、参数并取消软件许可，不发运动；open/close不带--execute只生成本地预览，甚至不打开串口。

现场确认供电、夹爪为空、周围无障碍并安排好实验后，单次执行：

```powershell
..\.venv\Scripts\python.exe -m src.gripper_client open --execute
..\.venv\Scripts\python.exe -m src.gripper_client close --execute
```

每次只运行其中一条，不连续粘贴执行。实际动作需要使用者观察，SENT_UNVERIFIED不代表成功夹持。失败或超时不得自动重试。

客户端自动核验OpenMV USB标识37C5:124A，拒绝把CH340当作此协议设备。也可显式传--port COM5；PYNQ可传实际/dev/ttyACM路径。原生OpenMV USB启用DTR，不改变底座CH340的启动引脚。客户端先核对V2握手、机械参数和状态，再取设备当前编号+1作为新请求编号；一次许可只用于一个请求，断开即取消许可，重启后请求编号从0开始，不自动恢复旧动作。

## 协议与边界

PING → READY KM1_GRIPPER_V2 NO_VISION。
CONFIG → CONFIG ID5 OPEN1000 CLOSE1700 MOVE800。
STATUS → STATUS LAST0 ARMED0（LAST为最近已消费编号）。
DISARM → DISARMED，取消后续软件许可，不是物理急停。
ARM → ARMED ONE_COMMAND，5秒有效。
OPEN 1 / CLOSE 2 → 使用设备参数；成功UART写入返回SENT_UNVERIFIED及编号。
保留GRIP id pwm ms兼容测试入口，仅ID5、1000至1700、300至5000ms，并执行同样的单次许可/编号检查。

V2转发全部经过原kinematics.Kinematics.send_str，采用与已通过试验相同的带大括号指令；OPEN对应{#005P1000T0800!}，CLOSE对应{#005P1700T0800!}。未单独证明此前失败由格式还是参数导致。

上电无运动仅指本脚本；ESP32底座自己的开机动作独立存在。程序停止、DISARM或USB断开不能撤销底座已经接收的运动。动作时间不等于实际到位反馈。最终动态抓取需要独立时序与路径验证。
