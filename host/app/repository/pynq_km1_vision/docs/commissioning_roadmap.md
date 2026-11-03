# 从设备到手到参赛演示：逐步开工手册

> 2026-10-26 路线更新：当前作品为“基于FPGA的可回收物智能分拣系统”，从[新的从零实施手册](../../docs/design/recyclable_sorting_from_zero.md)开始。本文保留瓶盖动态路线及底层操作参考，不按下方旧“当前先做什么”安排新项目，不直接将旧动态运行命令用于新三类静态分拣。

更新：2026-10-16。用户确认 AMD AI PC 已到位并接替原拯救者 Y9000P，后者已退出当前使用。HX 370/Windows 11 实机及 ROCm 基础检查见[环境记录](../../docs/team/reviews/2026-10-16-hx370-environment.md)。驱动更新及 GPU 复验已完成，当前先补齐本队权重并验证真实图片推理和板端通信；Vivado/Vitis 由用户安装，旧机安装路径和构建成绩不代表新机状态。其他器件沿用历史记录，输送带及编码器现状另行清点。

## 当前先做什么

**当前先完成 AI PC 的模型与通信迁移，再继续正常/变形两类动态链路。** 板卡启动、相机采图及 PL 图像核沿用已有历史证据；新机 ROCm 基础检查通过不代表本队模型或机械闭环通过。下文2026-10-07的 Y9000P/CUDA训练、推理和构建过程保留作历史复现；当前 AI PC 优先按[AMD部署手册](amd_deployment.md)操作，不执行 CUDA 环境安装。

| 启动介质清点 | 建议规格 | 说明 |
|---|---|---|
| microSD（TF）存储卡1张 | 优先32GB，Class 10/U1及以上；64GB也可作为备选 | 购买正规渠道产品，A1可优先；不要买成大尺寸SD卡。不需要为本项目专门购买高价U3/V30卡。容量/速度为工程建议，不是板卡最低要求。 |
| USB读卡器1个 | 支持microSD/TF，接口匹配电脑USB-A或USB-C | 电脑若已有可用读卡槽可不买；随卡附送的TF转SD卡套不是USB读卡器。 |

本次卡已写入Z2镜像，不再重复烧录。第3.1节保留为以后重制卡的参考；镜像写入与简单复制文件不同。官方启动流程要求已加载镜像的microSD，见 [PYNQ-Z2启动指南](https://pynq.readthedocs.io/en/v2.6.1/getting_started/pynq_z2_setup.html)。镜像下载从 [PYNQ官网 Boards and images](https://www.pynq.io/index.html)选择对应Z2的条目；不要按其他板型或“最新版本”字样选错镜像。

**执行方式：沿v6缺陷检测目标、复用现有v5控制器推进，已经通过且条件未变的结果直接复用。** 必要的电气、坐标和动作路径确认并入对应阶段；不反复安排独立开合夹爪练习。静态入口仅用于定位具体故障。

| 实际执行顺序 | 对应章节 | 阶段产物 |
|---|---|---|
| 1，已完成基础部分 | 第2～4节 | 写卡、启动、直连、上传、训练环境；未验收配件仍逐项确认 |
| 2，等带期间 | 第5～6节 | 相机采图、串口与接口核对；最终动作路线待机构固定 |
| 3，等带期间 | 第10～12节 | 编码器接口核对、构建位流、PL一致性、手转计数；毫米比例待安装 |
| 4，等带期间 | 第8～9节 | 六状态预采数据、标注、第一版模型、真实图片检测通信 |
| 5，传送带到货 | 第2、6、7、12、13节 | 固定工位、坐标/毫米比例/带速/延迟标定，补采带面数据 |
| 6 | 第14～15节 | 最终程序只规划、单件、批次、异常处理 |
| 同步准备 | 第17节与AMD部署手册 | HX 370环境；正式统计前迁移并重做实物闭环 |
| 7 | 第16～19节 | 指标、原始证据、冷启动复现、发布和比赛彩排 |

### 命令在哪里执行

| 标记 | 运行位置/目录 | 职责 |
|---|---|---|
| Windows / PowerShell | 当前 AI PC，仓库根目录 `C:\FPGA`，项目模块在 `pynq_km1_vision/` 运行 | 使用根目录 `.venv-rocm`；构建待本机 Vivado/Vitis 安装并预检 |
| PYNQ / Bash | SSH登录板卡后，`/home/xilinx/pynq_km1_vision` | 相机、标定、PL、机械臂 |
| Notebook / Python | 板卡Jupyter的Python单元格 | 先切到板端工程目录，再运行Python片段 |
| HX 370 | 已到位的当前 AI PC | 按[AMD部署手册](amd_deployment.md)复用现有环境并补齐验收 |

全文沿用固定直连目标地址：PYNQ为 `192.168.2.99/24`，当前 AI PC 有线口应为 `192.168.2.1/24`。旧 Y9000P 的实测 SSH 链路仅是历史证据，AI PC 按3.2重新核验网卡与连接。Wi-Fi地址不填入推理配置。`xilinx`账号按镜像调整。下文旧工程路径与 `.venv-yolo` 命令供历史复现；当前在 `C:\FPGA\pynq_km1_vision` 运行模块时使用 `..\.venv-rocm\Scripts\python.exe`，服务配置须匹配 ROCm 和本队两类模型。

板端 `config.json` 是现场工作配置。修改前备份，改完用 `python3 -m json.tool config.json` 检查语法。保存UTF-8，不用PowerShell 5默认重定向写JSON。现场标定后将配置备份回电脑，后续上传源码不能把实测值覆盖为默认值。

本文基于当前源码、配置和用户提供的AMD指南编写。2026-10-07核对命令入口及官方资料，2026-10-06工具检查是已有记录。本次文档更新没有写卡、接线、训练、综合或运行机械臂。数量和通过标准除明确引用指南外均是工程建议，不是赛方统一门槛或已有成绩。

**主线使用最终方案配置与入口 `src.run`。** 第6节接口确认并入最终工位搭建；第9节静态闭环保留为按需故障定位，不是默认必做的独立路线。最终分拣仍需完成现场标定、PL和编码器验证，再进入实际动作；不通过跳过校验或放宽门槛使缺件状态运行。

## 1. 先锁定参赛目标与完成标准

当前目标改为“基于FPGA与AI PC协同的轻质零件在线外观缺陷检测与动态剔除系统”（方案v6）。正式范围为两种零件、六种互斥状态：瓶盖正常/缺口/明显变形，带孔塑料片正常/缺角/缺孔。四种缺陷状态是抓取候选，正常状态不生成抓取计划；直行到末端的物体需复核，不能把漏检或抓取失败当作合格放行。样件选择、采集标注及配置迁移以第8节为准。

**迁移边界：**既有 `config/defect_board.json` 及 `data/defect_dataset.yaml` 为六状态迁移样例；当前仍先做8.0的两类任务。本次更新文档，不覆盖根配置、模型、实测标定或板端文件。已验收的板卡与 C270 记录保留；新 AI PC 的环境、网络和模型分别复验，不再将 Y9000P 环境作为当前依赖。

```text
C270 --USB--> PYNQ PS --DMA--> PL 分割/开运算 --DMA--> PS 几何和目标关联
                  |                                  ^
                  +--直连网线整帧--> PC YOLO --框/类别---+
SEN0230 A/B --> PL 四倍频计数/时间快照 --> PS 位移补偿与拦截预测
PS --USB串口--> KM1 ESP32 --> 舵机与夹爪 --> 分类盒图像 + 外部录像核验
```

PS 是板上的 ARM 处理器，执行 Linux/Python；PL 是 FPGA 可编程逻辑，运行 HLS/RTL 电路。C270 图像先进入 PS，再经 DMA 送 PL。不要将其描述成相机直连 PL 或 YOLO 在 FPGA 上运行。

参赛依据是 AMD 指南 **3.2 具身智能赛道**，不是 3.1 智能体赛道或 3.3 自主选题赛道。以下均为 PDF 阅读器物理页码，印刷页码减 3：

| 必须交付什么 | 指南依据 | 本项目怎样证明 |
|---|---|---|
| AMD Ryzen AI PC 上位机，ROCm 7.14.0 及以上 | 第13、15页，3.2.2.1/3.2.3.1 | 整机/处理器/内存/驱动与实际 ROCm 环境，真实模型推理 |
| FPGA 承担真实硬件功能 | 第13–15页 | 图像 HLS 核、编码器 RTL、DMA、实际位流与板端输出 |
| 感知到物理执行的完整任务 | 第13–16页 | 分拣录像、任务记录、成功失败及人工干预统计 |
| 五类核心指标 + 至少两项专项指标 | 第16–17页 | 见本手册第16节 |
| 工程、模型、构建、接线、报告与原始证据 | 第18–19页 | 从零复现包及 metrics 原始数据 |

评分为系统完整度20、端侧AI20、FPGA设计25、两端协同20、工程创新15。优先保证 PL 真实上板、协同和闭环；10G、多模态、NPU、大模型和 Agent 并非本项目全部必做项。Skill/Workflow 在本赛道为加分内容。

**中期时间：**提供的指南第20、26页写明 AI PC 支持申请为 **2026-10-09**，评审/发放从 **2026-10-13** 起。它是借机申请节点，不是最终作品截止日。本次访问赛事官网超时，未核实后续变更；队内安排一人核对[官方公告](http://www.fpgachina.cn/)，不要将计划日期当作新的赛事通知。

## 2. 启动介质到位后：整理最终工位与设备记录

先在 `metrics/evidence/` 建器件记录，逐件写“手中已见实物/待到货/型号待确认”。清点：Z2及适配电源、卡与读卡器、C270、KM1及原电源、网线和电脑有线网口、USB数据线/Hub、编码器及适配电路、输送带与驱动/电源、支架与灯、样品与盒子。TF转SD卡套不是USB读卡器。传送带未到先预留空间，暂不将桌面标定作为最终带面标定；本节SEN0230参数仅在实物型号核实后适用。

1. 记录板卡型号/版本、SD镜像、C270的USB标识、SEN0230机身线序标签、输送带型号/驱动/最低稳定带速，以及KM1固件版本。
2. 备好：已写卡或microSD与读卡器、板卡合规电源、原配KM1电源、网线和电脑有线网口、相机支架、漫射照明、标尺/标定板、分类盒和末端接料盒。
3. PYNQ USB HOST 通过可靠的有源Hub接相机和KM1；先单独接相机，后接机械臂。USB线要支持数据。
4. KM1和电机使用各自匹配的动力电源，PYNQ独立供电。SEN0230使用稳压5V；备两只10kΩ上拉电阻及万用表，借用示波器/逻辑分析仪验收脉冲。
5. 将相机固定在外部支架，工作区、抓取窗口和两个分类盒均可见。先用单色哑光背景，避开反光和机械臂投影。
6. 按原厂方法单独验证机械臂手柄和输送带调速。断开OpenMV向KM1发指令的数据通路，避免多个控制源同时发动作。

机电实际风险：停止Python不能撤销ESP32已经收到的运动，也不能关闭输送电机。桌边应能分别切断机械臂和输送带动力；联调时清空轨迹范围。当前程序没有硬件急停互锁或舵机位置/力反馈。

**留存：**工位照片、器件标签、供电和接线图，放 `metrics/evidence/`；新增文件名用英文。不要拍入登录口令。

## 3. 第一步：PYNQ 独立启动和联网

先不接编码器信号，也不运行抓取程序。

### 3.1 在Windows下载并写卡

**当前已完成，以下保留为重制卡参考。** 2026-10-07已对照本地 `D:\AMD_Downloads\pynq_z2_v3.1.1\pynq_z2_v3.1.1.img` 核对启动文件，详细范围与未验证项见3.2及检查报告。

1. microSD插入读卡器，再插电脑；打开“此电脑”，根据插拔前后变化记录设备和容量。卡上有需要保留的数据先复制出来。
2. 打开[PYNQ官方镜像页](https://www.pynq.io/index.html)，展开 **Previous downloads → Previous board images → PYNQ-Z2**。2026-10-07页面列出Z2 v3.1.1；选择Z2完整镜像，不选AArch64根文件系统、源码包或其他板型。
3. 下载后按写卡工具支持格式使用；不支持压缩输入时先解压到 `.img`。打开镜像写入工具，依次选择镜像、microSD目标盘，按容量再次核对。
4. 开始写入，等待写入和校验完成；这个过程清空所选卡。校验失败先检查卡、读卡器、下载完整性，不带着校验失败继续启动。
5. 安全弹出。Windows若提示格式化Linux分区，选择取消，不再格式化。
6. 记录文件名、下载链接、镜像版本、写卡工具版本和日期到 `metrics/evidence/board_setup.txt`。可用 `Get-FileHash -Algorithm SHA256 -LiteralPath '实际镜像路径'` 记录文件摘要；自算摘要用于追踪，官方提供摘要时再做比对。

### 3.2 上电、查IP和登录

**当前从这里做。** 2026-10-07电脑侧只读检查发现：microSD为磁盘1、约31.27GB（十进制），启动分区为E:，卷标PYNQ。BOOT.BIN、image.ub、boot.scr、boot.py、REVISION共5个文件与本地Z2 v3.1.1镜像逐字节一致，分区起点/大小一致。检查记录见[SD卡检查报告](../report/sd_card_check_20261007.json)。

这是启动文件及分区布局检查通过，不是整卡校验或上电成功：Windows当前权限拒绝读取物理盘，因此未核验卡上Linux分区内容；也未用官方发布摘要认证本地镜像。若烧录工具已提示写入/校验成功，保留其结果；目前没有发现需要重新烧录的启动文件问题。

Windows只显示约130MiB的PYNQ启动分区是正常现象，Linux分区不会作为普通盘符显示；不要因此格式化或重新分区。E:和磁盘1仅为本次检查时的编号，重插读卡器后可能变化。

**全项目统一采用网线直连。** 当前使用 AI PC 的实际物理以太网卡或 USB 网卡，先用 Get-NetAdapter 确认；旧 Y9000P 的“以太网 / Realtek PCIe GbE Family Controller”和 TAP 名称均是历史记录，不能照搬。此前192.168.2.99的成功 SSH 登录见[板卡启动检查](../report/board_boot_check_20261007.json)，不代表新机已经连通。Wi-Fi只负责电脑上网，其地址不参与板卡通信。

| 设备/功能 | 固定地址与配置 |
|---|---|
| 当前 AI PC 物理以太网（目标设置） | 192.168.2.1，掩码255.255.255.0，无网关、无DNS；需核验实际配置 |
| PYNQ-Z2有线网口 | 官方镜像默认直连地址192.168.2.99，需上电核实 |
| 浏览器/SSH/上传文件 | 访问192.168.2.99 |
| 板端请求PC识别 | http://192.168.2.1:8765 |
| 原 Y9000P | 已退出当前使用，旧网卡及连通记录仅作历史参考 |
| 电脑Wi-Fi | 保持原自动配置，仅用于下载、开发联网；比赛闭环不依赖它 |

安装所需文件和模型准备齐全后，采集、推理、PL处理、机械臂控制、日志回传都能在这条直连链路上运行。板卡临时安装缺失系统包见4.3.1；不要求把设备搬到路由器旁。

#### 3.2.1 安全取卡，板卡先断电

1. 关闭电脑上打开E:的资源管理器窗口和文件。
2. 在Windows任务栏“安全删除硬件”中弹出读卡器，或在“此电脑”右键PYNQ盘选择弹出，等提示可移除后拔读卡器。
3. 把microSD从读卡器取出。此后电脑看不到E:是正常现象；卡要插到Z2上，不能一直留在电脑读卡器。
4. Z2电源开关置OFF，暂不接相机、Hub、机械臂、编码器。
5. 板放在干净、非导电桌面上，避免金属物件触碰背面。

#### 3.2.2 找到两个跳帽，按丝印设置

先打开[Z2官方位置图](https://pynq.readthedocs.io/en/v3.1/_images/pynqz2_setup.png)，对应板上文字找位置。跳帽是套在针脚上的小塑料帽，用来短接相邻针脚；断电时调整。不要凭照片“朝左/朝右”猜方向。

| 板上选择 | 此次怎么设置 | 用途 |
|---|---|---|
| Boot启动选择 | 设为SD标记对应的针脚 | 从microSD启动，初次不用JTAG/QSPI |
| Power供电选择，USB供电时 | 设为USB | 从PROG/UART Micro-USB口供电 |
| Power供电选择，Z2外部适配器供电时 | 设为REG | 从Z2外部电源接口供电 |

**首次只启动裸板，可沿官方USB路线：Boot=SD、Power=USB。** 使用可靠的USB数据线，把电脑USB口连到板上标有PROG/UART的Micro-USB口；不是板上的大USB-A HOST口。若已经准备好与Z2手册匹配的独立适配器，则使用REG路线，供电规格/极性确认后再插入。KM1原配适配器不用于Z2。后续接相机/机械臂时按整套供电方案布线。

把烧录好的microSD插入板背面的卡槽，确认插牢。不要带电插拔卡或改跳帽。

#### 3.2.3 网线直连当前 AI PC，再上电

连接如下，数据网线不用经过路由器：

```text
互联网 <--Wi-Fi--> AI PC（下载工具、依赖、训练权重）
                    |
              物理以太网192.168.2.1
                    |
                  一根网线
                    |
              PYNQ-Z2 192.168.2.99
                    |-- USB HOST/有源Hub --> C270、KM1控制器
                    |-- PL输入 <-- 编码器（后续按电平方案接线）

首次启动：电脑USB --> Z2 PROG/UART；SD卡插Z2背面
正式整机：Z2、机械臂、输送带按各自合规电源供电
```

1. 网线一端插Z2网口，另一端插 AI PC 的物理以太网口（或 USB 以太网适配器）。电脑Wi-Fi保持连接。
2. 首次启动仍只接板卡，图中相机、KM1和编码器在后续阶段逐项接入。
3. 确认Boot=SD、供电选择匹配、卡插牢，再把Z2开关拨到ON。
4. 首次预留2～3分钟，观察电源灯、DONE与用户LED变化和网口链路灯。灯的状态只作参考，最终以登录成功为准。
5. Windows显示“未识别的网络/无Internet”可以是正常现象；这根网线只连接板卡，不承担电脑上网。

#### 3.2.4 给当前 AI PC 物理以太网设置固定IP

这一步只需首次设置，后续冷启动继续使用。不必查询路由器。

1. 按Win+R，输入 `ncpa.cpl`，回车，打开“网络连接”。
2. 找到名称“以太网”、设备说明“Realtek PCIe GbE Family Controller”的网卡。不要选WLAN、蓝牙、“以太网3/4/5”或TAP/VPN网卡。插拔这根网线时，目标网卡的连接状态应相应变化。
3. 先截图/记录该网卡原IPv4设置，便于以后恢复普通上网用途。
4. 右键该网卡 → 属性 → 选“Internet协议版本4（TCP/IPv4）” → 属性。
5. 选“使用下面的IP地址”，按表填写。若界面用“子网前缀长度”，填24。
6. 默认网关留空；DNS两项留空，不给直连口设置公网网关。点击确定保存。

| 项目 | 填写 |
|---|---|
| IP地址 | 192.168.2.1 |
| 子网掩码 | 255.255.255.0 |
| 默认网关 | 留空 |
| 首选/备用DNS | 留空 |

在电脑普通PowerShell检查：

```powershell
Get-NetAdapter -Name "以太网"
Get-NetIPAddress -InterfaceAlias "以太网" -AddressFamily IPv4
Test-NetConnection -ComputerName 192.168.2.99 -Port 22
```

预期：板已启动且线已连接时网卡Status为Up；本机IPv4为192.168.2.1、PrefixLength为24；最后TcpTestSucceeded=True。这只是SSH连通，YOLO服务还需第9节验收。

- 网卡仍Disconnected：先查网线、物理口、板供电。
- 本机仍169.254开头：通常是没有保存静态IPv4或选错网卡，回到属性核对。
- 本机地址正确但22端口不通：等板启动、检查SD/跳帽，用PROG/UART串口按3.3查看 `ip -4 addr`；官方镜像的192.168.2.99是默认值，不把它当作本次已实测值。若镜像网络已改过，先查实际配置，不盲改启动文件。
- 如果Wi-Fi/VPN也使用192.168.2.0/24，存在路由冲突；本次Wi-Fi为192.168.31.0/24未见该冲突。换网络后遇到冲突，先停用冲突VPN或换Wi-Fi，再测直连；不要继续把请求发到另一台设备。
- 网页有代理/VPN时，应让192.168.2.99和192.168.2.1走本地直连；不关闭整个Windows防火墙。
- 正常运行不打开网络桥接或Internet连接共享；只有4.3.1的临时依赖安装才使用共享。
- 以后该网口改接普通路由器时，按记录恢复IPv4/DNS自动获取；再接板卡则恢复本节静态配置。

依据：[PYNQ-Z2直连指南](https://pynq.readthedocs.io/en/v3.1/getting_started/pynq_z2_setup.html)、[官方电脑静态IP设置](https://pynq.readthedocs.io/en/v3.1/appendix/assign_a_static_ip.html)。

#### 3.2.5 浏览器打开Jupyter

1. 在地址栏输入 http://192.168.2.99。直接输入地址，不把它输到搜索框；首次先用http。
2. 出现PYNQ/Jupyter登录页时，官方v3.1指南默认账号为xilinx，密码为xilinx；如果页面只有密码框，只填写密码。如果该镜像或提供者改过口令，以其说明为准。
3. 登录后应看到文件/Notebook列表。能打开页面说明浏览器到板卡的网络已经通了。
4. 暂不打开项目抓取程序，不加载自定义位流；先做下文基础检查。

可以先用浏览器工作，不必接显示器、键盘到Z2。此时电脑只是通过网页和终端操作板卡。

#### 3.2.6 从Windows登录SSH并核对板端环境

在**电脑PowerShell**执行下列代码；使用本方案固定板卡地址：

```powershell
$PynqIp = "192.168.2.99"
Test-NetConnection -ComputerName $PynqIp -Port 22
ssh "xilinx@$PynqIp"
```

1. TcpTestSucceeded=True表示SSH端口可访问；False时先确认板卡启动/IP/网络，不反复输密码。
2. 首次连接出现主机指纹提示，确认IP对应自己的板卡，再输入yes并回车。
3. 输入板卡密码，默认镜像可尝试xilinx；终端不显示星号或字符正常，输完回车。
4. 提示符变为类似 xilinx@pynq:~$，才表示已经进入板卡。以下Bash/Python检查在这个窗口运行，不在本机Windows直接运行。

```bash
hostname
hostname -I
cat /etc/os-release
python3 -c "import sys,platform; print(sys.executable); print(platform.platform())"
python3 -c "import numpy,cv2,pynq; print('numpy',numpy.__version__); print('opencv',cv2.__version__); print('pynq',pynq.__version__)"
```

预期看到主机名、板卡IP、Linux信息和模块版本。最后一条若提示缺pynq，先在Jupyter新建Python Notebook，运行：

```python
import sys
print(sys.executable)
import numpy, cv2, pynq
print(numpy.__version__, cv2.__version__, pynq.__version__)
```

若Notebook正常而SSH报错，就是两处解释器不同。记下Notebook输出的完整路径，在SSH用该路径替代python3执行检查及后续板端命令。不要向Windows装pynq来解决板端问题，也不要为这个错误升级整个镜像。

**2026-10-07用户实测：**hostname=pynq，地址192.168.2.99；解释器为/usr/local/share/pynq-venv/bin/python3，NumPy=1.21.5、OpenCV=4.5.4、PYNQ=3.1.1。Linux内核为6.6.10-xilinx-v2024.1，/etc/os-release标记PynqLinux 3.0、Ubuntu jammy。系统发行标记和Python包版本分别记录，不能仅凭两者数字不同就判定烧录错误。原始字段见[启动检查记录](../report/board_boot_check_20261007.json)。

#### 3.2.7 跑一次基础示例并重启确认

本步骤在浏览器Notebook控制板上自带用户LED，不接机械臂/编码器。已有随镜像LED示例可直接使用；若不方便寻找，按下面新建一个检查Notebook即可，无需重复做两套。

1. 电脑浏览器打开http://192.168.2.99，登录。默认密码xilinx；有用户名框时也是xilinx。
2. 在文件页点New（新建）→ Python 3；若是JupyterLab界面，在Launcher中选Python 3 Notebook。将Notebook命名为board_check.ipynb。
3. 把下面第一段粘贴进第一个单元格，按Shift+Enter。只复制代码内容，不复制Markdown围栏，也不粘到SSH终端。

```python
import sys
import pynq
print(sys.executable)
print(pynq.__version__)
from pynq.overlays.base import BaseOverlay
base = BaseOverlay("base.bit")
print("Base overlay loaded:", base.is_loaded())
```

预期版本为3.1.1，最后输出True。它加载镜像自带的基础位流，不代表项目vision.bit已经构建。若报错，保留完整红色错误，不继续LED单元格。

4. 在下一个单元格运行下面代码，观察板上LD0用户LED亮灭5次，最后熄灭；不是观察一直亮着的电源灯或DONE灯。

```python
import time
led = base.leds[0]
led.off()
try:
    for _ in range(5):
        led.on()
        time.sleep(0.5)
        led.off()
        time.sleep(0.5)
finally:
    led.off()
print("LED sequence finished; confirm the physical LED blinked.")
```

只有看见实物按命令闪烁才算通过，打印文字本身不能证明LED工作。[官方BaseOverlay说明](https://pynq.readthedocs.io/en/v3.1/pynq_overlays/loading_an_overlay.html)

5. 保存Notebook（Ctrl+S）。在Kernel菜单选择Shut Down/关闭内核；关闭网页标签本身不等于停止内核。
6. 回到已登录的SSH窗口，在xilinx@pynq提示符后执行：

```bash
sudo reboot
```

7. SSH断开正常，等待约1～2分钟；回到Windows PowerShell重新连接：

```powershell
ssh xilinx@192.168.2.99
```

8. 同时刷新浏览器确认Jupyter能重新登录，并在SSH执行hostname -I确认仍为192.168.2.99。记录LED结果、重启后SSH和网页结果。
9. 三项通过后，继续第4节建立电脑环境并上传工程，再按第5节连接C270。当前不需要重刷SD、升级NumPy或构建项目位流来完成本检查。

**3.2通过标准：**Jupyter能登录、板端正确解释器能导入模块、基础Overlay能操作LED、重启后能再登录。用户已确认上述各项全部通过。本次只证明镜像基础Overlay工作，项目vision.bit、相机、编码器和整机闭环仍需后续验收。

### 3.3 留记录与排障

Jupyter打开 `Getting_Started` 或镜像基础示例，保存副本后按单元格从上到下执行LED例程。SSH执行 `sudo reboot`，等重新上线再登录，记录结果。任务结束执行 `sudo poweroff`，待系统停止后断电。

| 现象 | 下一步 |
|---|---|
| 无电源灯 | 检查开关、电源选择、规格和插接 |
| 有电但没有网络设备 | 检查SD启动、Z2镜像、写卡校验、网线、PC物理以太网及静态IPv4 |
| IP已知但网页打不开 | 核对PC为192.168.2.1/24、板为192.168.2.99、代理/VPN绕过本地地址；测试SSH |
| 网页能开但SSH缺模块 | 核对Notebook实际解释器，后续统一使用它 |
| 仍不能启动 | Z2调试USB接电脑，按官方说明用115200、8N1、无流控打开串口，保存启动日志 |

不要以只有电源灯亮判定启动通过。

## 4. 第二步：建立 Windows 开发环境并上传工程

Windows本地路径统一为 `C:\fpga\pynq_km1_vision`。Vivado工程避免放在中文路径或同步盘中。PowerShell命令在该目录执行；Bash命令在PYNQ的 `/home/xilinx/pynq_km1_vision` 执行。

### 4.1 建英文副本

2026-10-07已由助手创建C:\fpga\pynq_km1_vision英文工作副本，供当前上板使用。原工作区继续保存文档更新；以后同步前核对副本内新增数据和标定，不整目录覆盖。下面首次复制说明保留用于复现，当前无需再复制一次。

第一次用资源管理器把 `D:\work\嵌赛_FPGA\pynq_km1_vision` 复制到 `C:\fpga\`。目标若已存在，先核对里面的模型、现场配置、构建产物，不直接覆盖。文档更新保存在原工作区，实机工作前同步最新文档/源码到英文副本；联调开始后固定一个主工作副本。

打开该目录的PowerShell，运行 `Get-Location`、`Get-ChildItem`，应看见 `src`、`config.json`、`tools`、`build`。执行 `py -3.11 --version`、`ssh -V`；缺Python则用官方安装器安装3.11 x64，缺SSH则启用Windows“可选功能 → OpenSSH客户端”。

### 4.2 建Y9000P训练环境（历史复现）

当前 AI PC 跳过本节 CUDA 安装，按[AMD部署手册](amd_deployment.md)及[安装记录](../../docs/team/reviews/2026-10-16-hx370-environment.md)复用根目录 `.venv-rocm`。以下为旧机环境复现步骤。

2026-10-07初始清点只有3.13和2.7；用户随后已安装3.11.9并建立.venv-yolo，当前环境检查通过。下面首次安装步骤用于复现；已有环境无需再安装。HX 370使用独立ROCm环境，不照搬本节CUDA环境。

用户已选择跳过进一步相机画质调试，当前直接进入本节；照片已能显示，最终工位画质和长期稳定性不标记为已验收。

本机GPU实测为RTX 4070 Laptop、8188MiB显存，驱动560.94。2026-10-07把本项目CUDA包从cu128调整为cu126，保留torch 2.7.1/torchvision 0.22.1。版本依据：[PyTorch安装矩阵](https://pytorch.org/get-started/previous-versions/)、[NVIDIA CUDA 12.6驱动说明](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html)。这只是安装选择，最终以脚本GPU运算结果为准。

**当前操作：**

1. 在Y9000P浏览器打开[Python 3.11.9官方Windows 64位安装包](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)。本项目选择3.11系列；3.11.9是该系列最后常规Windows安装器版本，不宣称它是最新Python安全版本。
2. 运行安装器，保留pip，安装到当前用户。现有py启动器保留；不需要改变其他项目使用的Python或全局PATH。安装后新开Windows PowerShell。
3. 执行 `py -3.11 --version`，应显示Python 3.11.9。若找不到，先核对安装结果和py -0p，暂不运行依赖安装脚本。
4. 接交流电、保持Wi-Fi可用，然后在下面的英文工程目录执行安装脚本。依赖下载较大，出现错误先保存第一处错误。所有操作在电脑PowerShell执行，不在板卡SSH或Notebook执行。
5. 脚本依次创建独立.venv-yolo、安装固定版本依赖、检查包冲突、执行GPU卷积前向/反向和torchvision NMS。应看到 `GPU forward/backward and NMS: PASS` 及RTX 4070型号；失败不会报告已就绪。
6. 安装后才能继续标注数据、训练本队模型。环境验收不等于已经训练好识别六状态缺陷样品的模型。

**当前已完成：**用户安装Python 3.11并用逐条pip命令完成依赖安装（Windows执行策略阻止.ps1文件运行，未修改该策略）。pip check无冲突，用户GPU矩阵前向/反向通过；助手随后在本机验证GPU卷积反向、torchvision CUDA NMS、随机初始化YOLO11n对合成图像的GPU前向全部通过。详见[环境实测记录](../metrics/evidence/y9000p_environment/gpu_smoke_check.json)。这是环境检查，不代表本队六状态模型已训练或准确率通过。下方安装命令保留用于复现，当前无需重复安装。

安装与检查命令：

```powershell
cd C:\fpga\pynq_km1_vision
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
powershell -File tools/setup_windows.ps1 -Backend cuda
.\.venv-yolo\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))"
```

脚本安装独立 `.venv-yolo`，包含torch 2.7.1+cu126、torchvision 0.22.1及YOLO依赖。这是项目开发配置，需在你的Windows实机验收；若脚本策略阻止运行，使用[Windows手册第2节](windows_final.md)的逐条命令，不改全局策略。暂时只能CPU运行时使用 `-Backend cpu` 和 `config/host_cpu.json`；后续服务配置要与实际后端一致。

### 4.3 首次上传与板端依赖

首次采图不依赖电脑Python、torch或YOLO环境，可以先做本节和第5节，再完成4.2。上传命令在新开的Windows PowerShell里执行，先运行 `cd C:\fpga\pynq_km1_vision`；不要粘到xilinx@pynq的SSH终端。

首次上传源码与配置，不上传Windows虚拟环境；后续更新不得重复覆盖板上的实测 `config.json` 和 `data/calibration.json`：

```powershell
ssh xilinx@192.168.2.99 "mkdir -p /home/xilinx/pynq_km1_vision"
scp -r src config tools sim build docs data board report metrics skill config.json requirements.txt README.md xilinx@192.168.2.99:/home/xilinx/pynq_km1_vision/
```

板上执行：

```bash
cd /home/xilinx/pynq_km1_vision
mkdir -p metrics/logs metrics/evidence
python3 -m src.doctor
```

`doctor`是设备清单，不是完整通过/失败判定。板端缺torch、CUDA、Vivado是正常的；板端需要PYNQ、NumPy、cv2和串口库，PC需要torch/ultralytics等。直连默认不给板端提供互联网。板端缺 `serial` 或 `v4l2-ctl` 时先按4.3.1取得依赖，再按镜像环境补齐 `python3-serial`、`v4l-utils`，并确认安装位置对实际解释器可见。

可在PYNQ执行以下检查；安装命令仅在对应组件缺失时执行：

```bash
python3 -c "import serial; print(serial.__version__)"
command -v v4l2-ctl
# 对应组件缺失且已按4.3.1临时接通板端互联网时：
sudo apt update
sudo apt install python3-serial v4l-utils
```

如果apt装好但实际Python仍导入不到serial，先核对解释器；仅向镜像实际解释器补装pyserial，不执行PC的requirements_yolo清单。缺包、磁盘空间不足、网络失败分别保存错误，不把整个镜像环境升级一遍来试运气。

**验收文件：**电脑GPU检查输出、板端doctor输出、实际IP和解释器路径。板端执行 `ls src/run.py config.json` 应能看到文件。

### 4.3.1 直连状态下怎样补装板端依赖

先检查模块是否已存在；齐全就跳过本节。当前 AI PC 通过Wi-Fi下载依赖/模型；PYNQ通过网线拿源码/位流/文件。单纯网线直连不会把电脑Wi-Fi的互联网自动共享给PYNQ。

**仅缺pyserial时，可以始终保持192.168.2.x直连，离线补装：**

电脑在工程目录执行，使用4.2已经安装好的Python：

```powershell
py -3.11 -m pip download --only-binary=:all: --no-deps pyserial==3.5 -d board/wheels
ssh xilinx@192.168.2.99 "mkdir -p /home/xilinx/board-wheels"
scp board/wheels/pyserial-3.5-py2.py3-none-any.whl xilinx@192.168.2.99:/home/xilinx/board-wheels/
```

板端先核对Notebook的解释器路径，以下python3必须替换为该解释器（若两者一致直接执行）：

```bash
python3 -m pip install --no-index --find-links=/home/xilinx/board-wheels pyserial==3.5
python3 -c "import serial; print(serial.__version__)"
```

pyserial此wheel为纯Python通用包；不能照此把Windows版OpenCV/NumPy或x86二进制包传给ARM板。若镜像不允许该解释器直接pip安装，保留报错，使用对应系统包或镜像已有环境，不用强制覆盖系统包的开关。

**缺v4l-utils等系统包时，使用一次临时Internet连接共享（ICS）。线仍然直连电脑，开发/比赛运行地址不永久改动：**

1. 先记录第3.2节有线IPv4设置，关闭检测服务和板端任务。准备PROG/UART串口作备用登录，串口设置见3.3。
2. Win+R → ncpa.cpl → 右键正在上网的WLAN → 属性 → 共享，勾选允许其他网络用户通过此计算机的Internet连接来连接；“家庭网络连接”选择真实“以太网 / Realtek”。
3. Windows可能重设有线地址（常见192.168.137.1）。执行 `ipconfig` 查看实际值，不再假设此时是192.168.2.1。不要选择“网络桥接”，也不要选择VPN/TAP网卡。企业策略禁用共享或没有共享选项时，不能把本步骤当作已成功。
4. 重启PYNQ以重新申请DHCP地址。用PROG/UART串口登录后执行 `hostname -I`、`ip -4 route`，记录新分配的共享网段地址和指向电脑共享口的默认路由。板上可能同时保留192.168.2.99；此临时阶段用与电脑共享口同网段的地址SSH。若没有DHCP地址/默认路由，保存这两个输出排查，不把apt失败当成SD损坏。
5. 此时板端执行 `sudo apt update`，无DNS/连接/签名错误后再安装缺失的 `python3-serial`、`v4l-utils`。只补缺包，不执行整机升级，不下载Windows训练依赖。共享若仍无法访问外网，检查电脑自身Wi-Fi是否能上网以及VPN/组织策略；不要关闭防火墙来尝试。
6. 安装后验证导入/工具。退出SSH前在板端执行 `sudo poweroff`，系统停止后断电。
7. 回到WLAN“共享”，取消共享；按3.2.4将真实以太网重新设为192.168.2.1/24，网关和DNS留空。Wi-Fi保持原设置。
8. Z2重新上电，验证 `Test-NetConnection 192.168.2.99 -Port 22`；确认板端 `inference.url`仍为http://192.168.2.1:8765，再继续项目。临时共享地址不写入正式配置或成绩报告。

ICS是仅在缺包时采用的维护操作，实际菜单和能否共享取决于Windows版本/策略；不是比赛依赖。记录已装依赖版本，保留离线wheel及镜像备份。最终必须在关闭ICS、断开电脑Wi-Fi后重新验证第9节健康/真实图像请求及第14节任务，以证明运行不依赖公网。

## 5. 第三步：让 C270 在板上稳定采图

**先做这个，暂时不要训练模型或驱动机械臂。**

C270官方规格为固定焦点摄像头。先通过调整支架高度取得清晰画面，不能把软件自动对焦作为调试手段。项目初始统一 **640×480**；官方资料给出了该分辨率最高30fps，但真实板端帧率、曝光和USB格式要实测。[Logitech技术规格](https://support.logi.com/hc/en-us/articles/360023462093-Logitech-HD-Webcam-C270-Technical-Specifications)

板端只插C270，执行：

```bash
lsusb
v4l2-ctl --list-devices
v4l2-ctl --list-formats-ext -d /dev/video0
v4l2-ctl --list-ctrls-menus -d /dev/video0
```

**本次实测进度：**C270已被USB识别。首次采图失败是SSH用户xilinx不属于video组；添加组并重新登录后，采图命令连续输出5条saved。图片解码、实际尺寸、画质及长时间稳定性仍待检查，不能只凭saved就全部判定通过。见[相机检查记录](../report/camera_check_20261007.json)。

**若首次出现No camera，先检查SSH用户权限。** 在板卡SSH执行，不能放进网页Python单元格：

```bash
id
ls -l /dev/video*
sudo fuser -v /dev/video0 /dev/video1
```

若设备为root:video、权限crw-rw----，而id没有video组，则在同一SSH窗口执行：

```bash
sudo usermod -aG video xilinx
exit
```

回到Windows PowerShell后重新登录：

```powershell
ssh xilinx@192.168.2.99
```

在新的SSH会话执行id，确认有video，再用普通python3采图，不将整个项目长期以root运行，也不把设备权限改成所有人可写。本次重新登录后已解决此问题。GStreamer警告在本次后续成功输出saved的情况下没有阻止采集，但仍需检查实际图像；其他失败不能一律忽略该警告。

若相机在另一个video节点，替换命令并修改 `config.json` 的 `camera.index`。同一摄像头可能暴露多个节点，选择真正能采图的节点。根配置应为宽640、高480。

```bash
python3 -m src.capture --config config.json --count 20 --interval 2
```

输出完整图像到 `data/frames/`。Windows取回：

```powershell
scp -r xilinx@192.168.2.99:/home/xilinx/pynq_km1_vision/data/frames .\data\
```

资源管理器打开 `C:\fpga\pynq_km1_vision\data\frames`，抽查刚生成文件的修改时间和画面。每次换物体位置再采图，避免把旧图误认为实时结果。PYNQ检查最近图像的实际尺寸：

```bash
python3 - <<'PY'
from pathlib import Path
import cv2
paths = sorted(Path('data/frames').glob('*.jpg'))[-20:]
assert paths, 'No captured frames'
for p in paths:
    im = cv2.imread(str(p))
    assert im is not None and im.shape[:2] == (480, 640), str(p)
print('Checked frames:', len(paths))
PY
```

传送带未到时可用接近未来带面高度的临时平面验证画质，照片标记为预采数据。正式安装后重新检查视场、背景和曝光，不能沿用不再对应的几何标定。

逐张检查：实际尺寸640×480、样品边缘清晰、暗部可见、两个盒区可见、图像持续更新。再把采集数量改为300、间隔2秒，做约10分钟稳定性检查；这检验连续取图与存图，不等同于测出视频帧率。

曝光调试按 `--list-ctrls-menus` 的实际结果操作：先增加稳定照明；若支持手动曝光，再用 `v4l2-ctl --set-ctrl=<实际控制名>=<实际菜单值>` 关闭自动曝光并缩短曝光。不要照搬其他摄像头的菜单数字。必要时锁定白平衡并记录设置，重启后检查是否保留。

**代码限制：**当前 `src/camera.py` 只请求宽高和小缓冲，没有设置FPS、FOURCC或曝光。驱动可能忽略缓冲设置；“最新帧”不代表无采集延迟。若实测需要固定MJPG/YUYV或FPS，后续应修改相机初始化并重做测量，不能只加无效JSON字段。

**通过：**稳定采图；实际640×480；同位置的物体清晰、亮度稳定。失败顺序：节点/占用 → USB线与Hub供电 → 支持格式 → 光照/距离 → 驱动。C270能否满足动态抓取，留到第13节以实测误差判断。

## 6. 第四步：将KM1接口和工作范围并入最终工位

这一节在最终工位搭建/标定时一次完成，不作为反复抓夹练习。已有原厂控制记录且机构/固件未变时，沿用结果；下列 `arm_cli` 单夹爪命令仅在串口或方向问题需要定位时使用。正常主线完成接口核对、实测参数及路径确认后继续第7节，不要求逐条执行每个点动示例。

1. 用厂家手柄确认机械臂动作正常、舵机ID正确。按项目资料，0–3为关节，4为腕旋转，5为夹爪。
2. KM1使用原配电源，按原厂步骤先供动力电再接USB。此时输送带停止，工作区无人手及障碍物。
3. 查稳定串口路径：

```bash
ls /dev/serial/by-id/
```

将真实路径填入 `config.json` 的 `arm.port`。没有设备时检查数据线、CH340/ESP32接口和固件；不要靠反复发关节指令排查USB。

4. 先打印指令，不开串口：

```bash
python3 -m src.arm_cli --config config.json --servo 5 1000
```

5. 核实5号确为夹爪、该PWM在允许范围且可以随时断电后，将 `arm.commissioned` 改为true，执行一条夹爪指令：

```bash
python3 -m src.arm_cli --config config.json --servo 5 1000 --execute
```

`1000`是工程初值，需要结合夹爪当前姿态确认，不能当作所有KM1的保证值。若不动作，先用厂家USB控制程序核对固件是否转发该ASCII协议；本项目没有自动替换ESP32固件。

6. 再低速验证关节与末端坐标。`--pose X Y Z` 单位mm，先不带 `--execute` 生成指令。坐标由现场实际位置测得，不要把文档里的任意示例点直接执行。
7. 测量并填写：`home`、`grab_z`、`lift_z`、`pick_bounds_mm`、四种缺陷状态的 `bins`、开闭夹爪PWM。最初保留较慢运动，逐点检查从准备位到目标、从目标到盒的完整路径。

**通过：**方向、关节、夹爪、抬起和分类盒路线均经过低速确认；样品能夹住又不被压坏。软件的“可达”不等于轨迹无碰撞。`arm.commissioned=true`只是操作者启用动作的标记，不会自动完成验收。

### 6.1 如何填写现场机械参数

传送带未到先记录USB路径、固件、关节和夹爪方向；以下最终位置在工位固定后测量。用原厂低速点动或手柄核对实物坐标系，再使用本项目逆运动学，不将桌面任意原点当机械臂原点。

| 字段 | 实际操作 | 记录方式 |
|---|---|---|
| `arm.port` | 插拔KM1 USB，比较 `/dev/serial/by-id/` 前后变化 | 填真实稳定路径，关闭占用此串口的其他软件 |
| `home` | 选不遮挡观察区、可到抓取区的准备位 | `[x,y,z]`，毫米；整臂运动路线也要留空间 |
| `pick_bounds_mm` | 在带面可达区测四边，向内留余量 | `[xmin,xmax,ymin,ymax]` |
| `grab_z` | 停带放真实样品，低速下降至可夹高度 | 夹爪不能碰带面；高度按最终样品范围选 |
| `lift_z` | 抬至能越过带边和盒沿的高度 | 检查整段转移，不只检查端点 |
| `open_pwm/close_pwm` | 复用已确认的开闭位置，核对当前样品 | 不按默认值强夹；无夹持力反馈 |
| `bins` | 核对瓶盖不良盒与塑料片不良盒的实际放下位置 | 仅保留4个缺陷键；同种零件的两种缺陷可共用坐标；正常键不加入 |

点动命令的写法是 `python3 -m src.arm_cli --config config.json --pose X Y Z`；把X/Y/Z换成**现场确认的数值**才执行。先不加 `--execute` 检查生成指令，确认整条轨迹后才加。`--pose`不能自动规划避障，夹爪/腕旋转也需按实际任务核对。把动作时间测量并入这些路径确认，供第13节复用。

当暂时还没有最终几何参数，保持动态执行标志关闭，不运行默认 `home`、`bins` 的自动动作。只见到串口发送成功不能认定实物已到位。

## 7. 第五步：做视觉坐标标定与背景分割

输送带停止。相机、分辨率、镜头位置、工作平面高度固定后再标定。

### 7.1 取点和保存配对文件

把标定板或带有编号的标记放到最终观察平面，机械臂底座固定。为每个点同时记录“原图像素u/v”和“机械臂坐标x/y，毫米”；建议纸面先列 `point_id,u,v,x_mm,y_mm,use`。拟合点覆盖观察区边界和内部，独立检查点不能重复使用拟合点。

Windows双击 `tools/point_picker.html`，选择未经缩放裁剪的640×480照片，按编号点击。前6个以上作为拟合点，后3个以上作为检查点；工具显示的所有点击都在 `pixels` 中，需要你按用途分入下述两个数组，不能直接全部粘入拟合数组。每个像素数组与世界坐标数组按同一编号顺序配对。

在板端创建现场文件前确认不存在旧标定，已有文件先备份：

```bash
cp -n data/calibration_pairs.example.json data/calibration_pairs.json
nano data/calibration_pairs.json
```

`nano`内修改，Ctrl+O、回车保存，Ctrl+X退出。文件中 `pixels` 与 `world_mm` 填6个以上数值对，`check_pixels` 与 `check_world_mm` 填3个以上数值对；`plane_note`写相机位置、工作平面高度、日期。数组项例如 `[123.4, 234.5]` 只是格式说明，不能当现场数据。没有nano时用Jupyter文件编辑器保存UTF-8。

1. 在抓取平面布置至少6个拟合点，覆盖边界和内部；另取至少3个独立检查点。用机械臂坐标系测量对应x/y，不能把像素坐标或桌面随意原点当机械臂坐标。
2. 拍一张完整图片，在电脑浏览器打开 `tools/point_picker.html`，点击这些点，获得原图像素位置。
3. 复制 `data/calibration_pairs.example.json` 为 `data/calibration_pairs.json`，填写配对坐标、检查点、`image_size:[640,480]` 和 `plane_note`。
4. 板端运行：

```bash
python3 -m src.calibrate data/calibration_pairs.json
```

默认误差门槛为3mm，输出 `data/calibration.json`。这是拟合/独立点误差，不包括机械回差、物体高度和运动延迟；小物体应按夹取容差收紧要求。

5. 调整根配置的 `vision.background_bgr`、`threshold`、`min_area/max_area`。初始 `[240,240,240]` 是浅色背景占位；你的输送带若是黑色/绿色，必须重新取样设置。颜色顺序是 **B、G、R**。
6. 可在Notebook运行以下离线检查，逐张查看掩码：

Notebook先在一个Python单元格执行 `import os; os.chdir('/home/xilinx/pynq_km1_vision')`，确认 `os.getcwd()` 正确，再运行下面片段。

```python
import json, cv2
from pathlib import Path
from src.vision import mask_cpu, open_mask
c = json.loads(Path('config.json').read_text(encoding='utf-8-sig'))
v = c['vision']
out = Path('metrics/evidence/mask_check')
out.mkdir(parents=True, exist_ok=True)
for p in sorted(Path('data/frames').glob('*.jpg')):
    im = cv2.imread(str(p))
    if im is None:
        raise RuntimeError(str(p))
    mask = open_mask(mask_cpu(im, v['background_bgr'], v['threshold']))
    cv2.imwrite(str(out / (p.stem + '.png')), mask)
```

**通过：**独立点误差满足夹爪需求；待抓物完整保留，背景大部分排除，分开的物体不被合并。塑料环应保留环状材料，不能把孔中心当可抓材料；小环被3×3开运算抹掉时，先改样品/成像尺度和光照。

现场标定凸包应覆盖输送观察区，动态抓取窗口必须另限制在机械臂可达区。两个盒的ROI均在凸包外，格式为 `[x,y,width,height]`，不是两个对角点。可让三类塑料共用同一盒的坐标/ROI，但四个类别键都需存在。

### 7.2 背景和盒区怎么调

1. 在最终光照下分别拍空带、每类单件、相邻两件和空盒照片。
2. 对空带无阴影的小区域取B/G/R代表值，填 `vision.background_bgr`；本项目阈值是像素与背景的三通道绝对差之和。运行上面的掩码片段，把生成PNG取回电脑查看。
3. 调整 `threshold`，使带面大部分为黑、物体为白；调整面积范围时以真实最小/最大样品像素面积为依据。改一项就重看同一组图片，保存参数与图片。
4. 若环被抹掉、木块与背景颜色接近或机械臂阴影成为大块前景，先调整样品尺度、背景和照明；记录无法分开的场景，不用无限放宽面积阈值解决。
5. 用取点工具点瓶盖不良盒可见内部左上角 `(x1,y1)` 和右下角 `(x2,y2)`，计算 `[x1,y1,x2-x1,y2-y1]`。塑料片不良盒同样做。避开盒沿、带面和机械臂长时间遮挡位置。
6. `verification.bin_rois`中仅填cap_notch、cap_deformed、plate_corner_missing、plate_hole_missing四个缺陷键及上述矩形。同一不良盒的两种缺陷可共用矩形，cap_ok和plate_ok不填。空盒/放入一件对照查看白像素增量，据此设 `min_bin_gain_px`；这是视觉证据门槛，不是物理成功真值。
7. 保存 `config.json`、`calibration_pairs.json`、`calibration.json`、原图和掩码到同一批次证据目录。相机位置/分辨率/工作平面变动后复核，不沿用失效标定。

## 8. 第六步：两种零件、六种状态的外观缺陷检测

本节取代旧“四种回收物分类”。主线目标是轻质零件的可见外观缺陷检测与动态剔除。用户引用聊天只能读取所附预览；不把预览中的0.2m/s示例、任意变速或新增PL算法当作本机已有能力。

### 8.0 当前先做：只检测瓶盖变形，跑通一次软件链路

用户选择先收窄到“变形”一种缺陷。本轮采用正常cap_ok和变形cap_deformed两种状态，缺口、塑料片暂不采。六状态最终方案保留作后续扩展，当前按本节推进，不再以首批三类或六类数据齐全为前置条件。

**本轮目标：**在当前 AI PC 迁入已训练的两类模型，完成 ROCm 真实图片推理 → PYNQ发送真实图片并收到正常/变形结果；需要补训时在新环境单独验证训练能力。机械剔除仍需配对位流、编码器、标定和单件执行验收。下文 CUDA 命令及结果为2026-10-07历史演练，当前按AMD部署手册选择模型及环境。

| 用途 | 本轮独立文件/目录 |
|---|---|
| 原图 | data/defect_raw/session01/cap_ok_001、cap_deformed_001等实物目录 |
| 两类训练YAML | data/cap_deformation.yaml |
| 标注后数据 | data/cap_deformation/images与labels下的train/val/test |
| 类别编号 | 0=cap_ok，1=cap_deformed |
| 本轮模型 | data/models/cap_deformation/best.pt |
| PC服务配置 | 原 `config/host_cap_deformation_cuda.json` 仅作历史对照；当前使用 ROCm，按AMD部署手册核对两类模型路径与标签 |
| 后续板端试验配置 | config/cap_deformation_board.json，仅含cap_deformed抓取键，两个执行开关关闭 |

**ID注意：**本轮cap_deformed的ID是1，六状态方案中是2。两套数据集分别保存；扩展时按类别名称重新映射ID，不能直接拼接TXT。独立目录不是旧数据自动重标注。

**2026-10-07首批实物进度：**本机已收到正常9张、变形10张，逐张检查并给完整瓶盖标注矩形框。标注预览见[19张标注检查页](../report/cap_deformation_annotation_review.html)，坐标、原图及划分清单见[数据记录](../report/cap_deformation_demo_dataset_20261007.json)。当前每类仅一个实物，采用同件演练划分：train=11、val=4、test=4，各集合都有两类。此处test只检查软件流程，**不构成独立测试集，分数不能用于泛化或比赛精度结论**。原始照片未改动，训练数据位于Windows英文工作副本；以下采图和标注步骤本批已完成，不必重复。

可在Windows打开检查页逐张查看：绿色框为正常，橙色框为变形，框应包含整个瓶盖，不包含大块投影。标注坐标可在`tools/prepare_cap_deformation_demo.py`中追溯；该脚本遇到已有不同内容会拒绝覆盖。重新收集独立样件后应另行按实物分组，不能沿用本批同件划分当正式验收。

**本批训练及电脑端验证已完成：**20轮初版收敛不足，已保留到`data/models/cap_deformation_20epoch`；随后从预训练权重重新训练100轮，当前服务模型为`data/models/cap_deformation/best.pt`，同目录已有`best.onnx`和`model.json`。训练日志为`report/cap_deformation_training_100.log`。导出依赖已在本项目虚拟环境调整为ONNX 1.19.1，pip check与ONNX结构检查通过。电脑本机HTTP回放确认protocol=2、provider=cuda_pytorch、类别仅两种；19张旧图中18张类别正确且框IoU≥0.5，`1746295663883768303.jpg`的变形件被误判为正常。完整响应见[电脑端回放记录](../report/cap_deformation_local_inference_20261007.json)。这不是独立识别率，模型仍只供联调。用户随后在PYNQ SSH终端完成health请求，返回protocol=2、task=detect、两类标签和cuda_pytorch，模型与运行配置指纹均与电脑端一致：板卡到电脑的HTTP访问已通过。随后首张现场变形件请求已返回cap_deformed，confidence=0.7324，图像640×480；电脑处理总耗时40.50ms，其中模型推理9.86ms，板端RPC往返130.94ms。这是单次测量，不是整机端到端延迟或稳定帧率。板端证据目录为`metrics/evidence/live_1746296834790066835`，本地记录见[现场请求记录](../report/cap_deformation_live_request_20261007.json)。正常件、空场及输入图与框的视觉核对待完成；PL加速和机械动作仍未验收。

**服务、板端health及首张变形件实际请求已通过。用户选择暂跳过正常件和空场检查，现直接推进11.0的FPGA构建；跳过项仍记为未测。以下为以后重启服务时的操作，不必立即重复：**先按9.0在管理员PowerShell增加仅允许板卡访问的防火墙规则；再在普通Windows PowerShell执行下列命令。第二行会生成本次随机口令，第三行显示它供你复制到板端；不要把口令发到聊天中。服务启动后保持该窗口打开。

```powershell
cd C:\fpga\pynq_km1_vision
$env:KM1_TOKEN = [guid]::NewGuid().ToString("N")
$env:KM1_TOKEN
.\.venv-yolo\Scripts\python.exe -m src.detect_server --config config/host_cap_deformation_cuda.json --host 192.168.2.1 --port 8765
```

另开PowerShell通过`ssh xilinx@192.168.2.99`进入板卡。在`xilinx@pynq`提示符后执行第7步的export和curl，口令替换成刚生成的32字符字符串。health通过后按9.1发送相机新帧，本轮只需正常、变形各一例及空场，不必采六种状态。保存每次input.jpg及response.json，确认返回类别与画面一致；没有框不等于正常。9.1中的多行`python3 - <<'PY'`片段只能粘贴到板端SSH终端，不能放进Windows PowerShell或Jupyter Python单元格。

1. 已有正常瓶盖照片继续复用。先补同规格变形瓶盖：顶视轮廓明显变形、主体完整、同面朝上。单件转动方向和移动位置，相机固定。当前先采10张有变化的变形图片；数量仅用于开始标注，不保证训练质量。在板端SSH执行：

```bash
cd /home/xilinx/pynq_km1_vision
python3 -m src.capture --config config.json --count 10 --interval 5 --output-dir data/defect_raw/session01/cap_deformed_001
```

每次saved后改变摆放，并在下一次拍摄前移开手；也可count=1逐张拍。换另一只实物使用002、003；同一件变换布局保留原编号。

2. 在Windows PowerShell下载：

```powershell
cd C:\fpga\pynq_km1_vision
scp -r xilinx@192.168.2.99:/home/xilinx/pynq_km1_vision/data/defect_raw .\data\
```

3. 逐个完整瓶盖画一个框，正常标cap_ok，变形标cap_deformed；不要只框变形部位。目录名不是检测标签。按8.4的矩形标注流程导出，但本轮任务只建上述两个类别，ID检查为0/1。空场图用空TXT。
4. 按物理样件/source_group分组到train/val/test，图像与同名TXT成对。建议正常/变形各至少有3个独立样件以便三个集合各有对应实物，这仍只是流程起步规模。只有每类一个实物时可以继续采集/标注；若使用同件不同照片作流程演练，必须记录为“同件演练划分”，所得分数不能作为独立泛化或比赛成绩。已有数据不足时不虚构测试集通过。
5. 数据及标签准备后，使用本轮YAML短训练。以下命令在Windows英文项目目录执行；新train_yolo.py已同步到该工作副本：

```powershell
.\.venv-yolo\Scripts\python.exe -m src.train_yolo --data data/cap_deformation.yaml --weights yolo11n.pt --device 0 --epochs 20 --batch 8 --imgsz 640 --output-dir data/models/cap_deformation
```

20轮只用于初版软件链路，不保证缺陷识别效果。脚本仍会评估test并导出ONNX；第一次需要PC联网取得预训练权重/必要导出依赖。确认生成本轮目录的best.pt、best.onnx和model.json，类别只有正常/变形。旧六状态模型目录不作为本轮服务入口；重复本轮训练会更新本轮目录，需保留版本时先备份。

6. 按第9.0节配置PC直连防火墙，使用同一随机KM1_TOKEN，启动本轮独立服务：

```powershell
$env:KM1_TOKEN = "替换为本队保存的至少24字符随机口令"
.\.venv-yolo\Scripts\python.exe -m src.detect_server --config config/host_cap_deformation_cuda.json --host 192.168.2.1 --port 8765
```

7. PYNQ SSH设置相同口令，先访问health：

```bash
export KM1_TOKEN='替换为与电脑完全相同的口令'
curl -H "Authorization: Bearer $KM1_TOKEN" http://192.168.2.1:8765/health
```

必须为protocol=2、task=detect、labels含且仅含cap_ok/cap_deformed、provider为cuda_pytorch。再按第9.1节发送真实相机图像，分别放正常/变形样件检查返回框和状态。不要只验证health就算完成；没有检测框时记录为模型尚未识别，不能把空结果当正常。

8. 本轮通信可沿用根配置的camera/inference设置；若固定过旧模型指纹，先备份并更新指纹以匹配本轮模型。此时不运行机械任务。后续进入动态剔除时，以cap_deformation_board.json为参考迁移已标定参数，只对变形类安排候选，不把未经标定初值直接执行。
9. 通过标准：训练完成及产物齐全、两类名称一致、PC真实GPU推理、PYNQ实际图片请求和响应对应正确，至少记录正常与变形各一例。完整六状态、独立准确率和机械剔除仍按后续阶段验收。

### 8.1 做什么零件、几种状态

**完整目标保留2种零件、共6类；当前优先按8.0做正常/变形两状态。** 达到两种零件后先把完整闭环做好，不为增加类别数量扩成很多种零件。以下选择是根据当前C270、KM1抓取能力和现有程序作出的工程建议，不是比赛硬性指标。

| ID | 零件与状态 | 标签 | 如何定义样件 | 当前程序动作策略 |
|---|---|---|---|---|
| 0 | 瓶盖，外观正常 | cap_ok | 同规格、完整外轮廓、无本项目定义的明显形变 | 不安排抓取 |
| 1 | 瓶盖，边缘缺口 | cap_notch | 在俯视可见外轮廓上有一个明显缺失区域，主体仍完整 | 瓶盖不良盒 |
| 2 | 瓶盖，明显变形 | cap_deformed | 外轮廓明显偏离标准圆形；无缺口，以免双重标签 | 瓶盖不良盒 |
| 3 | 单孔塑料片，外观正常 | plate_ok | 同外形规格的平整小片，有一个规定位置的通孔 | 不安排抓取 |
| 4 | 单孔塑料片，缺角 | plate_corner_missing | 外轮廓一角明显缺失，通孔仍正常 | 塑料片不良盒 |
| 5 | 单孔塑料片，缺孔 | plate_hole_missing | 外轮廓完整，但应有的通孔未加工 | 塑料片不良盒 |

“正常”仅指本项目规定的可见外观正常，不代表瓶盖密封或零件功能合格。缺陷种类为4种，加上2种正常状态，共6个YOLO类别，不是6种零件。

**实物准备：**

1. 瓶盖统一规格、材质、颜色和朝向；优先哑光、不透明、没有显著品牌图案的一批。同一面朝上，缺口和形变必须从顶视图真实可见；侧壁上被遮住的小损伤不纳入。
2. 塑料片选择同一外形/厚度的一孔小片，通孔能透出背景。优先按同一设计制作正常、缺角、缺孔样件；缺孔件用未加工孔的实物，不能用明显异色胶带盖洞制造捷径。
3. 每种状态起步准备约10个独立样件，合计约60件，是开发备料建议而非保证精度的最低样本数。材料不足先把瓶盖3种状态各准备若干独立件，跨场次采集；不要把一个缺陷件拍几百张当作几百个独立缺陷。
4. 先核实夹爪实际开口、重量、稳定性和可夹位置，再定尺寸。约25～35mm的轻质样件可作为选型起点，实际必须以夹爪与相机测量为准；塑料片不能薄到夹不住，瓶盖变形不能导致滚动或明显离开标定平面。
5. 样件编号写在底面或托盘记录中，不让编号、胶带、颜色、加工痕迹成为某一类独有的线索。相同颜色/批次尽可能同时含正常与缺陷件。
6. 首轮每件只含一种已定义缺陷；复合缺陷、边界件、反面、重叠件单独放入待复核集合，不硬塞进正常类。若后续纳入复合缺陷，先定义标签规则再增补数据。
7. 保留标准正常件和每类缺陷的参考照片，记录缺口/缺角宽深、形变程度、孔径和成像尺寸。训练标签来自实物定义，不能让模型自己决定何为“不良”。

**其他可做对象与取舍：**

| 候选 | 可见缺陷 | 对本工程的评价 |
|---|---|---|
| 较厚塑料垫圈 | 断口、外形变形、缺孔 | 可替代第二种塑料片，但细环可夹面积小，需先验夹持；替换时重定标签，不直接追加 |
| 3D打印大齿塑料齿轮 | 缺齿、断齿 | 适合作为后续扩展，细齿要求更多像素，数据与轮廓算法工作量增加 |
| 金属螺母/垫片 | 缺角、孔异常 | 反光和重量增加当前调试负担，不列入首轮正式组合 |
| 芯片引脚、细裂纹、表面微划痕 | 弯针、裂纹、微小划伤 | 当前C270大视场与普通照明不能预先保证可辨，不列入当前验收范围 |

### 8.2 从旧配置迁移，保留已完成环境

本节必须在训练/闭环前执行一次。无需重刷SD、重装Python或重建直连。

1. 新训练数据统一使用data/defect_detection，YAML为data/defect_dataset.yaml。旧data/detection、旧data/dataset.yaml及旧四类照片/标签保留，不改名冒充缺陷数据。以前拍的人像只用于验证相机，不放入零件数据集。
2. 先打开config/defect_board.json检查：arm.bins只有4个缺陷键；cap_ok、plate_ok不在其中；两个commissioned均为false，bin_rois为空。文件里的坐标、时序、面积和误差门槛都是待标定初值，不是实测参数。
3. 电脑根config.json若还只是未经标定的发布初值，先备份再复制该样例；若已有现场相机/编码器/几何/时序标定，只在备份后迁移arm.bins、verification.bin_rois和模型指纹字段，保留有效实测数值。不要用样例整文件覆盖已标定工程。
4. 新模型启动前，清空旧expected_model_id/expected_runtime_id；验收新health/真实图片后再按需要固定新指纹。两个执行开关继续false，等物理验收完成再启用。
5. 把已迁移的根config.json上传到板端。先备份板端原配置，再上传；后续控制命令仍使用根config.json，不混用旧四类配置或遗漏板端更新。

**仅适用于根配置未含现场标定、现在要首次采用v6初值的情况，在Windows PowerShell执行：**

```powershell
cd C:\fpga\pynq_km1_vision
$ConfigBackup = "config.before_defects." + (Get-Date -Format "yyyyMMdd_HHmmss") + ".json"
Copy-Item -LiteralPath config.json -Destination $ConfigBackup
Copy-Item -LiteralPath config/defect_board.json -Destination config.json
ssh xilinx@192.168.2.99 "cd /home/xilinx/pynq_km1_vision && cp -n config.json config.before_defects.json"
scp config.json xilinx@192.168.2.99:/home/xilinx/pynq_km1_vision/config.json
```

首次备份文件若已存在，保留原备份并另存当前版本；不要把反复迁移当日常启动步骤。迁移后打开板端config.json，确认直连URL仍为http://192.168.2.1:8765，camera.index仍对应能采图的节点。

**程序现状核对：**src.moving_targets.intercept只对arm.bins中的类别安排抓取，因此可让正常类不抓、4个缺陷类成为候选。未识别、低置信、失跟或机械臂忙碌也可能不抓。当前程序没有完整区分“正常件已正确放行”与“缺陷件漏剔”的自动账本，末端只能标为直行待复核收集区，不能标成已经自动验收的合格出口。静态调试入口也只会抓bins中的缺陷类。

### 8.3 缺陷定义与成像可见性

用户已选择跳过重复相机调试；下面只在正式样件和工作高度确定时做一次任务可见性验收，不重新走板卡启动流程。

1. 相机固定俯视，最终区域使用稳定补光；盖/片单层、无重叠、不滚动。同类别换位置、旋转和朝向时仍应满足缺陷可见。
2. 在实际640×480照片上量缺口、角缺失和孔的像素尺寸。开发起点要求关键结构有多个连续像素，建议最小关键宽度先争取8～12像素，再通过独立样件确定实际可检下限；这不是检测保证。若一项缺陷仅1～2像素，缩小视场或选择更明显样件，不靠训练轮数弥补信息缺失。
3. 3×3形态学开运算会改变细小边缘。保留原始RGB照片给YOLO判别缺陷，PL掩码当前用于几何定位和抓取约束；不能只保存处理后的掩码来证明微缺陷检测。
4. 对形变、缺角和孔定义能人工一致判定的正常/缺陷标准；临界尺寸标为待复核，不把难以判定的样件当硬标签。
5. 单孔片若被夹爪或背景同色区域遮挡，先拒绝这一展示条件。孔的存在是结构判断，不能只靠印一个黑点模拟通孔。

### 8.4 采图、分组与标注

先各状态采少量图把标签链路走通，再扩充。两种零件正式初版可计划600～1000张变化充分的完整画面作为开发起点，覆盖6类及空场；该数量不保证性能，最终由独立数据结果决定。瓶盖先行时仍沿用下面的固定ID0、1、2；没有第二种零件数据时不要宣称六类模型验收通过。

1. 先按实物编号分配训练/验证/测试组，再按采集场次记录。避免同一正常件及其加工后的缺陷版本落入不同集合；将它们作为同一来源组。不同批次、照明和位置都要覆盖，最终另留新批次盲测。
2. 记录sample_id、family、state、defect_measurement、source_group、session、split。起步按组约70%/15%/15%划分，并确认每种状态在三个集合都有独立实物，不能仅随机拆相邻帧。
3. 固定设备和照明后，单件/多件、正常/缺陷混合布局采集。新采图入口支持--output-dir，先按8.4.1上传更新的src/capture.py。每摆好一次、手离开后在板端SSH执行：

```bash
cd /home/xilinx/pynq_km1_vision
python3 -m src.capture --config config.json --count 1 --output-dir data/defect_raw/session01/cap_ok_001
```

4. 用8.4.1的SCP命令取回data/defect_raw，按session和实物编号保存原图。保留清晰但困难的样本，损坏文件剔除；空场单独记录，人物照片不混入。传送带未到的数据标记为预采，安装后必须补采带面及允许运动条件。
5. 在支持YOLO导出的矩形标注工具中建立本节6个英文标签。例如CVAT：新建检测任务、建立标签、上传图片、打开Job、逐物体画Rectangle、选状态、保存、导出YOLO检测格式。[格式说明](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo/)
6. **框住整个零件，标签表示这个零件的状态。** 不只框缺口或孔，否则现有“检测框对应物体掩码”的融合方法无法稳定关联。一个零件只一个状态框；同一物体不同时标cap_ok和cap_notch。多物体逐个标注，正常件也要标。空场对应空TXT；未知/复合缺陷首轮放在单独待复核集，不作为未标注前景混进训练背景。
7. 检查导出ID与8.1表完全一致，再把图像及同名TXT按既定分组放入下列目录：

```text
data/defect_detection/images/train/    data/defect_detection/labels/train/
data/defect_detection/images/val/      data/defect_detection/labels/val/
data/defect_detection/images/test/     data/defect_detection/labels/test/
```

8. 每行TXT为class_id、归一化中心x/y、宽/高，共5列；检查ID0～5、坐标0～1、图片与标签同名、无漏标。data/defect_dataset.yaml里的path指向本机实际目录。[Ultralytics格式](https://docs.ultralytics.com/datasets/detect/)

### 8.4.1 当前直接操作：第一批瓶盖照片

相机基础采图和电脑训练环境已通过，直接开始采集样件。首批推荐正常/缺口/明显变形各10个独立瓶盖，每件至少4种位置或朝向，约120张单件图；这只是第一批开发数据，后续还需混合布局、空场、其他场次及第二种零件。样件暂不足时按已有数量开始并如实记录，不把同一实物重复拍摄算成新增独立样件。

1. 实物按cap_ok_001～010、cap_notch_001～010、cap_deformed_001～010编号，编号写底面或记录表，不放进画面。相同原始瓶盖加工前后的照片必须记录同一个source_group，后续不能跨训练/测试组。
2. 布置单一平面，固定相机，物体完整入画。依次放一个已编号样件；缺口朝向要轮换，形变从顶视图可见。每次改变布局后手退出画面。
3. Windows PowerShell上传更新的采图文件；该操作只更新采图入口，不覆盖板端配置或标定：

```powershell
cd C:\fpga\pynq_km1_vision
scp src/capture.py xilinx@192.168.2.99:/home/xilinx/pynq_km1_vision/src/capture.py
```

4. 板卡SSH先确认新参数存在：

```bash
cd /home/xilinx/pynq_km1_vision
python3 -m src.capture --help
```

帮助中应有--output-dir。若出现unrecognized arguments，说明没有上传到当前运行目录，先核对，不在错误目录继续采集。

5. 放正常样件001，执行以下命令。在每次saved之后改变位置/角度，并在下一次拍摄前移开手；共4张，每次等待约5秒。若操作较慢，可改成--count 1，摆好后逐次执行。

```bash
python3 -m src.capture --config config.json --count 4 --interval 5 --output-dir data/defect_raw/session01/cap_ok_001
```

6. 放缺口样件001和变形样件001时，分别使用各自目录：

```bash
python3 -m src.capture --config config.json --count 4 --interval 5 --output-dir data/defect_raw/session01/cap_notch_001
python3 -m src.capture --config config.json --count 4 --interval 5 --output-dir data/defect_raw/session01/cap_deformed_001
```

必须换好对应样件再执行对应命令，不一次粘贴两行连跑。换另一实物时把001改成002，依次递增；换采集场次时把session01改成session02，实物编号保持原编号。目录名只帮助归档，不会自动生成YOLO框标签，也不是--label旧裁剪分类功能。

7. Windows PowerShell下载本批原图，保留目录结构：

```powershell
cd C:\fpga\pynq_km1_vision
scp -r xilinx@192.168.2.99:/home/xilinx/pynq_km1_vision/data/defect_raw .\data\
explorer .\data\defect_raw
```

8. 检查每件的4张确实是不同布局，剔除手遮挡或拍错样件的图；记录sample_id、source_group、state、session、文件名。相同sample_id在多场次出现时仍按同一组划分。原图先留在defect_raw，标注和按组划分完成后才进入defect_detection的train/val/test目录。
9. 采样只使用相机配置，不要求先切换抓取配置。进入新缺陷闭环前仍须执行8.2迁移。下一步按8.4第5～8项给每个完整零件画框、选择状态；人工目录分类不能代替检测标签。

工具变更：--output-dir默认仍为data/frames；指定目录时自动创建。只有JPEG写入成功才打印saved。已用合成相机验证默认目录、自定义目录、JPEG可解码、写入失败报错及释放相机；更新脚本仍需用户上传到实物板卡后使用。

### 8.5 训练与模型验收

在Windows工程目录执行：

```powershell
.\.venv-yolo\Scripts\python.exe -m src.train_yolo --data data/defect_dataset.yaml --weights yolo11n.pt --device 0 --epochs 100 --batch 8 --imgsz 640
.\.venv-yolo\Scripts\python.exe -m src.evaluate_yolo --config config/host_cuda.json --data data/defect_dataset.yaml --images data/defect_detection/images/test --repeat 5 --output metrics/logs/yolo_defects_cuda_run1.json
```

第一次下载预训练权重需要PC联网。显存不足先减batch；100轮/8批量仅为起点。当前脚本输出data/models/best.pt、best.onnx、model.json，重复训练会覆盖该目录的当前版本，先按版本备份。服务使用best.pt，ONNX用于交付；YOLO在PC运行，板端不安装训练依赖。

训练脚本每次结束会评估test，不能据此反复调参；调参依据为val，最终再用锁模后采集的新批次盲测。检查model.json与health返回的名称/ID为6状态，权重路径和模型指纹对应本次模型，不将旧四类模型改文件名使用。

验收分别记录：每类Precision/Recall、mAP、混淆矩阵、缺陷漏检/错分、正常误报、未判定比例。检测mAP不能代替整件质检或抓取成功率；低置信未检出的缺陷也要计入漏检/待复核，不能从分母删除。起步不预填98%等成绩。

真实图像通信按第9节检查六种状态和空场：框覆盖整个零件，标签与实物一致，RPC/指纹记录正常。正常类不进入抓取候选；四种缺陷类才进入预测计划。随机初始化网络的环境检查结果不能当作这个模型的识别结果。

### 8.6 证明FPGA作用，并完成缺陷剔除验收

当前已有代码职责：PC做YOLO缺陷状态识别；PL做阈值分割/3×3开运算与编码器计数；PS做连通域、内部抓取点、跟踪和拦截计划。增加零件或缺陷类别本身不证明FPGA加速。引用讨论中让PC做抓取决策只是建议，本工程继续用PS调度，以实际代码为准。

1. 第一阶段复用现有PL几何链路，比较相同输入/算法的PS-CPU与PL实现：包含DMA传输、打包、PS后处理，记录P50/P95/P99/max及CPU占用。CPU图像路径对照仍保留同一PL编码器，以隔离图像加速贡献。
2. 同一模型、相机、图像尺寸、样件、带速下比较整机几何更新间隔、预测接触偏差、漏剔率与正常误抓率；逐步增加PC推理/日志负载，看延迟变化。没有实测收益时报告结果，不把流水线设计目标写成加速成绩。
3. 面积、圆度、孔洞数、轮廓模板差异等，可作为后续增强算法的研究方向；当前没有实现PL缺陷特征核，不声称已经在FPGA上完成孔检测或缺口判定。若要把它作为答辩卖点，需要另行实现并与CPU黄金结果、资源/时序及板端性能一起验收。[AMD视觉库背景](https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vitis/vitis-libraries/vitis-vision.html)
4. 先在已标定低速恒速下完成“检测缺陷→持续跟踪→预测抓取→放入不良盒”。变速仅在完成相机时延、编码器尺度、加速度上界和抓取时间标定后，做小幅缓变实验；现程序超过max_accel_mm_s2/误差预算会拒绝或停止，不承诺任意变速成功。0.2m/s不是本机目标；默认1～8mm/s也是待实测初值。
5. 用录像为每个入场样件赋人工真值，至少区分true_state、predicted_state、是否尝试、是否抓住、是否正确不良盒、是否直行、是否待复核。正常未抓不是漏抓；缺陷直行仍是漏剔，无论原因是漏检、计划拒绝还是机械臂忙碌。
6. 缺陷剔除率=正确进入不良盒的真实缺陷件/全部入场真实缺陷件；正常误剔率=被错误剔除的真实正常件/全部入场真实正常件。另记缺陷错盒率、漏检率、未判定率、人工干预和吞吐；分母按实物入场计，不用max-tasks或程序成功数代替。
7. 正式演示准备瓶盖不良盒、塑料片不良盒及末端直行待复核区。先单件验证四种缺陷去向，再混合六状态批次；仅在标定和动作路径验收后打开执行开关。若要宣称自动合格品放行/自动质检统计，尚须补充正常通过、未知、漏剔、轨迹退出等事件与独立计数验证，当前手册不将其列为已实现。

## 9. 第七步：检测服务；静态闭环仅作按需排障

最终路线需要本节的检测服务和网络健康检查。服务联通后，正常主线继续第10–13节，再用第14节的最终 `src.run` 联调。下面独立静态配置和 `src.contest_run` 流程只在需要分离定位视觉/机械问题时使用，不作为必做的另一套系统或固定次数抓夹练习。

PC先生成并保存本队随机口令，终端环境设置同一 `KM1_TOKEN`。不要把真实口令写入提交包。

```powershell
.\.venv-yolo\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
$env:KM1_TOKEN = "替换为刚生成的口令"
.\.venv-yolo\Scripts\python.exe -m src.detect_server --config config/host_cuda.json --host 192.168.2.1 --port 8765
```

PC绑定直连网卡192.168.2.1。先按下面9.0开放仅限板卡的8765端口，再在板端执行：

```bash
export KM1_TOKEN='与PC相同的口令'
curl -H "Authorization: Bearer $KM1_TOKEN" http://192.168.2.1:8765/health
```

根配置 `inference.url`及板端配置样例已统一为 `http://192.168.2.1:8765`；旧板端工程只更新该字段，保留已标定参数。health检查 `protocol=2`、`task=detect`、模型和实际运行后端。不能启动旧 `src.pc_server` 的 `/infer` 分类服务。

### 9.0 允许板卡访问PC检测服务

直连网卡常被Windows归类为“公用/未识别网络”。若规则只选“专用”，可能网页/SSH正常但板端访问8765超时。下面规则覆盖网络类型，但同时限制物理网卡、本机地址、远端板卡地址和TCP端口；不关闭防火墙，不开放整个Wi-Fi网段。

在**管理员PowerShell**执行一次（修改防火墙需要管理员权限；这是后续人工执行的命令，本次文档更新未自动运行）：

```powershell
$DirectAdapter = "以太网"
Get-NetAdapter -Name $DirectAdapter
if (-not (Get-NetFirewallRule -Name "PYNQ-Direct-YOLO-8765" -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -Name "PYNQ-Direct-YOLO-8765" -DisplayName "PYNQ direct YOLO 8765" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8765 -LocalAddress 192.168.2.1 -RemoteAddress 192.168.2.99 -InterfaceAlias $DirectAdapter -Profile Any
}
```

当前 AI PC 的网卡名先用 Get-NetAdapter 确认，再修改 DirectAdapter；旧 Y9000P 的“以太网”名称不作为新机默认值。若已有同名规则，核对其接口/地址/端口，不能只因名称存在就认定配置正确。

服务终端开始监听后，在电脑另开普通PowerShell执行：

```powershell
Get-NetTCPConnection -LocalAddress 192.168.2.1 -LocalPort 8765 -State Listen
```

无监听：检查服务终端中的模型路径、依赖、GPU后端或绑定地址错误。出现WinError 10049通常表示电脑还没有192.168.2.1；出现10048检查端口占用。监听存在但板端health超时：核对本节规则、VPN/代理和直连IP。401表示已到达服务但口令不一致。无需开启Windows文件共享、远程桌面或Windows SSH服务器；SCP/SSH由电脑主动连接板卡。

板卡访问http://192.168.2.1:8765，电脑浏览器访问http://192.168.2.99；两个方向不要填反。软件中默认127.0.0.1只供本机测试，运行服务时必须保留上文的 `--host 192.168.2.1`。

依据：[Microsoft New-NetFirewallRule参数说明](https://learn.microsoft.com/en-us/powershell/module/netsecurity/new-netfirewallrule)。

### 9.1 验证真正的整帧检测请求

Windows服务绑定192.168.2.1，终端保持打开；防火墙规则见9.0。PYNQ口令每次新开终端需重设。health返回401查口令，连接拒绝查服务/绑定IP，超时查路由/防火墙。

先保持Wi-Fi完成依赖和模型准备；真实请求成功后，关闭电脑Wi-Fi、保留网线，再重复health和本节请求，确认不触发在线下载。当前 HX 370 重复此项，不以Y9000P通过代替新机验收。

PYNQ执行以下检查；不加载位流、不调用机械臂、不需要传送带。先关闭其他相机程序：

```bash
python3 - <<'PY'
import json, time, cv2
from pathlib import Path
from src.detect_remote import DetectionClient
c = json.loads(Path('config.json').read_text(encoding='utf-8-sig'))
cap = cv2.VideoCapture(c['camera']['index'])
try:
    cap.set(3, c['camera']['width']); cap.set(4, c['camera']['height'])
    assert cap.isOpened(), 'Camera unavailable'
    for _ in range(10):
        ok, frame = cap.read()
    assert ok, 'Camera read failed'
    stamp = time.monotonic()
    result = DetectionClient(c['inference']).predict(frame, 1, stamp)
    out = Path('metrics/evidence/detect_' + str(time.time_ns()))
    out.mkdir(parents=True)
    cv2.imwrite(str(out/'input.jpg'), frame)
    (out/'response.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
finally:
    cap.release()
PY
```

对照输入图与返回框：image_size为 `[640,480]`、类别正确、框覆盖物体、RPC有记录。分别检查六种状态和空场。此处frame_id=1是单次独立请求，不表示动态跟踪已验收。迁移HX 370后重做，provider应为rocm_pytorch。

**仅在选择静态排障时，使用CPU静态入口，并停住输送带。** 当前 `config/static_board.json` 仍是block/washer旧类别，别直接使用。完成前面根配置的现场修改后，在板端从根配置复制一份独立调试配置：

```bash
python3 - <<'PY'
import json
from pathlib import Path
c = json.loads(Path('config.json').read_text(encoding='utf-8-sig'))
c['vision']['backend'] = 'cpu'
c['arm']['commissioned'] = False
c['dynamic']['commissioned'] = False
p = Path('config/commission_static.json')
if p.exists():
    raise SystemExit('File exists: inspect it before replacing measured values')
p.write_text(json.dumps(c, indent=2), encoding='utf-8')
print(p)
PY
python3 -m src.contest_run --config config/commission_static.json --frames 300
```

这一入口无需项目位流/编码器，但仍需真实相机、标定、模型服务与正确类别。把单个目标放在安全抓取区，检查 `board/last_plan.json`、`board/latest.jpg`、`board/mask.png`。

确认类别、抓取点、全部路径、四种缺陷状态的放置坐标和盒ROI都正确后，仅将该调试配置的 `arm.commissioned` 改为true，执行 **1件**：

```bash
python3 -m src.contest_run --config config/commission_static.json --execute --max-tasks 1 --frames 300
```

静态排障以定位并解决具体问题为结束条件，不另设每类固定次数的抓夹练习。记录实际抓取/入盒和失败原因；出现明显坐标偏差或碰撞趋势时回查标定和姿态。程序的图像证据只辅助判定，成功以人眼和录像为准。

以后静态配置不会自动跟随根配置更新；修正标定、bins、ROI或网络设置时同步有效值。它只是调试入口，最终仍回到动态 `src.run`。

## 10. 第八步：SEN0230 接线与机械安装

先核对编码器标签/型号/输出方式；不是SEN0230时，以下电源、线序、每圈计数不直接照搬，按真实资料确认。传送带未到可先做电气与手转计数，安装后再标定毫米比例。

官方资料确认：SEN0230为 **NPN开集电极**，5V供电，A/B每相400周期/转，本项目四倍频为 **1600计数/编码器轴转**。当前官方线序为红VCC、黑GND、白A、绿B；实际产品以机身标签核对，不能仅凭线色接电。[DFRobot官方Wiki](https://wiki.dfrobot.com/sen0230/)

开集电极表示输出能拉低，输出高电平由外接上拉电源决定。因而采用“**编码器供电5V，A/B各通过电阻上拉到3.3V**”。这是依据输出结构设计的接法，仍要测量实物确认，无需因为供电5V就一律给A/B上拉5V。

建议固定使用空闲的 **PMODB物理3、4脚**：

| SEN0230 / 电阻 | 接哪里 | 备注 |
|---|---|---|
| VCC（当前官方红线） | 稳压5V电源正端 | 不是PMOD的3.3V脚 |
| GND（当前官方黑线） | 5V电源地 + PYNQ PMODB物理5脚GND | 采用共地连接，电机大电流不经信号地线回流 |
| A（当前官方白线） | PMODB物理3脚，FPGA封装脚T11 | 顶层端口encoder_a |
| B（当前官方绿线） | PMODB物理4脚，FPGA封装脚T10 | 顶层端口encoder_b |
| 10kΩ电阻1 | A与PMODB物理6脚3.3V之间 | A独立上拉 |
| 10kΩ电阻2 | B与PMODB物理6脚3.3V之间 | B独立上拉 |

引脚依据：[Xilinx官方Z2约束](https://raw.githubusercontent.com/Xilinx/PYNQ/master/boards/Pynq-Z2/base/vivado/constraints/base.xdc)与[TUL Z2参考手册第16节表14（镜像）](https://pynq.tue.nl/doc/pynqz2_user_manual_v1_0.pdf)。物理3/4脚不是软件GPIO索引3/4；它们对应索引2/3。按板上1脚标记和手册图确认朝向，不按“左边第几个”猜。

```text
稳压5V ─────────── SEN0230 VCC
共同GND ────────── SEN0230 GND / PYNQ GND
PYNQ 3.3V ─10kΩ─┬─ SEN0230 A
                 └─ PMODB pin 3 / T11
PYNQ 3.3V ─10kΩ─┬─ SEN0230 B
                 └─ PMODB pin 4 / T10
```

接板验收顺序：

1. 断电布线，先不把A/B插入FPGA。核对5V电源、共地和两个上拉电阻，移除任何已有5V上拉。
2. 让PYNQ正常供电，编码器供电后手转轴，用示波器测A/B各为约0～3.3V且相位相差四分之一周期。万用表可在静止不同轴角检查高低电平，但不能代替脉冲时序验收。
3. 再断电接A/B；加载本工程Overlay，确认两端口是输入。不要同时运行会把这些脚配置成输出的其他Overlay。避免PYNQ断电时从外设向IO送电。
4. 10kΩ是短线低速起点。若上升沿太慢或电机干扰导致错误跃迁，依据实测波形调整上拉/线缆/缓冲或隔离；不要先放宽错误检测来掩盖脉冲问题。
5. 编码器最好接实际输送辊/测量轮，保证无打滑。若接电机轴，必须纳入减速比，且它不能直接测到皮带打滑。

本次已附 [encoder_sen0230_pmodb.xdc](../build/encoder_sen0230_pmodb.xdc)。它只定义已核对的引脚与LVCMOS33，**没有假装完成实机电气验收、CDC/时序审查或综合**。

## 11. 第九步：构建真实 FPGA 位流并验收图像核

### 11.0 Y9000P构建项目位流（历史记录）

当前 AI PC 的构建须先安装并预检 Vivado/Vitis 2026.1，使用仓库根目录 [Windows构建入口](../../docs/team/BUILD_WINDOWS.md)。以下为旧机2026-10-07记录，工具路径和“本机已完成”不适用于新机。

2026-10-07用户选择暂不补正常件和空场测试，直接推进FPGA构建。仅有首张变形件现场请求通过；跳过项记为未测，不算验收通过。机械执行开关保持关闭。

本机已完成准备，无需重复安装软件或手动找板卡包：

- Vivado与Vitis均为2026.1，构建脚本CheckOnly通过。
- PYNQ-Z2板卡文件已放到`C:\fpga\board_files\pynq-z2\A.0`，XML可解析。
- Vivado实际查询返回器件`xc7z020clg400-1`、板型`tul.com.tw:pynq-z2:part0:1.0`；预检日志为`report/board_preflight_20261007.log`。
- 板卡包来自历史TUL CDN链接，下载来源及SHA256记录见[板卡文件记录](../report/pynq_z2_board_files_20261007.json)。SHA256是本次下载记录，不是厂家公布的校验值。
- 现已完成segment_open构建并生成`board/vision.bit`和`board/vision.hwh`，详见下方本轮结果。板端实际加载和CPU/PL一致性仍待测试。

在Windows普通PowerShell中执行以下命令，不能在`xilinx@pynq`的SSH终端执行。电脑接电源并保持不休眠。可在原检测服务窗口按Ctrl+C暂时结束服务，后续需要时再启动。此阶段不需要连接编码器或传送带，也不需要取出SD卡。

```powershell
cd C:\fpga\pynq_km1_vision
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_windows.ps1 -BoardRepo C:\fpga\board_files -Kernel segment_open -EncoderXdc .\build\encoder_sen0230_pmodb.xdc 2>&1 | Tee-Object -FilePath .\report\fpga_build_first.log
```

ExecutionPolicy Bypass只作用于这次启动的PowerShell进程，避免此前脚本执行策略报错；不修改系统的永久执行策略。命令依次运行HLS C仿真、综合、C/RTL协同仿真、IP导出和Vivado实现。脚本会更新`build/hls_segment_open`（选择segment时为hls_segment）并重建`build/vivado_project`，后续重跑前先保存需要保留的构建产物。

等待命令结束，不同时重复启动同一构建。预期末尾提示`Built segment_open with AMD 2026.1`，并生成配对的`board/vision.bit`和`board/vision.hwh`。出现ERROR或异常退出时保留日志，先处理第一处失败；不要拿残留文件继续上传。成功后按11.1检查仿真、时序和资源报告，再传到板卡运行CPU/PL一致性测试。该阶段的图像核负责分割和形态学开运算，正常/变形YOLO仍在电脑GPU运行。

**2026-10-07构建修复：**首轮C仿真漏编译segment_open.cpp而链接失败；现已切换为HLS配置文件入口，原构建命令继续适用。图像核C仿真、C综合、Verilog协同仿真及IP导出已通过。另修复DMA写回通路未连接DDR的问题：两方向共用同一个内存互连，并在构建前检查双方DDR地址映射。HLS日志为`report/fpga_build_config_20261007.log`，修复后的Vivado日志为`report/vivado_fixed_console_20261007.log`；这些软件检查不等同于上板验收。详细状态见[构建修复记录](../report/fpga_build_repair_20261007.json)。

**本轮构建结果已完成：**C仿真0错误，Verilog协同仿真Pass，IP打包通过，Vivado write_bitstream Complete；配对位流已放入Windows工作副本的board目录。100MHz下WNS=1.423ns、TNS=0，WHS=0.016ns、THS=0；无未约束内部端点，但两个编码器异步输入的约束/CDC审查仍待后续完成，不能称为整机时序签核。LUT=4277、FF=6243、BRAM=4块、DSP=3。保留厂家DDR预设及工具报告中的警告。后续构建已改为串行子任务，避免本机并行时出现过的Tcl-app启动失败；未重新测量串行全量从零构建时间。最终续跑/导出日志为`report/vivado_resume_console_20261007.log`和`report/vivado_finalize_console_20261007.log`。

**2026-10-08上板结果：**本节上传与图像验算已完成，无需重复构建。四组共475次CPU/PL逐像素一致，包含12类边界/阈值/噪点测试和相机新采3帧；PL含打包与DMA平均约105～108ms，CPU约158～161ms，加速约1.5倍。详见[上板图文报告](../report/pl_board_20261008/index.html)。原背景参数使照片几乎全前景，另用独立测试参数验证有效分割，未覆盖现场config.json；背景/ROI/坐标标定、编码器与机械闭环仍待完成。以下命令保留供复现。

**上传图像核与测试图片：**Windows执行：

```powershell
cd C:\fpga\pynq_km1_vision
scp .\board\vision.bit .\board\vision.hwh xilinx@192.168.2.99:/home/xilinx/pynq_km1_vision/board/
scp -r .\data\benchmark_frames xilinx@192.168.2.99:/home/xilinx/pynq_km1_vision/data/
```

本机已准备10张640×480瓶盖图片到benchmark_frames，仅用于CPU/PL逐像素与耗时对比，不用于补做已暂跳过的分类测试。关闭其他使用Overlay的Notebook或程序，编码器暂不接，机械臂不运行。然后在板端SSH执行：

```bash
cd /home/xilinx/pynq_km1_vision
sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -m src.benchmark --config config.json --pl --images data/benchmark_frames --repeat 20
```

显式使用已确认的板端PYNQ解释器，以root权限加载Overlay并访问DMA。2026-10-08实测，sudo后的环境还需显式传入`XILINX_XRT=/usr`（与板端`/etc/profile.d/xrt_setup.sh`一致）；仅指定解释器会报`No Devices Found`。通过应返回exact_match=true；耗时和speedup按实测报告，不能预设FPGA一定更快。测试不依赖电脑YOLO服务，不执行机械动作。可用`--output report/批次名.json --evidence-dir report/批次目录`保留独立报告及首轮原图、CPU掩膜、PL掩膜和差异图；保存图片不计入算法耗时。

### 11.1 准备板卡预设与可追溯的构建

TUL站点本次访问超时；遇到下载不可达先从下方PYNQ官方说明核对厂商入口，或向板卡提供方索取原版Z2 board files，保留来源和版本。不用名称相似的其他板型文件代替。

1. 从TUL PYNQ-Z2官方产品支持资料取得board files，解压到例如 `C:\fpga\board_files`；它是含 `board.xml` 等文件的目录，不是SD镜像，也不是本项目位流。下载入口见[TUL官方PYNQ-Z2产品页](https://www.tulembedded.com/FPGA/ProductsPYNQ-Z2.html)的Downloads；也可从[PYNQ官方board settings说明](https://github.com/Xilinx/PYNQ/blob/master/docs/source/overlay_design_methodology/board_settings.rst)进入。
2. PowerShell执行 `Get-ChildItem C:\fpga\board_files -Recurse -Filter board.xml`，确认存在Z2板卡定义。Vivado中在Board Repository设置加入该目录，在新建工程板卡列表查PYNQ-Z2；器件应为 `xc7z020clg400-1`。找不到时先解决目录层级/board files/器件组件，不替换成Z1。
3. 切到英文工程目录，确保本次需要保留的旧构建报告和位流已另存。构建脚本会重建工程，不能把旧结果和本次结果混在一起作为成功证据。
4. 运行下文CheckOnly后开始构建；Windows保持交流供电和不睡眠。启动前可用 `Start-Transcript -Path ('build_' + (Get-Date -Format yyyyMMdd_HHmmss) + '.txt')` 记录控制台，结束后 `Stop-Transcript`。第一个失败阶段解决后再继续，不把末尾报错当根因。

这一步和前面的数据采集/训练可在队内分工推进。仅有Python测试和RTL仿真不能代替上板。

1. 本机已安装 **Vivado 2026.1 + Vitis HLS 2026.1**，位于 `D:\work\AMDDesignTools\2026.1`，无需重复安装。后续检查Zynq-7000器件支持。其他电脑的具体Windows版本、工具组件和许可按[AMD 2026.1官方支持表](https://docs.amd.com/r/en-US/ug973-vivado-release-notes-install-license/Supported-Operating-Systems)核对。不要只安装Lab Edition就期望能综合。
2. 安装TUL PYNQ-Z2 board files，确认Vivado能够选择Z2与 `xc7z020clg400-1`。board files、SD镜像和项目bit/hwh是三种不同文件。
3. 按 [2026.1 环境说明](vivado_2026_1.md)，在纯英文短路径工程中启动PowerShell，先运行 `./tools/build_windows.ps1 -CheckOnly`；脚本可自动定位本机工具，或用 `-ToolRoot` 指定版本目录。确认工具检查通过后，再运行：

```powershell
.\tools\build_windows.ps1 -BoardRepo C:\fpga\board_files -Kernel segment_open -EncoderXdc .\build\encoder_sen0230_pmodb.xdc
```

`BoardRepo`替换成实际board files根目录。现有脚本始终加入编码器IP，不能省略EncoderXdc。若Tcl/BD报错，保存第一处错误和工具版本，按其解决；本机本轮segment_open路径已完成构建，换工具环境仍需核对适配。

4. 检查HLS C仿真、综合、C/RTL协同仿真输出；再检查Vivado实现、DRC、时钟/未约束路径、资源、WNS/TNS及保持时序。当前脚本只对部分实现状态/setup slack做自动门槛，成功退出不等于所有时序审查已完成；100MHz和II=1是设计目标，报告实际结果。
5. 应产生 `board/vision.bit`、`board/vision.hwh`、`report/timing.rpt`、`report/utilization.rpt`。HLS报告在 `build/hls_segment_open/hls/` 内。保存工具日志、XDC和同次产物摘要。

```powershell
scp board/vision.bit board/vision.hwh xilinx@192.168.2.99:/home/xilinx/pynq_km1_vision/board/
```

6. 所有其他使用Overlay的程序退出后，在PYNQ用同一批少量640×480图像测试：

```bash
python3 -m src.benchmark --config config.json --pl --images data/benchmark_frames --repeat 20
```

运行前创建 `data/benchmark_frames`，只复制10～20张640×480代表性图片进去，避免脚本一次读入大数据集耗尽板卡512MB内存。CPU和PL应逐像素完全相等，结果 `exact_match=true`；出现不一致先查BGR打包、零边界、核类型和寄存器。

报告默认写 `report/benchmark.json`，下一次会覆盖，运行后立即按批次另存。完整PL耗时包含打包、DMA、等待和结果拷贝；若比CPU慢，真实报告并查瓶颈，不删除DMA开销制造加速。

**通过：**生成配对位流，板上加载成功，图像核与CPU黄金参考一致，资源和时序报告可追溯。DMA超时按bit/hwh配对、kernel类型、TLAST、传输长度、时钟复位排查；不要无限重发故障DMA。

### 11.2 构建后具体检查哪些文件

Windows运行 `Get-Item board\vision.bit,board\vision.hwh,report\timing.rpt,report\utilization.rpt`，检查存在、非空、时间属于本次构建；再 `Get-FileHash -Algorithm SHA256 board\vision.bit,board\vision.hwh` 保存摘要。传输后PYNQ运行 `sha256sum board/vision.bit board/vision.hwh` 对照。

在报告中记录LUT/FF/BRAM/DSP、目标时钟、WNS/TNS、保持时序、未约束路径和DRC。还要核对HLS协同仿真结果；文件存在不能替代报告审查。板端只能由一个程序管理Overlay，基础LED Notebook、benchmark、编码器诊断和最终分拣不要同时运行。

| 报错 | 先检查 |
|---|---|
| Board preset not found | BoardRepo根目录和Z2预设内容 |
| 器件不可用 | Zynq-7000组件和正确芯片 |
| 非ASCII路径 | 工程、BoardRepo、XDC均改为英文无空格路径 |
| HLS/IP/BD错误 | 保留首个失败信息与工具日志，修复后重构建 |
| Overlay找不到IP | 同批bit/hwh、顶层接口和核版本 |
| DMA超时 | 保存日志，退出其他Overlay程序；必要时受控重启后重新加载配对文件 |

## 12. 第十步：编码器计数、毫米比例与带速

先停电机、手动转轴。板端可在SSH执行下面的只读采样示例；它会加载项目位流，运行时不要同时开其他Overlay任务：

```bash
python3 - <<'PY'
import json, time
from pathlib import Path
from pynq import Overlay
from src.conveyor import Encoder
c = json.loads(Path('config.json').read_text(encoding='utf-8-sig'))
e = dict(c['encoder'])
e['mm_per_count'] = 1.0
e['direction'] = 1
enc = Encoder(Overlay(c['bitfile']), e)
print('Diagnostic scale only: position_mm below is interpreted as counts')
try:
    while True:
        s = enc.sample()
        print({'time':s['time'], 'count':s['position_mm'],
               'ticks':s['ticks'], 'errors':s['errors']}, flush=True)
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
PY
```

这个片段只在内存里把尺度设为1方便读计数，不修改根配置。

1. 在轴上做标记，准确转一圈应变化约1600计数，反转符号相反；端点人工对准误差单独记录。回到原角度应回到接近起始计数。
2. 若读数约400或800，核对实际解码方式；本项目RTL设计为四倍频。若无变化，检查5V供电、上拉、引脚和Overlay；若非法跃迁增加，检查波形和干扰。
3. 带面贴标记，测量走过真实距离L，同时记录计数差ΔN：`encoder.mm_per_count = L / abs(ΔN)`。重复至少3次，并测一段不同长度作独立检查。
4. 无减速且编码器与滚筒同轴时，理论初值为 `πD/1600` mm/count。接电机轴则纳入“编码器转数/滚筒转数”传动比。不要直接把当前0.05占位值当实测，也不要再额外乘一次4。
5. 使正常输送方向对应正位移，通过 `encoder.direction=1/-1`确定；`dynamic.axis_xy`填机械臂x/y坐标下的输送方向单位向量。
6. 低速开电机，测稳定带速与波动：`v = ΔN × mm_per_count × direction / Δt`。测试启动、停止、反转，检查错误计数和速度符号。错误计数的绝对值及是否增长都应记录。

**通过：**正反向和一圈计数正确；不同距离测试可复现；电机运行不引入异常计数；毫米比例、稳定最低带速与波动有记录。`src.run`会拒绝停带、反转和超出允许带速，不能用它代替最初的手动计数测试。

### 12.1 安装后如何测毫米比例和方向

1. 机械臂动力关闭，在带面贴一条标记；沿带旁固定标尺，选两个相距L毫米的基准线。编码器已按第10节装牢。
2. 保持上面计数程序连续运行；标记到起点时记录N0，到终点时记录N1。停止电机后读取停稳值时，实际距离也按停稳位置测，不把惯性滑行忽略掉。
3. 用 `L/abs(N1-N0)` 得本次比例，做至少3次；如果比例随距离变化明显，先查打滑、读数时刻和传动，不直接取平均掩盖问题。
4. 将比例填 `encoder.mm_per_count`。正常输送时计数若为负，填direction=-1，否则为1。
5. 由标定得到带面起终点机械臂坐标，令dx=x1-x0、dy=y1-y0，填 `dynamic.axis_xy=[dx/sqrt(dx²+dy²),dy/sqrt(dx²+dy²)]`。这是机械臂平面方向，不是图像向下方向；保留足够小数使向量长度接近1。
6. 换一段未用于拟合的长度复核，记录预测位移、标尺位移和差值。

### 12.2 连续记录实际速度

再次运行上面的采样片段时，删除临时 `mm_per_count=1.0` 和 `direction=1` 两行，使用已测配置；print改为输出整个 `s`，就能看到position_mm和velocity_mm_s。预热满速度窗口前velocity为None正常。每个工作速度稳定运行一段，记录最小/最大速度、波动、errors及异常；启动/停止阶段单独记录。

采样保存时可把同一片段写到自己记录目录的诊断脚本，再把控制台结果保存为日志。其作用只是读取传感器，不控制电机。毫米比例与速度合格后才进入动态预测。

## 13. 第十一步：测 C270 延迟、机械动作时间与可抓窗口

这是最终动态拦截的必要参数准备。当前根配置中的1～8mm/s、80ms采集延迟、0.5mm位置误差等都是占位值，不能通过打开两个commissioned开关把它们变成真实参数。

**测相机与编码器时间偏差：**在输送带放一个高对比标记，用已标定图像测其位置，同时记录每帧主机读完时刻、编码器快照及带速。选择相同机械位置作为两种测量的零点，分多个恒速及两个方向比较视觉位置与编码器位置，估计有效采集延迟并区分固定标定偏移；多次试验看误差分布，必要时用同一外部时基的LED/示波器或高速录像辅助。不能把PC显示画面的延迟直接当相机延迟。

填写 `capture_delay_s` 和保守的 `capture_uncertainty_s`，并说明测量方法。这只能修正/估计UVC采集时间，不能称为曝光硬件同步；现有程序没有一键完成该标定的工具。

**测动作：**把时间测量并入最终工位的行程/路径确认：用相同样品记录闭爪指令到真正接触的时间及波动，设置 `grip_contact_s`、`grip_ms`、`actuation_uncertainty_s`，已有同条件有效记录可复用。用录像/仪器确认动作时间，不把设置的PWM时长直接等同于接触时刻；记录完整抓取到返回准备位的周期 `T_cycle`。测量需要足够样本估计波动，但不另设重复开合夹爪的教学流程。

当前程序估计接触提前量：

`H = 2 × (arm.move_ms/1000 + arm.settle_s) + dynamic.grip_contact_s`

例如默认1500ms、0.3s、0.3s算得H=3.9s；若**实测**带速5mm/s，H期间走19.5mm。这只是公式示例，不是你的实测性能。观察区到接触点的可用距离还要留出识别、连续跟踪、物体尺寸与误差余量。

配置顺序：

| 参数 | 按什么定 |
|---|---|
| `dynamic.pick_window_mm` | 机械臂实际可达区内较窄的抓取窗口 |
| `min_speed_mm_s/max_speed_mm_s` | 输送带经过验证的稳定正向速度范围 |
| `max_accel_mm_s2` | 在目标工作速度下测得的加速度上界与余量 |
| `position_uncertainty_mm` | 标定/成像/定位误差的保守估计 |
| `max_error_mm` | 样品与夹爪真正允许的接触偏差 |
| `actuation_uncertainty_s` | 多次实测接触时间波动 |
| `schedule_slack_s` | 可接受且经过测试的软件调度迟到量 |
| `association_mm`、类别有效期 | 实测定位噪声、目标间距和检测往返时间 |

误差需计入至少：标定误差、带速乘相机/动作时间不确定度、未来运动加速度。源码已有保守误差预算，禁止单纯放宽 `max_error_mm` 来消除拒抓。

供料起步采用“一件完成后再放下一件”。稳定后，若希望大部分来料都处理，间距应满足约 `d > v × T_cycle` 并增加物体尺寸和余量。没有实测T_cycle前不承诺连续高吞吐。

**通过：**低速时图像不明显拖影；预测接触点误差小于实际夹取余量；观察距离足以提前规划。如果C270在实际最低带速仍有不可接受的模糊/延迟，先改善光照、降低带速或加长观察区，仍不满足才评估相机升级。

### 13.1 采集延迟的实际测量步骤

1. 机械臂动力关闭，单个高对比标记随带运动。相机、光照、编码器尺度和方向均与最终配置相同。
2. 使用本项目 `LatestCamera` 的读帧完成时间与 `Encoder.sample()` 的主机中点时间作为记录时基；不能拿PC显示窗口时刻与PL ticks直接相减。
3. 每帧保存图、frame_id、读帧时间、编码器采样时间、位移和速度。用第7节取点工具标记同一个目标点，经标定换成机械臂平面位置，再投影到axis_xy方向，得到视觉沿带位置q。
4. 将编码器位置外推到读帧时刻：`s_read = position_mm + velocity_mm_s × (frame_time - encoder_time)`。恒速且不打滑时有 `q - s_read ≈ offset - velocity × delay`。
5. 只用一个速度无法区分固定坐标偏移与时延。保持零点和同一标记安装不变，在至少两个不同稳定速度采样，条件允许可包含反向；拟合差值对速度的直线，截距是固定偏移，负斜率是有效延迟。每次重启采样会重置相对位移，必须重新对齐同一机械零点，不能直接混合。
6. 排除启动/加速段，检查剩余残差、运动模糊及读数抖动，为capture_uncertainty_s留保守余量。用未参与拟合的速度复核补偿后位置。位移噪声不要全部当成时间噪声。
7. 保存原图、配对表、拟合过程和复核误差，再写入capture_delay_s/uncertainty_s。在正常推理/日志负载下复核，记录纯采样与最终负载的差异。

下面是采集原始配对材料的PYNQ片段，**不自动求延迟**。机械臂动力关闭，退出其他Overlay和相机程序，在已验收的稳定带速下采集。每次程序启动相对编码器位移归零，按上面方法对齐机械零点。保存图片有额外负载，最后仍须在正常系统负载下复核。

```bash
python3 - <<'PY'
import csv, json, time, cv2
from pathlib import Path
from pynq import Overlay
from src.camera import LatestCamera
from src.conveyor import Encoder
c = json.loads(Path('config.json').read_text(encoding='utf-8-sig'))
out = Path('metrics/evidence') / ('delay_' + str(time.time_ns()))
out.mkdir(parents=True)
enc = Encoder(Overlay(c['bitfile']), c['encoder'])
cam = LatestCamera(c['camera'])
last = 0
try:
    with (out/'pairs.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image','frame_id','frame_time','encoder_time','position_mm',
                    'velocity_mm_s','snapshot_uncertainty_s','errors'])
        for _ in range(300):
            fid, stamp, frame = cam.read(after=last)
            last = fid
            s = enc.sample()
            name = str(fid) + '.jpg'
            if not cv2.imwrite(str(out/name), frame):
                raise IOError('Image write failed')
            w.writerow([name,fid,stamp,s['time'],s['position_mm'],
                        s['velocity_mm_s'],s['uncertainty_s'],s['errors']])
            time.sleep(0.05)
finally:
    cam.close()
print(out)
PY
```

将CSV和对应照片取回电脑，按13.1步骤加入目标像素/世界位置、计算s_read及差值并拟合。若不能分清时延与几何偏移，先改测量方法，必要时用同一外部时基的LED/高速录像辅助；不可用默认80ms代替实测。

### 13.2 动作时间和允许偏差怎么定

1. 把相机或高速录像固定到能看到夹爪与样品接触的位置，记录录像实际帧率。运动发送时刻需有可对齐事件/外部时基，普通不同设备屏幕时钟不能直接相减。
2. 在第6节最终路径确认时记录：动作指令发出、开始移动、到达上方、下降完成、首次接触、夹紧、入盒、返回准备位。帧差/帧率换算时间，并保留时间分辨率。
3. grip_contact_s是闭爪指令到首次接触的时间，须不大于grip_ms/1000；grip_ms仍需覆盖足够的夹持动作。actuation_uncertainty_s来自多次相同条件的波动和测量误差。
4. 在停带、已验证路线下观察不同微小偏移时能否稳定夹持，结合样品尺寸、夹爪接触宽度和最坏定位误差确定max_error_mm。不能仅为让软件放行而加大它。
5. 测T_cycle，从本件开始处理到机械臂回到可接下一件；首次供料每件完成再放下一件。以后用v×T_cycle再加物体尺寸/余量定最小间距。
6. 把观察区到抓取窗口距离与本节H比较，必须容纳识别、连续跟踪、运动和误差余量。若不够，调整布局/降低稳定带速，再复测；不只改软件窗口数字。

把每个参数记录为“值、单位、方法、样本数、波动、证据路径、日期”。在动作/相机/速度实测未完成前，dynamic.commissioned保持false。

## 14. 第十二步：动态只规划，再执行一件

### 14.1 每次开批次前的顺序

1. 确认工位固定、接线可靠、盒中有可见余量，准备末端接料。操作者能切断机械臂与输送带动力。
2. 接好正式推理电脑与PYNQ的直连网线并上电，确认固定IP与串口路径；关闭占用相机/串口/Overlay的其他任务。
3. 启动电脑detect_server并保持终端打开；板端新终端设置KM1_TOKEN，health及真实图请求检查通过。
4. 检查本次配置、模型、配对位流和标定摘要。确认四种缺陷bins和bin_rois齐全、背景正确、抓取窗口在机械范围内。
5. 先空带运行到已验收稳定速度，用传感器检查方向与错误计数；退出传感器诊断后再运行src.run，避免同时管理Overlay。
6. 开始录像，口头说明本批run_id、带速、模型版本；只规划时机械臂不执行运动。

切回板端根 `config.json`，确认六状态标签与仅4种缺陷的bins映射、标定、bins/ROI、编码器尺度、IP、时序和窗口均为现场值。先保留 `dynamic.commissioned=false`，不带execute运行。输送带先达到已验证的稳定速度：

```bash
python3 -m src.run --config config.json --frames 3000
```

这不是脱离硬件的仿真：需要真实相机、包含编码器的位流、编码器、模型服务和标定。即使 `vision.backend="cpu"`，动态入口仍加载PL编码器。

检查 `board/events.jsonl` 中tracking、semantic_result和intercept_plan，以及 `board/last_plan.json` 的track_id、类别、预计接触点、时间和error_bound。当前dry-run在找到**第一个可行计划后返回**，不是持续跑完3000帧；3000是最多观察帧数。多次改变起始位置重复验证。

**通过：**单件从观察区进入时编号稳定，类别绑定同一物体，预测点在窗口内，误差预算有实测依据。遮挡/合并/旧检测不应继续套用旧编号抓取。没有计划时顺序查：掩码 → 类别键 → 标定凸包 → 编码器/带速 → 时间和误差预算 → 窗口，不是先放宽所有阈值。

另开板端只读终端执行 `tail -n 30 board/events.jsonl`，查看末尾事件；生成计划后 `python3 -m json.tool board/last_plan.json` 检查。把计划、预览和日志立即另存，下一次会更新。无计划时先看程序是正常达到帧数上限还是异常退出。

| 现象 | 定位顺序 |
|---|---|
| 没有tracking候选 | 相机实际帧、掩码、面积、标定覆盖区 |
| 有轨迹但没类别 | detect服务、口令、框与几何覆盖、类别键 |
| 对象编号反复改变 | 视觉噪声、多个物体合并、编码器方向/尺度、关联半径 |
| Belt stopped/reversed | 是否已稳定正向运行，范围是否实测，计数是否正确 |
| Frame too old / deadline missed | 相机缓冲、CPU负载、日志/网络、实测延迟与允许余量 |
| Window outside / error bound过大 | 可达范围、观察距离、动作时间和误差预算；不先增大容差 |

完成这些验收并确认整条动作路线后，将根配置 `arm.commissioned` 与 `dynamic.commissioned` 设为true，只做1件：

```bash
python3 -m src.run --config config.json --execute --max-tasks 1 --frames 3000
```

保留同步录像和程序日志；从大、轻、易抓样品开始。每次结束人工检查是否真的夹起、是否放对盒、是否留下了错误证据。原物体随带离开视野不算抓取成功；`bin_visual_evidence_pass`也不等同于舵机ACK或物理成功真值。

实际执行前清空轨迹，按原厂上电顺序给KM1动力。程序启动的准备动作本身就会移动机械臂，因此必须先确认home及准备路线。出现异常先停止后续任务并按现场情况切断动力，保存日志，再分析；按Ctrl+C不能撤销已经发出的舵机命令。

## 15. 第十三步：逐步扩大到连续批次和故障试验

单件反复稳定后，先5件，再20件，保持低速、稀疏供料：

```bash
python3 -m src.run --config config.json --execute --max-tasks 5 --frames 3000
```

程序的max-tasks是软件验证完成剔除任务上限，不是所有入场物体数。正常件直行单独统计；机械臂忙时流到末端的缺陷件仍计入全部缺陷件分母和漏剔。现有自动日志不提供完整的正常放行账本，以录像人工真值补齐。

建议把批次表记为：`run_id, object_id, true_class, speed, entered, attempted, grasped, correct_bin, missed, rejected, intervention, video_time, failure_reason`。每批保留配置、模型指纹、日志、录像和文件摘要；`last_plan.json`与预览会覆盖，证据要按run_id另存。

故障先在不执行机械动作时注入，再在已清空范围的受控状态确认：

| 场景 | 要观察的结果 |
|---|---|
| PC服务停止、网络断开/超时 | 停止后续动作，记录原因，人工确认后重新开批次 |
| 目标被拿走、遮挡或两个目标合并 | 不沿用错误ID/旧类别继续抓 |
| 停带、反转、速度异常、编码器错误 | 记录停止原因，不发后续抓取指令 |
| 盒满、遮挡、无入盒证据 | 进入人工检查，不能无限重抓 |
| 串口失败、已发命令后软件停止 | 如实记录当前机械运动仍可能继续，现场动力停止措施有效 |

不要通过接入错误电压/短路进行“故障注入”。脉冲异常可用测试源/仿真或正常接口断连试验，但断连不一定立即形成非法跃迁，需验证软件实际能检测的条件，不虚构所有断线都能立刻停机。

**通过：**成功与失败均可追溯；异常不会继续盲目发送后续序列；连续运行、人工干预和恢复方式有记录。根据失败原因调整后再重复相应试验。

### 15.1 每批数据怎么留存

每批结束、程序退出后在板端执行下面片段。它创建唯一目录，复制已有证据，不删除历史日志。events.jsonl可能包含多个会话，summary只汇总最新会话，人工表用本批run_id关联录像。

```bash
python3 - <<'PY'
from pathlib import Path
from datetime import datetime
import shutil, subprocess, sys
root = Path('metrics/evidence') / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
root.mkdir(parents=True, exist_ok=False)
for name in ('config.json', 'data/calibration.json', 'data/calibration_pairs.json',
             'board/events.jsonl', 'board/last_plan.json', 'board/latest.jpg', 'board/mask.png'):
    p = Path(name)
    if p.is_file():
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
if Path('board/evidence').is_dir():
    shutil.copytree('board/evidence', root/'board/evidence')
subprocess.run([sys.executable, '-m', 'src.summarize', '--log', 'board/events.jsonl',
                '--output', str(root/'summary.json')], check=True)
print(root)
PY
```

把输出目录名写进逐件记录表，补入带速、PC模型/环境指纹、录像文件名和本批失败原因。Windows通过scp取回该目录，录像放相同run_id文件夹。板卡本地时间不是跨设备精确同步；录像与日志按明确事件对应，不凭文件名时间测端到端延迟。

### 15.2 故障演练的具体操作

先只规划：在PC服务终端Ctrl+C，观察板端是否记录推理异常；恢复服务、health与真图通过后开新批次。对目标丢失，用不进入机械运动区的方法在上游移除/遮挡样品，核对旧编号不再继续使用。停带/反向只在机械臂不执行时测试传感器与判断；不让人手接近运动夹爪做演示。

物理执行阶段先布置已验收的停机措施，再做必要受控故障验证。每次记录触发方式、程序反应、残余机械动作、人工措施和恢复步骤。未观察到的保护能力不写成已实现。

## 16. 第十四步：按比赛要求收集五类核心和专项指标

| 指标 | 操作与口径 | 证据 |
|---|---|---|
| 任务闭环成功率 | 测试前定义有效目标；正确抓取并分类入盒数/所有有效入场目标数，同时报告抓取尝试成功率、漏抓和干预 | 逐件表、同步录像 |
| 端到端物理响应 | 明确从传感器事件/采集完成到机械臂实际响应的端点；同一外部时基测量，报告P50/P95/P99/max，另报到接触与完整周期 | 波形/录像、换算方法和原始记录 |
| AI效果与性能 | 独立测试mAP@0.5:0.95、P/R；batch=1预处理/推理/后处理/总耗时，持续吞吐 | evaluate_yolo输出与原始测试集 |
| FPGA量化贡献 | 同一板、图片、算法比较CPU与PL，包括打包/DMA；再对比整机窗口错失和成功率 | benchmark、完整批次数据 |
| 两端通信 | health RTT与真实detect事务分开，记录超时/错误/负载 | RTT CSV、事件日志 |
| 专项1：资源与时序 | LUT/FF/BRAM/DSP、目标/实际时钟、WNS/TNS与约束检查 | Vivado/HLS报告 |
| 专项2：可靠性 | 连续运行时长、异常与干预、故障恢复/停止行为 | 连续日志和故障试验表 |

开始正式统计时，建议至少100个有效入场目标，分至少3个已验收带速；这是项目建议，不是指南统一门槛。P99样本少时不稳定，应报告样本数并增加持续测量，不能从几个抓取推导尾延迟。

现成命令：

```bash
python3 -m src.summarize --output report/session_summary.json
python3 metrics/scripts/measure_rtt.py --url http://192.168.2.1:8765 --count 100 --output metrics/logs/health_rtt_run1.csv
```

运行前设置同一KM1_TOKEN。summary是软件日志汇总，不自动得到人工真值、相机曝光时刻或物理端到端延迟。health请求小，不能代替整帧PNG上传与检测响应的事务耗时。

CPU/PL整机对照必须保留同一编码器、模型、输入、带速和机械参数，仅切换图像backend。做无位移补偿、框中心/几何点等额外消融时，先离线回放比较，不为了对照让已知错误策略实物盲抓。

将每项指标填入 [metrics/metrics.csv](../metrics/metrics.csv)，包含硬件/模型精度/输入/版本/时钟/功耗模式/次数时长/证据路径。延迟与网络不要直接相减不同机器的时钟；无时钟同步时分别报告同机时段和RTT。

### 16.1 一轮正式测量怎么安排

1. 锁定模型、配置和样品范围；正式上位机使用已验收HX 370。为每件样品编号，记录真实类别，先定义哪些来料算有效目标。
2. 按已通过的工作速度分组，录像与程序一起记录；每组包括六种状态、有间距的单层样品；分别统计正常误剔与缺陷漏剔。建议累计至少100件；只有一个速度通过时先如实报告这一速度，不为凑组数进入未验收速度。
3. 逐帧核对有效入场、是否尝试、是否夹起、是否入对盒、流出/拒抓和人工干预；分别算闭环成功率与尝试抓取成功率。
4. 用第15.1节存档脚本保存每批；另在HX 370运行evaluate_yolo，记录正式AI指标。health RTT使用现成脚本，真实detect事务用事件中的rpc_ms统计，两者分开。
5. 在相同图像集做CPU/PL benchmark。整机对照只将根配置复制为专用对照配置并切换vision.backend，保留同一PL编码器、模型、机械参数和带速。正式动作仍须按相同验收条件；不能运行两份控制器争用设备。
6. 从同一外部时基的录像/测量记录计算物理响应，注明起止事件、帧率/时间分辨率、样本数。软件frame_age_ms不是曝光延迟，health不是完整识别时延。
7. 报告资源/时序、连续运行时长与故障演练。未测项留空并说明，成功率不只截取最顺利一段。
8. 填metrics.csv时把证据文件名写全，另一位队友随机抽一个数字回算，确认可追溯。

## 17. 第十五步：迁移 AMD 正式机器，再完成提交包

当前开发与部署机器为已到位的 **AMD Ryzen AI 9 HX 370 / Windows 11 AI PC**，原 Y9000P 已退出使用。按[Windows 11正式部署手册](amd_deployment.md)复用已有 ROCm 环境，驱动更新及 GPU 复验已完成，继续本队模型迁入和通信验收；正式统计前完成整机复验，NPU不是当前任务的前置条件。

本次已查到AMD正式发布的 [ROCm 7.14.0说明](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html)及[兼容矩阵](https://rocm.docs.amd.com/en/docs-7.14.0/compatibility/compatibility-matrix.html)。因此不是“这个版本是否存在”的问题；仍须针对实际CPU/APU、OS、驱动和PyTorch组合选择安装步骤，并运行真实YOLO验证。

按 [AMD部署手册](amd_deployment.md)执行：

1. 记录整机、准确APU型号、RAM、OS、驱动、ROCm/PyTorch版本和安装来源。
2. 先安装AMD对应组合的torch/torchvision，再安装本项目YOLO依赖；不要在AMD环境执行Windows CUDA安装脚本。
3. 复制权重、model.json、独立测试集和源码，不复制Windows虚拟环境。
4. 检查 `torch.version.hip`、GPU可用性，并用 `config/host_rocm.json` 跑真实测试图。
5. 启动detect_server，更新板端IP/口令/模型指纹配置，重做完整物理闭环和指标测试。
6. 预下载权重与依赖，关闭Internet连接共享、断开电脑Wi-Fi但保留直连网线，验证离线冷启动运行。

最后按README目录对照整理：源码和构建脚本、实际XDC、bit/hwh、HLS/Vivado报告、模型及来源/许可/数值精度、数据说明、环境锁定文件、接线和BOM、实测指标/原始证据、失败分析、从零复现说明。skill目录可整理经过实测的DMA排错/Overlay校验/部署流程，但只宣称已验证的复用效果。

工程文件命名用英文；排除虚拟环境、缓存、真实口令和无关旧包。当前 `report/manifest.json`、`release_checks.json` 是旧发布记录，增加本文/XDC和产生实机产物后，最终提交前需要重新生成发布检查，不能冒用旧摘要。

找没有参与主要开发的队友，按文档从新环境复现。记录失败位置和实际耗时，修正文档后再录最终演示。

## 18. 按实际到货与最终系统依赖推进

| 阶段 | 工作内容 | 进入条件 |
|---|---|---|
| 当前 | 第3.2节基础启动验收全部通过 | 上传工程、C270采图；项目位流和整机闭环仍待完成 |
| 板卡启动 | 写Z2镜像、启动、联网、上传最终工程 | 卡与读卡器到位 |
| 感知和模型 | 先预采训练，传送带安装后补采/最终标定 | 临时数据不得冒充最终工位验收 |
| FPGA部署 | 准备board files、构建bit/hwh、PL一致性验证 | 工具可运行；上板需要已启动的PYNQ |
| 动态准备 | 安装输送带/编码器、确认电平、位移和时序标定 | 缺件到位，接口和位流可用 |
| 最终联调 | `src.run`只规划 → 有界分拣 → 批次与指标 | 所有现场参数及路径已确认 |
| 正式交付 | HX 370 Windows 11实机、对照实验、报告及复现包 | 正式环境与最终系统验收通过 |

全文已展开，按开头依赖表推进；旧日期不代表已完成。接线、机构、固件或配置不变时，复用已有有效记录。

### 18.1 发布包逐项装什么

在电脑新建带日期的发布目录，复制下面已验收内容；运行目录继续保留，发布目录不用于试验。真实口令、虚拟环境、缓存、无关旧ZIP不打入包。

| 发布内容 | 来源/具体检查 |
|---|---|
| 源码和构建入口 | src、build脚本/XDC、tools、sim；排除build的大型临时工程 |
| 板端电路 | 同批board/vision.bit与vision.hwh、SHA256、工具版本 |
| 现场参数 | 最终config、calibration及配对原始点；说明IP需换成复现者网络 |
| PC模型 | best.pt、best.onnx、model.json、来源/许可和FP32说明 |
| 数据说明 | 类别、标注、划分原则、独立测试集及许可；可追溯到实际测试 |
| 环境 | 当前 HX 370 的安装源、依赖、驱动、机器记录；Y9000P/CUDA仅留历史证据 |
| 工程证明 | HLS仿真/综合/协同仿真、Vivado资源/时序/DRC及实际XDC |
| 指标和证据 | metrics.csv、逐物体表、原始日志、录像、对照与失败记录 |
| 文档 | 架构、接线/BOM、启动步骤、故障恢复、设计报告、演示说明 |

在发布目录重新计算全部文件摘要，不能沿用旧report/manifest.json。下面片段在**已经整理好的发布目录**打开PowerShell执行；将末尾解释器版本换成该电脑已安装版本，它只读发布文件并新增清单，清单已存在时拒绝覆盖：

```powershell
@'
from pathlib import Path
import hashlib, json
root = Path.cwd()
dest = root/'release_manifest.json'
if dest.exists():
    raise SystemExit('Existing manifest: use a new release directory')
rows = []
for p in sorted(root.rglob('*')):
    if p.is_file() and p != dest:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                h.update(chunk)
        rows.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
dest.write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
print('Files:',len(rows))
'@ | py -3.13 -
```

目录打ZIP后解压到新的复现目录，逐项核对清单、模型、位流、配置和链接。若内容有任何变更，重新生成新的发布版本，不拿旧摘要配新文件。

### 18.2 比赛现场从关机到演示

1. 按接线照片装好系统，确认样品、盒、供电与动力停止措施；固定相机和机构，运输导致位移时复核标定。
2. HX 370与PYNQ网线直连并上电，确认192.168.2.1/192.168.2.99；HX 370接交流电、避免睡眠，打开正式工程PowerShell。
3. 设置口令，启动AMD部署手册第7节的ROCm服务；板端登录，设置同一口令，检查health。
4. 相机画面、盒区、配对位流和编码器状态正确后，低速空带稳定，执行一次只规划。退出诊断程序再启动分拣。
5. 清空轨迹，按原厂顺序给KM1动力；开始同步录像，运行已验收的有界批次命令，按约定间距供料。
6. 展示类别、预测位置、实际入盒及统计；说明PL图像处理和编码器的实测贡献。展示一种已经演练过的异常处理，不临场增加危险动作。
7. 批次结束先停止供料和带，确认机械动作结束，保存证据；停止服务，板卡正常关机，再切动力和其他电源。
8. 让另一位队友照本节独立完成一次，记录卡在哪一步并修正文档。再冻结包、录最终视频、检查上传回执。

## 19. 验收勾选表

- [ ] M0：设备、供电、相机支架、末端接料和现场断电措施就绪。
- [ ] M1：PYNQ可重启登录，基础Overlay正常。
- [ ] M2：C270在640×480稳定采图，实际控制项和曝光/光照记录完整。
- [ ] M3：最终工位的KM1接口、行程和动作路线已确认，记录可复用。
- [ ] M4：6个以上拟合点和3个以上独立点通过，背景掩码可用。
- [ ] M5：六状态缺陷模型和detect协议联通，类别及目标几何对应正确。
- [ ] M6：实际编码器型号、电平与计划引脚核对，位流构建并上板。
- [ ] M7：PL与CPU逐像素一致，实际编码器每圈计数及毫米比例验收（SEN0230四倍频目标1600/转）。
- [ ] M8：采集时延、动作时序、带速和误差预算完成实测。
- [ ] M9：动态只规划正确，低速单件成功。
- [ ] M10：连续批次和异常处理有录像及原始记录。
- [ ] M11：HX 370/Windows 11/ROCm实机闭环和五类核心+两项专项指标完整。
- [ ] M12：最终包可从零复现，报告数字与原始证据一致。

来源说明：比赛要求依据用户提供AMD指南物理第13～20、26页，2026-10-07再次核对第13～15页平台条款；PYNQ镜像、AMD安装及标注格式入口于2026-10-07在线核对，链接随步骤给出。引脚等硬件依据保留原来源，使用前核对实物。命令参数按当前src/config/build核对；[历史验证记录](../report/validation.md)不能替代本手册的实机验收。
