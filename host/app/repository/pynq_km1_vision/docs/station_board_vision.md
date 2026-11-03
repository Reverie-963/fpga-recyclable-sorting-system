# 板端连续视觉核验：新候选链路

**后续执行接口已接通：**现役KnownItem快档已[直接使用本新视觉接口](sorting_station_interface.md)，移除旧双核远程依赖；PC做抓点/语义掩码统计，PL做原图六区持续变化。金属/塑料快档已各执行一次并获操作者动作确认，失败及补证分开记录；当前置信度按队长要求为0.7。图像服务仍不单独授权机械动作，原生产和无人值守门槛未放行。下文“旧执行器待迁移”为切换当时状态。

**已切换（2026-10-29）：**队长确认旧联调结束后已启用新位流与板端服务，8779实时诊断工作；12次启动对照及3组真实窗口/9帧CPU重算一致，持续60.5秒无故障。板端ARM参考165.71ms、PL完整处理68.67ms，约2.41×；同输入本机HX370为15.11ms，因此该倍数仅为板端卸载收益。ESP32与v8/ROCm模型不变，旧机械执行入口因接口迁移受保护。详见[切换实测与未完成项](../../docs/team/reviews/2026-10-29-board-vision-switch.md)。下文暂存/等待状态为本次切换前历史。

2026-10-29，队长决定以板端连续图像处理和放置后视觉核验作为 FPGA 主线，并明确保留 KM1 的 ESP32。当前 GPU 三类识别和 ESP32 已测执行链路不因本候选自动改变。原远程双核几何/掩码统计仍被正在集成的单件执行器使用，不能把“迁移目标”写成“已经全部停用”。

## 本次实现

相机仍通过 USB 接板上 ARM/Linux；原始 BGR 帧留在板端。新 `station_vision_accel` 读取基线和当前图像，一次计算 RGB 三通道绝对差之和、阈值变化、六个区域的统计及窗口内持续变化。六区为取物区、塑料/金属/纸类箱、两个静态背景区。没有相机直接接入 PL、材质分类、自动持物/脱落判断、传感器改装或 ESP32 替换。

新驱动长期复用基线、当前帧、历史、区域配置和结果五个缓冲区，HLS 主端口直接访问 DDR，不再每次申请 DMA 缓冲区，不回传整张输出图给 PC 计算哈希。每次结果仅 48 个整数。基线只有明确操作才更新，三帧窗口结束后重开持续变化窗口，避免以从基线以来的无限交集导致后来出现的变化永久归零。

像素计算保持精确规则：`abs(B1-B0)+abs(G1-G0)+abs(R1-R0) > threshold`，阈值范围0～765。矩形为 `[x0,y0,x1,y1)`，输出包围框含端点，无变化时为四个0；每区输出变化面积、窗口持续面积、差值总和、四个边界和区域面积。窗口持续变化按像素交集计算，不以数量相同冒充位置一致。初版未加入形态学去噪、材质语义或抓牢判断。

PS 服务只打开一份相机，保留四帧有界缓存；CPU 参考用于开发验证，正式 PL 服务没有静默 CPU 回退，不逐帧重复计算 CPU 对照。当前输出仍为诊断证据，`semantic_verified=false`、`automatic_motion_allowed=false`、`physical_success_verified=false`。它不能独立授权机械动作或替代现有正式验收。

启动相机前还会用合成 BGR 图做3次预热和12次板端 ARM CPU/PL 对照；整数结果不一致则拒绝启动相机，不自动回退。保存输入哈希、逐次耗时和配对资产；PL计时包含输入整理、缓存同步及结果读回，排除采图、JPEG和网络，因此属于板端启动自检，不是整机或真实分拣加速成绩。此次尚未实际运行板端启动自检。

## 文件和接口

- `src/hls/station_vision_accel.cpp` 与独立 C++ testbench：新核和数值参考。
- `board/km1_station_vision.py`：配对资产校验、持久缓冲、带超时的任务号/状态核对，不自动重载位流。
- `src/station_vision.py`：CPU 参考及基线/帧号/窗口状态机。
- `src/station_vision_server.py`：板端相机、原图缓存和核验服务，默认仅绑定127.0.0.1:8783，需要环境中的令牌；导入模块不连接设备。
- `tools/station_vision_bridge.py`：通过既有 SSH 账号把板端接口映射至本机8780，不自行启动远端程序或加载位流。
- `tools/build_station_vision.ps1`：在新的 local-only 目录构建独立候选 bit/hwh，不覆盖现役 `vision.bit/hwh`。
- `tools/stage_station_vision.py`：默认仅上传独立候选目录；显式 start 也会在旧相机或统计服务仍活动时拒绝加载，不能抢占现役服务。
- `config/station_vision.example.json` 为模拟坐标，板端服务拒绝直接使用。现场配置及令牌不进入 Git。

板端和本地桥接提供 `GET /health`、`/frame.jpg`、`/vision`、`/snapshot.png?frame_id=...`；`POST /baseline` 更新诊断基线，要求相同 stream_id、配置指纹和明确的、0.5秒内缓存帧号。另有未部署的可选`baseline_snapshot_v1`扩展，将同一次绑定回执和该帧PNG合并返回，客户端默认关闭，详见[接口扩展](sorting_station_interface.md#2026-10-31-候选不可变png缓存与基线原图合并响应)。无 execute、grasp、serial、PWM 接口。原图 PNG 带解码 BGR 哈希，JPEG 不冒充原始像素；缓存帧超过2秒即不可用。捕获间隔超过配置上限会使基线失效，硬件异常停用证据、等待复核。

8779 综合页面新增“FPGA 板端视觉核验”，只有配置/位流/基线/帧身份匹配才显示数据；未切换时明确显示待部署。旧远程统计性能表默认折叠到“历史记录与迁移依据”，原始负收益数据继续保留，不能隐藏后宣称已获得加速。抓点及真实执行历史继续显示。

## 验证状态与切换边界

当前已通过 C 仿真（含640×480）及14笔小尺寸 C/RTL 联仿。第一次联仿因 testbench 缓冲区小于 m_axi 声明深度而失败，第二轮修正为完整接口深度后通过，首次日志保留。HLS 像素循环达到 II=1 的综合约束，仅为综合结果；不能据此宣称板端或整机吞吐。

Python 68项针对性测试及独立桥接1项测试通过，覆盖 RGB 位宽、严格阈值、区域边界、持续变化的像素交集、窗口重启、旧帧拒绝、基线绑定、故障不回退、令牌/来源校验与不存在机械接口。HTTP回归固定使用本机直连，排除系统代理对端口测试的影响。

Vivado 配对候选已于18:39构建完成：100MHz，setup WNS=0.134ns、hold WHS=0.006ns，DRC无错误。bit SHA256=`b03281dbd650465402cf711d230605a72712318915129814b0dc808dc261d2af`，hwh SHA256=`e7adf114088f17fe4af130d394190e75ff3cca19f95bcf81124ad85c6469de5b`。构建、冻结源、报告与candidate-review.json仅在本机 `local-only/station-vision-build/20261029-second/`，无共享下载地址。

配对文件及新PS程序已上传板上独立候选目录，所有传输文件（令牌除外）的哈希已核对；状态为staged_not_started，记录位于 `local-only/station-vision-deployment/20261029-1839/stage-result.json`。没有启动候选或加载新位流，现役相机、旧双核服务、模型指纹和ESP32保持不变；因此未获得新链路真实加速或CPU占用成绩。

队长已明确选择“等旧联调结束，再切换新视觉链路”。当前另一联调会话仍等待奶盒摆放及后续试验，因此保留旧服务，新候选不启动。本会话已安排每10分钟跟进；只有明确结束、无持物未结束任务、串口关闭且无设备租约时才切换，不把idle或turn completed等同于联调结束。

切换会改变相机 stream_id，且新位流不含旧几何核，旧执行器必须拒绝不匹配的资产。统一切换时才能停止旧服务、加载候选、核对新相机身份并重新验证调用方，不能在两个会话之间交替重载。旧执行器尚未迁移到新视觉服务，不能在现役配置中直接替换backend或声称已完成新架构的实物闭环。后续只做图像服务切换、诊断与真实性能测量，不自动发送机械指令；完成本次迁移后暂停跟进。

新链路需补测：相同输入下板端 ARM CPU 与 PL 的完整处理耗时（含输入整理、缓存同步、结果读回），相机至证据的端到端延迟、持续负载、CPU占用；还需在加入GPU语义和新执行适配后重新做实物正负例验收。本次不新增机械动作、不修改 ESP32、不更改旧 commissioned 标记，没有创建本轮 PR。

```powershell
./tools/build_station_vision.ps1 -ToolRoot <AMD2026.1目录> -Stage Full
cd pynq_km1_vision
..\.venv-rocm\Scripts\python.exe -m unittest sim.test_station_vision sim.test_monitor_station_vision sim.test_monitor_fpga_benchmark sim.test_monitor_grasp sim.test_monitor_control sim.test_monitor_benchmark sim.test_yolo11s_monitor sim.test_yoloe_inference sim.test_yoloe_live_preview -q
```
