# TACO 作者原始模型本机部署

2026-10-29：用户要求直接部署作者模型，并明确将原始十类归并为瓶子类、罐子类、其他类。原始十类识别结果继续保留，归并在推理后完成。

**实时预览入口：`http://127.0.0.1:8781/`。** 用户随后要求接入实时相机，现已连接 PYNQ `/dev/video0`：左侧持续刷新相机画面，右侧显示最近一次 TACO 识别对应的帧及画面接收年龄。原8777上传图片服务保留。当前模型在CPU连续识别，每次约数秒；不能将右侧旧帧轮廓叠加到左侧新画面。

实时功能使用独立相机只读桥接，不驱动机械臂、串口或FPGA位流。相机采集与推理使用有界最新帧，繁忙时丢弃中间帧；持续预览仅保留内存中最新画面与识别，不逐帧写盘。断线/超时后隐藏过期画面和结果，并自动尝试恢复。可以暂停/恢复连续识别，相机画面仍继续显示。

### 启动实时预览

相机桥接使用根目录Python3.11环境和 `requirements_camera_preview.txt` 的Paramiko依赖；本机已安装并验证SSH连接、设备存在及持续JPEG采集。私有连接模块只在本机，暴露 `connect()` 返回已验证的SSH连接；账号和连接配置不写入Git。本机现用模块为 `local-only/arm-speed-grasp-20261028/camera_recovery_connection.py`，其他机器须使用自己的私有模块。

根目录分别在两个终端运行，相机桥接只使用已授权的摄像头：

```powershell
.\.venv\Scripts\python.exe tools/camera_preview_bridge.py --connection-module local-only/arm-speed-grasp-20261028/camera_recovery_connection.py
```

```powershell
.\local-only\taco-deployment\python37\python.exe pynq_km1_vision/src/taco_preview_server.py --port 8781 --camera-url http://127.0.0.1:8780/frame.jpg
```

8780只绑定回环地址，`GET /frame.jpg`返回当前JPEG和 `X-Frame-Id/X-Stream-Id/X-Frame-Age-Ms`；源帧接收超过2秒不再返回图片。相机远端采集线程自身也拒绝超过2秒未更新的帧。这里的PC时间是接收时间，不冒充板端硬件曝光时间。8781后台线程轮询最新帧进行识别，识别和图片上传共用串行模型锁；HTTP连接仍并发处理。`--camera-url`只接受本机HTTP来源。

8781新增接口：`GET /camera.jpg`为当前相机画面；`GET /live-result`为最近一次实时识别JSON，包含与原图绑定的 `overlay_base64`、独立 `frame_id`、`source_frame_id`、`camera_received_monotonic_ms`、`result_age_ms`，不将结果当作当前画面；`POST /live`请求 `{ "enabled": true/false }`切换连续识别。健康接口增加 `live_available/live_enabled/camera_connected/camera_frame_id/inference_error`。三类映射、原始权重和推理筛选门槛不变。

8项针对性测试通过（类别/掩码、浏览器连接、相机新鲜度、结果帧身份、回环来源限制）。实际PYNQ相机已出图，浏览器显示相机已连接和连续识别结果；现场预览不代表自动抓放或识别质量验收通过。实时检查记录在 `local-only/taco-deployment/live-verification.json`。

## 来源与环境

- [作者仓库](https://github.com/pedropro/TACO)，源码固定为 `70479ad9adc3da971a8ae75ac97eb1026910eec7`。
- [官方发布页](https://github.com/pedropro/TACO/releases/tag/1.0)，使用 `taco_10_0.zip`，四折中的 fold 0；不按本地图片挑选其他折。
- [原始下载地址](https://github.com/pedropro/TACO/releases/download/1.0/taco_10_0.zip)，ZIP SHA256：`d61a58aed158adb30bac5e2f7c59beb67aa82c6eb09ef52d2f30a5e2325e4a38`。
- `mask_rcnn_taco_0100.h5`，179375072 字节，SHA256：`d79517accf4e4a0858db5fbe02df3c1f7572def76abf8ecf981398d67a4e73f1`。
- Windows x64 便携 Python 3.7.8、TensorFlow 1.15.0、Keras 2.1.6、NumPy 1.18.5、h5py 2.10.0，CPU 四线程。依赖见 `requirements_taco_legacy.txt`。不安装到 `.venv` 或 `.venv-rocm`。
- 作者发布页写 TensorFlow 1.9.0，本机采用 1.15.0 兼容运行；不是完全相同环境，也未使用 AMD GPU。Keras 2.2.4 缺少源码需要的旧加载接口，改用 2.1.6 后跑通。
- 原始源码未修改。加载方式与作者入口一样按层名加载，并额外逐层核验所有带权层的集合、权重数量及形状，避免静默跳过分类头。

模型、下载包、环境及图片输出仅存根目录 `local-only/taco-deployment/`，不进入 Git。模型有上述官方公共下载地址；本机工位照片和生成结果没有共享地址。

## 原始类别

类别 ID 来自作者 `dataset.py` 对 `map_10.csv` 目标名称排序，0 为背景。

| ID | 原始输出 | 含义 |
|---|---|---|
| 1 | Bottle | 塑料瓶及玻璃瓶 |
| 2 | Bottle cap | 塑料瓶盖及金属瓶盖 |
| 3 | Can | 饮料罐、食品罐、气雾罐 |
| 4 | Cigarette | 烟头 |
| 5 | Cup | 纸、塑料、泡沫及玻璃杯 |
| 6 | Lid | 塑料盖及金属盖 |
| 7 | Other | 纸盒、纸张、电池、碎玻璃等其他物品 |
| 8 | Plastic bag + wrapper | 塑料袋及包装物 |
| 9 | Pop tab | 易拉罐拉环 |
| 10 | Straw | 塑料及纸吸管 |

用户确认的映射如下，配置见 `sorting_mapping` 和 `sorting_classes`：

| 分拣 ID | 显示名称 | 原始类别 |
|---|---|---|
| bottle | 瓶子类 | Bottle、Bottle cap、Cup、Straw、Lid |
| can | 罐子类 | Can、Pop tab |
| other | 其他类 | Cigarette、Other、Plastic bag + wrapper |

这是用户定义的分组，不能作为塑料/金属/纸类材质结论。模型分类头、权重、筛选规则不变，不重训，不合并各实例掩码，不改变原始概率。模型未检出的物体不会自动生成“其他类”实例；未知原始标签或缺少映射会报错，而非悄悄划入其他。分组已配置，机械箱位与路径仍待独立对接。

## 安装和启动

根目录运行，下载源码、便携解释器及约 167 MB 权重包；旧版 TensorFlow 安装包约 295 MB：

```powershell
.\tools\setup_taco.ps1
.\tools\setup_taco.ps1 -StartPreview
```

第二条保持前台运行，预览页为 `http://127.0.0.1:8777`，支持上传图片，显示三类计数、分拣类别、原始十类识别、实例轮廓和推理耗时。轮廓按分组着色，页面列明全部归并规则。只绑定本机回环地址；不自动启动相机、连接 PYNQ、发送机械动作或代替正式板端推理服务。

已安装时也可直接启动：

```powershell
.\local-only\taco-deployment\python37\python.exe pynq_km1_vision/src/taco_preview_server.py
```

单图命令在主工程目录运行；输入使用操作者自己的图片，输出目录必须尚不存在：

```powershell
..\local-only\taco-deployment\python37\python.exe src/taco_inference.py --image <实际图片路径> --output-dir ../local-only/taco-deployment/my-check-001
```

使用作者 `test` 配置：ResNet50、最短边 800、最大边 1024、`DETECTION_SCORE_RATIO=true`、比值门槛 10、每类 NMS 0.3。不是 YOLO 的 0.8 置信门槛；未据样例调整参数。返回的 `confidence` 仍为类别概率，另返回 `score_ratio` 和 `background_score`。

## 独立识别接口

配置：`config/taco_original.json`；路径相对仓库根目录解析，启动时核验源码版本和权重哈希。

`GET /health` 给出加载状态、十类名称、模型 SHA256、CPU 后端、`sorting_mapping` 和 `sorting_classes`，`sorting_mapping_status=configured`。
`POST /detect` 的 JSON 请求为 `{ "image_base64": "<编码后的图片>" }`。只允许同源浏览器或本机客户端，请求体至多 8500000 字节，图片至多 1920×1080。
`GET /latest-result`、`GET /overlay.jpg` 返回最近一次上传结果及可视化；图片不代表实时相机画面。

响应 `schema_version=taco-observation-v1`，含 `instances`、`image_size=[width,height]`、`model_version`、`runtime_id`、`backend`、`inference_ms`。
每个实例保留原始 `class_id/label`、概率、比值、原始框 `detection_bbox_xyxy`、掩码紧框 `bbox_xyxy`、`area_px` 和 `mask_rle`，另返回 `sorting_bin=bottle/can/other` 和中文 `sorting_label`。`sorting_bin` 是逻辑分组，不是已标定的机械箱位。响应顶层同时携带映射，归并改变会更新 `runtime_id`，模型 SHA256 不变。框右下界不含。掩码是完整原图的**行优先、零段起始、0/1交替**整数 RLE，与本工程 CPU/PL 掩码表示一致；宽高超过 640×480 时不能直接交给当前 FPGA 核。

这不是原动态 HTTP 协议 v2，也没有将旧金属/塑料/纸类抓取/核验策略或 `commissioned` 配置改成新分组。`automatic_motion_allowed=false`、`quality_gate_passed=false`、`sorting_mapping_status=configured`。真实端到端部署仍需标定并绑定机械箱位、对接相机与现有调用方，并单独验证识别质量、实际可抓条件、板端通信和性能。

## 本次实测与限制

本机验证了权重完整加载、CLI 单图、网页服务健康检查、上传识别、轮廓下载。HTTP 无效 JSON、无效 Base64、外站 Origin、错误路径分别被拒绝。
初始掩码传输及类别契约三项测试通过；映射更新后四项测试通过，覆盖十类完整映射、错误映射拒绝和掩码传输。依赖检查和仓库轻量检查通过。没有训练，没有评分封存测试，也没有硬件、串口、综合、时序或机械动作验证。

冻结比值门槛 10，用现有**训练组**一张空景和三张源区照片做运行检查，不能当作独立准确率：

| 图像序号 | 内容 | 实际输出观察 | 单张推理耗时 |
|---|---|---|---|
| 0011 | 空工位 | 5 个 Other，箱体和十字标记被检出 | 3.364 秒 |
| 0012 | 易拉罐 | 目标 Can，概率约 0.815；另有背景 Other/Cup | 4.373 秒 |
| 0036 | 塑料瓶 | 瓶体被判 Other，概率约 0.751；另有背景 Other | 3.668 秒 |
| 0084 | 纸盒 | 目标 Other，概率约 0.417；另有背景 Other | 3.689 秒 |

本次不是正式质量验收。Other 高得分不保证目标是纸盒，当前工位背景也会被识别为 Other。不能把本机模型跑通写成已经适合自动分拣。

完整结果位于 `local-only/taco-deployment/smoke-summary.json`、`smoke-train-0011-complete-weights/`、`preview/` 和 `http-checks.json`。保留早期兼容性失败目录与错误记录，不覆盖旧模型和已有训练成果。

三类归并更新后，重启8777服务并通过真实图片HTTP复验：同一0012图的8个实例原始类别、概率、框和掩码与更新前逐项完全相同；新增分组为瓶子类1、罐子类1、其他类6。这里的瓶子类来自原先背景误检Cup，不表示图里有真实瓶子。模型SHA256不变，`runtime_id`随映射更新；页面与健康接口均返回已配置的三类。完整检查记录为 `local-only/taco-deployment/mapping-verification.json`。此轮只验证归并和服务生效，未提升识别质量结论。

## 页面无法打开的修复

2026-10-29 用户反馈无法打开8777。检查时进程和监听均存在，但连接队列中已有多个未完成连接，新请求被拒绝；原先单线程HTTP服务可能被浏览器预连接或不完整请求独占。改为并发处理HTTP连接，每条连接15秒超时，模型推理及证据写入仍串行，并处理客户端断开。此改动不改变模型或三类映射。

验证命令：在主工程目录运行 `..\.venv\Scripts\python.exe -m unittest sim.test_taco_inference sim.test_taco_preview_server -v`，5项通过。新增回归同时保留8条空闲连接和1条未完成上传，首页及健康接口仍在2秒内返回。实际服务重启后另保留8条空闲连接，首页与健康接口均返回200，内置浏览器刷新后已显示“三类映射已就绪，请选择图片”。
