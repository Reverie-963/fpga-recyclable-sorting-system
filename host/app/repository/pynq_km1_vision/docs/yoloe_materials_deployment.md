# YOLOE 塑料、金属、纸质三类识别预览

**后续更新：已完成[ROCm GPU部署与实时监控](yoloe_realtime_rocm.md)，当前8778入口使用Radeon 890M/FP16并接入既有相机桥接。下文CPU环境、无相机入口和首次试跑是历史记录，按需仍可显式运行，不代表当前实时服务后端。**

2026-10-29：按用户要求试用 YOLOE，输出固定为塑料、金属、纸质三类。当前是独立图片识别服务，已实际加载权重、运行工位图片并验证浏览器交互；识别质量尚未通过自动分拣验收。

## 模型及两种提示方式

采用官方 `yoloe-11m-seg.pt`，独立 Python 3.11 / PyTorch 2.6.0+cpu / torchvision 0.21.0+cpu / Ultralytics 8.3.230 环境。既有 `.venv`、`.venv-rocm` 和正在进行的 GPU 训练不作修改。本次实际使用 CPU 四线程，未验证 YOLOE 的 ROCm/GPU 性能。

- **文字提示**：塑料瓶、塑料容器、塑料袋、金属罐、纸盒、纸张、饮料纸盒七种具体提示，汇总为三类；精确英文词及顺序见[配置](../config/yoloe_materials.json)。
- **参考图提示**：当前本机使用原训练组的 8 件实物，每件取 2 张源区姿态和 1 张箱内图片，共 24 张。按已审核轮廓的包围框提取视觉提示，每个示例保留独立提示索引，再映射到同样的三类。仅生成提示嵌入，没有反向传播或微调。
- 已准备视觉权重时，页面默认参考图模式；两种模式均可切换。只有官方权重和文字缓存的其他电脑默认文字模式，不伪造参考图能力。

参考图模式是已登记样件的外观匹配，不代表能可靠判断任意新物品的材质。饮料纸盒按本项目纸质包装去向归类，并非声称复合包装只有纸。未区分材质的通用 bottle 标签没有直接改叫塑料。

## 来源与产物

[官方模型说明](https://docs.ultralytics.com/models/yoloe/)；权重下载自 Ultralytics 官方 assets 的 v8.3.0 发布：

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `yoloe-11m-seg.pt` | 60276714 | `785199b7cc75a5f4041dec15782a4b8a4105e5a34d7ffb287a0a10781724c5bd` |
| `mobileclip_blt.ts` | 599764649 | `a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54` |

下载地址分别为 `https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11m-seg.pt` 和 `https://github.com/ultralytics/assets/releases/download/v8.3.0/mobileclip_blt.ts`。CLIP 分词器固定到官方 Ultralytics fork 的提交 `7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40`。第三方组件沿用各自许可。

全部运行文件位于根目录 `local-only/yoloe-deployment/`，不提交权重、图片、环境或机器设置：

- `venv/`：独立 CPU 环境；`installed-requirements.txt`：本机依赖快照。
- `assets/deployment.json`：来源、官方哈希、派生权重哈希、配置指纹及提示顺序；启动时逐项核验。
- `assets/yoloe-materials-prompted.pt`：缓存文字提示的派生权重，本机 SHA256 为 `b5301654b1b55cf4ab8c197046c87e6243ca1d67d7d3cb30280ba29364b28a7d`。
- `assets/yoloe-materials-visual.pt`：缓存视觉提示的派生权重，本机 SHA256 为 `671411af722566cdd57e93ed2623f01a8c275bbd34719aeb16d0aa424732f120`。
- `assets/visual_references.json`：24 张参考图片及其哈希、框、类别、原训练组身份；不包含验证/封存测试图。
- `preview/`：逐次上传图片、结果 JSON 和轮廓覆盖图；每次单独建目录。
- `examples/`：本机八张训练组示例，全部排除参考图本身；图片不随 Git 同步。

官方资产可以从公共地址重取。本机派生视觉模型、参考图和工位照片没有共享下载地址，队友不能仅通过克隆获得相同的参考图部署。

## 安装及启动

在仓库根目录运行，首次会下载独立 CPU 依赖、约 60 MB 模型和约 600 MB 编码器：

```powershell
.\tools\setup_yoloe.ps1 -StartPreview
```

已安装后，直接启动，不必重复安装依赖：

```powershell
.\local-only\yoloe-deployment\venv\Scripts\python.exe pynq_km1_vision/src/yoloe_preview_server.py
```

打开 `http://127.0.0.1:8778`，选择图片或点击工位样例，切换提示方式后识别。支持单张 JPEG/PNG/WebP/BMP，最大 1920×1080；不启动相机、访问 PYNQ 或发送机械命令。服务仅监听本机回环地址；当前 TACO 8777 端口保持独立。

准备自己的视觉参考时，创建 JSON 数组，每项包含 `image`（仓库根目录相对路径，必须在 local-only 中）、`image_sha256`、`decoded_bgr_sha256`（解码后 uint8 BGR 像素字节哈希）、`bbox_xyxy`（原图像素，右下边界不包含）、`category`（plastic/metal/paper）、`title`、`source_split`（train）。3 至 30 项，三类都需有参考；修改参考集应归档旧视觉部署后另建，不能静默覆盖。

```powershell
.\tools\setup_yoloe.ps1 -References local-only/yoloe-deployment/references-to-prepare.json
```

上述参考清单是本机产物，其他电脑必须使用自己的真实参考数据，不应填写占位图片或封存测试图。推理服务只使用已经缓存的提示权重，不调用文字编码器或在线接口。已通过禁止出站 socket 连接的加载和推理检查。

单图命令在工程目录运行；输出目录须尚不存在：

```powershell
..\local-only\yoloe-deployment\venv\Scripts\python.exe -m src.yoloe_inference --image <实际图片路径> --output-dir ../local-only/yoloe-deployment/my-check-001 --prompt-mode visual
```

## 独立观察接口

- `GET /health`：就绪状态、三类、文字提示、可用模式、默认模式、各模式模型哈希、参考数量、当前默认模式 runtime_id。
- `GET /examples` 和返回的样例 URL：读取本机已登记的样例，不提供任意路径访问。
- `POST /detect`：JSON `{ "image_base64": "...", "prompt_mode": "visual" }`；mode 可为 text/visual，省略则使用默认模式。
- `GET /latest-result`、`GET /overlay.jpg`：最后一次成功识别的结果和覆盖图。

输出 schema 为 `yoloe-material-observation-v1`。类别顺序固定为 **0=plastic/塑料、1=metal/金属、2=paper/纸质**；这是新的观察接口，与旧训练标签 0=can 的编号不同，调用方必须按新 schema 和字符串 label 处理，不直接套用旧箱号。

每实例保留 `prompt_id`、`prompt_key`、具体提示/参考描述、材质类别、模型得分、检测框、掩码外包框、面积和 `mask_rle`。掩码为**原图坐标、全图行优先、从背景段开始的 0/1 交替游程**；包围框右下边界不包含，面积单位 pixel。启用 `retina_masks`，尺寸不一致则拒绝，不拉伸 letterbox 掩码；使用跨提示类别 NMS，IoU=0.5。输入 640×480 时已验证与现有 CPU 掩码契约一致；更大图片仅用于此预览，不因此扩大板端契约。

配置置信度为 0.3，仅是本次 YOLOE 预览工作点，不更改原正式模型的 0.8/90% 验收规则。`model_version` 和 `runtime_id` 随模式区分；输入恰好与某张参考图解码像素一致时 `reference_image_match=true`，页面明确不计独立识别验证。

始终返回 `automatic_motion_allowed=false`、`quality_gate_passed=false`、`material_identity_verified=false`；每实例 `sorting_bin=null`。三类识别已经实现，物理箱位、旧机械路径和 commissioned 配置没有绑定或变更。

## 本机试跑及实际限制

所有探索只使用原训练组。先保留七提示文字基线，再比较三组文字提示、单示例和多示例视觉提示；最终页面支持文字基线及 24 示例视觉模式。提示选择使用了训练组观测，不能将后续试跑称为独立质量评估。

在同一组 **18 张非参考图**上做探索性检查，包含每件实物的源区和箱内各一张、一个空景和一个双物体场景。以已审核实例轮廓为真值，按同类掩码 IoU≥0.5 一对一匹配，置信度固定 0.3：

| 模式 | 塑料 TP/FP/FN | 金属 TP/FP/FN | 纸质 TP/FP/FN |
|---|---|---|---|
| 文字基线 | 0 / 0 / 7 | 2 / 0 / 5 | 0 / 2 / 4 |
| 24 示例视觉提示 | 4 / 6 / 3 | 6 / 0 / 1 | 4 / 2 / 0 |

视觉提示在这些熟悉样件的非参考画面上有所改善，但仍有塑料错分、漏检和把箱体识别为物品的问题；不能据此认定超过已训练模型或可接管自动分拣。单张本机 CPU 约 0.31～0.68 秒（含首轮开销），不是独立性能验收。证据在 `local-only/yoloe-deployment/exploratory-20261029-112039/`。18 张均来自原训练组，未评分验证集或封存测试集。

八张样例另外通过离线加载/推理，9 个输出实例的 RLE、面积和外包框与像素逐项一致；浏览器已实际点击样例并显示轮廓、三类计数和得分。模型得分不等于识别准确率。

验证：YOLOE 独立环境运行 `python -m unittest sim.test_yoloe_inference -v`，9 项通过；原 `.venv` 工程模拟回归 272 项中 266 通过、6 跳过（4 项新预览 HTTP 检查缺可选 Pillow，已在独立环境通过；另 2 项为既有 ONNX/CNN 可选依赖）；依赖一致性、仓库轻量检查及差异空白检查通过。

本次没有训练新网络、调整封存测试、执行 GPU 性能测试、连接板端/串口、运行综合、烧写或驱动机械臂。当前本轮无 PR；保留既有未提交工作，新增实现位于 `feat/yoloe-three-materials` 分支。
