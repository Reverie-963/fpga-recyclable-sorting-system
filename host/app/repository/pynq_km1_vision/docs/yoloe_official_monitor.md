# 官方 YOLOE 通用识别监视器

2026-10-30：独立入口 **http://127.0.0.1:8811/**，用于查看官方通用模型效果。默认使用现有相机桥接的画面，运行无提示模式，不绑定三类材质映射。

## 使用

1. **无提示**：选择“无提示 · 自动发现物品”，点击“应用设置”。专用 `yoloe-11m-seg-pf.pt` 使用其实际内置的 4,585 个名称；左侧可展开、搜索完整词表。
2. **文字提示**：选择“文字提示”，输入英文物品名称，用逗号或换行分隔，点击“应用设置”。例如 `person, bottle, cup, box, can, chair`。界面允许 1–80 类；原始提示模型为 `yoloe-11m-seg.pt`。
3. **图片提示**：选择“图片提示”，上传参考图或从相机截取参考图，在图上拖动框选一个物品，然后应用。查找外观相似物品，结果名称为 `object0`，不自动赋予材质或产品身份。
4. **图片试用**：上传待识别图片、点击“官方样例”或“相机截图”，再点击“识别当前图片”。可以对同一张图切换模式比较。页面把图片缩放到 1920×1080 范围，JSON 可下载。
5. “实时监视器”显示当前相机及被识别帧的覆盖图、物体数量、推理耗时、结果更新率和帧龄。暂停识别时相机继续刷新；静态图片切换会暂停本监视器的实时推理。

阈值默认 0.25，可调 0.01–0.95。模型得分不是识别准确率。无提示结果可能包含场景或宽泛概念，并存在误检、漏检；内置词表规模不等于已验证类别数量。

## 启动、资产和环境

根目录执行：

```powershell
py -3.11 tools/prepare_yoloe_official.py
./tools/start_yoloe_official.ps1
```

启动脚本默认 ROCm GPU；可显式 `-Backend cpu` 或 `-Port 8811`。复用已有 `.venv-rocm`（Python 3.12.10、PyTorch 2.12.0+rocm7.14.0、Ultralytics 8.3.230）检测；文字提示通过已有 `local-only/yoloe-deployment/venv`（Python 3.11、PyTorch 2.6.0+cpu、Ultralytics 8.3.230）独立进程生成嵌入后传入 GPU。CPU 运行环境准备见 [已有 YOLOE 部署](yoloe_materials_deployment.md)。首次新词表编码需要等待，后续按词表内容缓存；没有训练或微调。

该 ROCm 环境直接执行 CPU TorchScript 编码曾触发 OpenBLAS 分配失败及原生异常，故保留独立 CPU 编码路径。不能以简单重试替代兼容性处理。GPU 进程沿用已有 `cudnn.enabled=False` 兼容方式，仅影响本进程。运行目录设为已有官方资产目录，使文字编码器从本机文件加载。

官方权重来自 [Ultralytics assets v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0)，使用和模式说明见 [官方文档](https://docs.ultralytics.com/models/yoloe/)。

| 资产 | SHA-256 |
|---|---|
| `yoloe-11m-seg-pf.pt`，63,552,937 字节 | `0e31ac208a31b591c20c69f6e525110cc9e96a7dbc1dc2e441ae46597205bbfd` |
| `yoloe-11m-seg.pt` | `785199b7cc75a5f4041dec15782a4b8a4105e5a34d7ffb287a0a10781724c5bd` |
| `mobileclip_blt.ts` | `a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54` |

下载脚本核验哈希，已有不同文件时拒绝覆盖。新无提示权重、下载记录、文字缓存和日志在根目录 `local-only/yoloe-official/`；原始提示模型与编码器复用已有 `local-only/yoloe-deployment/assets/`，不修改原文件。图片和运行产物仅本机，不随 Git 同步；官方资产可由脚本重取。

## 接口

仅监听回环地址，要求同源 Host/Origin。独立协议，不供现役分拣控制调用。

| 接口 | 含义 |
|---|---|
| `GET /health` | 当前模式、阈值、GPU 环境、哈希、相机状态、错误、更新率与配置代次 |
| `GET /vocabulary` | 当前无提示权重的完整词表 |
| `GET /camera.jpg` | 当前新鲜相机帧 |
| `GET /live-result` | 对应识别帧的 JSON 和 `overlay_base64`；暂停、过期或无结果返回 503 |
| `GET /example.jpg` | 官方公交车样例 |
| `POST /configure` | `mode=free/text/visual`、`confidence`、`enabled`；文字模式加 `classes`，视觉模式首次加 `reference_base64` 与 `box` |
| `POST /detect` | 当前配置下的单图识别，传 `image_base64`；或 `source=camera` 读取新鲜相机帧 |

输入参考框 `box=[x1,y1,x2,y2]` 为上传参考图原图像素；API 图片最长边不超过 2560、总像素不超过 450 万。输出 `schema_version=yoloe-official-observation-v1`，含实例名称、分数、原图像素框 `bbox_xyxy`、覆盖图、模式、权重哈希与实际环境。轮廓由官方模型原生分割结果绘制，目前不单独输出掩码数组。

相机复用本机 `http://127.0.0.1:8780/frame.jpg`，按上游流/帧 ID 去重、累加上游年龄，不刷新重复帧时间；只保留最新帧/结果，不逐帧写盘。配置切换清除旧结果，代次校验防止在途结果回流。实验监视器新鲜度为 5 秒，独立于现役控制；帧龄不是曝光时间戳。所有输出 `automatic_motion_allowed=false`。

## 本轮验证

- `py -3.11 -m unittest sim.test_yoloe_official_monitor -v`：10 项通过，0 失败、0 跳过；覆盖类别/框输入、同源、错误请求、配置失败保留、切换失效、上游帧龄、过期和暂停结果。
- 官方公交车样例按无提示→文字→参考图→文字→新词表→无提示顺序真实推理六次：无提示检测到人/公交车等；`person,bus` 检出 4 人和 1 辆公交车；人物参考图提示检出 4 个 `object0`；切换回来结果正常。仅为功能检查，没有完整真值评估。
- 已接入实际相机并获得无提示实时输出；现场部分名称有明显误判，不计入正式识别验收。
- 收尾原 8780 桥接发生线程内存异常并停止监听，所有旧监视器也断流。仅重启本机既有只读 `station_vision_bridge.py` 代理，复用原连接模块、令牌文件及板端服务；相机恢复、原上游流与帧 ID 继续递增。没有重启板端采图或改动基线/区域/位流。代理恢复日志在同一本机输出目录。
- Python/页面 JavaScript 语法、资产哈希及仓库检查通过。浏览器自动检查因工具无法确认当前 URL 而中止，页面交互未完成实际浏览器验收；入口可由用户直接打开。
- 本轮没有 FPGA 综合、位流/固件更新、串口或机械动作。原服务未替换。本轮无 PR，分支 `feat/yoloe-official-playground`，保留已有未提交修改。

本机样例输出与原始 JSON：`local-only/yoloe-official/verification/`；这是功能证据，不是准确率或独立性能报告。
