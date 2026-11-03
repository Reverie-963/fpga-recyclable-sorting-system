# YOLO11s-seg 本地三类模型：ROCm 实时监控

2026-10-29，按队长要求关闭YOLOE，将已完成本地训练和121张验证的YOLO11s-seg接入原8778实时页面。现有8780相机只读桥接保留；监控不驱动机械臂，不加载新位流，不重新训练。

## 当前运行内容

- 模型为固定16件对照campaign中已选出的YOLO11s-seg `training/weights/best.pt`，不是官方80类通用权重，也不是YOLOE。SHA-256：`d7436017eeb2aa1cc414c2afd23b666b545287fdbcd032536800f1c2998d0ba9`。
- 复制冻结到本机`local-only/yolo11s-deployment/assets/model.pt`，配套`deployment.json`保存原检查点、选模报告及数据指纹。启动核对权重哈希和实际类别顺序；没有修改训练目录中的原文件。
- 本机Radeon 890M / ROCm，默认FP16；PyTorch `2.12.0+rocm7.14.0`、HIP `7.14.60850`、Ultralytics `8.3.230`。设备不符直接拒绝，不回退CPU。沿用现有进程级MIOpen兼容设置，不升级环境。
- 固定三类，640输入、置信度0.3、NMS IoU0.7、max_det20、按类别去重、原图尺寸掩码。类别顺序严格转换：模型0=can→metal，1=plastic→plastic，2=cardboard→paper；统一观察输出仍是0=plastic、1=metal、2=paper。
- 页面显示实际模型身份和ROCm精度，隐藏YOLOE的文字/参考图选择。保留实时相机、对应帧轮廓、三类计数、暂停恢复、图片试用和下载结果。当前样例沿用已有训练组图片，仅供操作示例。

## 启动

本机`http://127.0.0.1:8778/`。根目录：

```powershell
.\tools\start_yolo11s_preview.ps1
```

需要FP32时先停止当前同端口监控，再运行：

```powershell
.\tools\start_yolo11s_preview.ps1 -Fp32
```

服务为前台进程，Ctrl+C关闭并释放GPU模型。启动前完成模型核验及GPU预热；端口采用已有排他绑定。相机桥接由原服务提供，不重复打开摄像头。当前实际YOLOE的8778进程已经退出；其源码与权重保留用于追溯，不与11s同时运行同端口。

## HTTP与调用方

复用既有相机及HTTP传输代码。`/health`、`/camera.jpg`、`/live-result`、`/detect`、`/live`、`/examples`保持原路由；本模型只接受`prompt_mode="supervised"`，不是可修改提示的模型。健康接口包含`model_family="yolo11s"`、`model_name`、`model_sha256`、`runtime_id`、`execution`和固定推理参数。

识别结果schema为`recyclable-material-observation-v1`；保留完整原图RLE掩码、字符串材质、当前GPU张量设备、模型及运行指纹、对应帧身份和年龄。始终`automatic_motion_allowed=false`、`quality_gate_passed=false`、`sorting_bin=null`。新鲜性和相机断线处理沿用[实时传输说明](yoloe_realtime_rocm.md)。

原`HTTPYoloe`物理闭环适配器针对YOLOE指纹和提示模式，不兼容本次11s服务；已增加启动时明确拒绝其他模型的检查，并保留原机械配置。后续接11s闭环须单独修改识别适配、绑定11s运行指纹和重新验证，不把切换监控当作完成闭环迁移。

## 实测与验证范围

1. 同一检查点在8张验证图做ROCm FP32/FP16对照，12个实例的类别和数量一致，掩码IoU最低0.98989，最大置信度差0.00338。不是重新评分全部验证集，也不把FP32旧成绩直接声明为FP16完整复测成绩。
2. 15.172秒实时采样取得38个不同识别结果，观察更新约2.58帧/秒，推理中位60.85ms、最大1149.04ms，最大结果年龄1360ms，0次不可用轮询。并行v8s训练和共享相机桥接仍在运行，不能将这些数据作为无竞争速度或稳定高帧率保证。
3. 实际浏览器已显示`YOLO11s-seg · 本地三类专项训练`、`ROCm GPU · FP16`及新鲜相机和对应轮廓。暂停时隐藏识别结果、相机继续刷新，恢复后显示新帧。最终页面保持实时识别开启。
4. 现场截图中细长包装被判为塑料，单帧材质尚未重新确认；这不是识别正确的验收证据。原11s塑料误报及箱内漏检仍存在，模型得分不能作为材质真值。

相关模拟测试命令（工程目录）：

```powershell
..\.venv-rocm\Scripts\python.exe -m unittest sim.test_yolo11s_monitor sim.test_yoloe_inference sim.test_yoloe_live_preview sim.test_sorting_io -q
```

50项通过，覆盖类别ID映射、旧模式拒绝、无隐式CPU回退、固定模式HTTP/实时链路、旧闭环识别器拒绝11s服务，以及已有新鲜性/暂停/并发HTTP/设备接口回归。仓库检查及Windows启动脚本语法检查另行通过。未新增训练、评分封存测试、启用机械动作、综合、烧写或测试新硬件。

证据仅在本机`local-only/yolo11s-deployment/`：`assets/deployment.json`、`smoke/{fp32,fp16,equivalence}.json`、`live-proof/{health,last-result,measurement,samples}.json`、对应轮廓和`monitor.jpg`。训练权重与证据不随Git同步，无共享下载地址；按[资产领取说明](../../docs/team/ASSETS.md)向队长领取配套权重和锁文件。当前无提交、推送或新PR。
