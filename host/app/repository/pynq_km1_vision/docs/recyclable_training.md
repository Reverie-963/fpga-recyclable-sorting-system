# 三类实例分割：先采小批数据再训练

后续已存在本机三轮试训及确认的 `can / plastic / cardboard` 材质类别，原文以下保留为启动时记录。继续扩训与网上素材收集见[数据扩充入口](recyclable_data_expansion.md)；具体结果以对应日期证据为准，不把启动阶段状态当作本机最新模型状态。

2026-10-27：用户要求跳过重复环境验证，直接推进标定、采集和训练。当前按可回收物方案的 `can`、`plastic_bottle`、`cardboard` 三类实施。保留旧瓶盖检测训练入口；新增 `src.train_segment` 专用于实例轮廓。

## 本轮实际进度

用户确认暂时没有数据；标定板等待用户完成另一项工作后摆放。本机已有相机到标定卡平面的历史记录，但机械臂坐标方向、原点、抓取高度与独立误差没有完成验收；旧卡平面拟合值不能直接作为机器人精度。

在现有 ROCm 环境启动 YOLO11n-seg 训练时，MIOpen BatchNorm 的运行时编译报 `type_traits file not found` / `miopenStatusUnknownError`。使用进程内 `torch.backends.cudnn.enabled=False` 后，GPU 原生卷积/BatchNorm 前后向通过；随后完成官方 COCO8-seg 的 1 个 epoch（4 张训练、4 张验证，160 输入、batch=1、FP32），并重载保存权重对本机旧现场图片完成推理调用。实际设备 `cuda:0` 是 ROCm 的 Radeon 890M 接口名称。

这仅说明小规模分割训练链路可运行，不代表三类可回收物识别、640 分辨率长时训练、NPU、正式精度或实时性能通过。原生 GPU 路径的速度需要实际测量。此次没有升级驱动、覆盖环境或修改包源码。失败日志、成功日志和权重在本机 `local-only/recyclable-start-20261027/`；公开样例数据下载在本机 `C:/datasets/coco8-seg`，不随 Git 同步，无团队共享链接。

## 标定板与首批采集

沿用用户已有标定板，先放在机械臂正前方待抓区域的桌面上，图案朝上，完整可见。保持相机、底座固定；有 TOP 箭头时朝远离底座方向，中心对齐底座前后中心线。放好后采新图并检查图案类型、打印尺度、清晰度和覆盖范围，再求相机到板平面的映射。随后单独建立到机械臂的坐标关系及抓取高度映射，不凭估计距离自动写入生产配置。用户当前暂缓摆板，本轮没有采新图或执行标定运动。

不等待数千张数据才开始。首批可用每类两个不同实物：第一批实物在训练场次改变位置、朝向拍摄；第二批不同实物在另一场次拍验证图。起步可每类约 20 张训练、10 张验证，并加少量空场景。数量是调试批次建议，不是精度保证。单个实物和同一场次只能进入一个集合；仅有一个实物时可以采训练图，但还缺独立验证样件。先检查少量标注和训练结果，再扩大样本、光照和材质覆盖。

每件可见物品标真实类别与实例轮廓，不能用矩形检测框充当分割标签；阴影和机械臂不是物品。空场景显式保存空标签文件。人工复核自动辅助轮廓后才用于训练。测试数据使用另外的实物与场次，训练调参阶段不评估封存测试集。

## 数据目录与分组清单

数据放在本机忽略目录，例如根仓库 `local-only/recyclables-v1/`：

```text
images/train/*.jpg
images/val/*.jpg
images/test/*.jpg       # 可在正式验收前补充
labels/train/*.txt
labels/val/*.txt
labels/test/*.txt
split_manifest.json
```

同名 `.txt` 每行一个实例：`class_id x1 y1 x2 y2 x3 y3 ...`。顶点相对于原图宽高归一化到 0～1，至少三个顶点；类别 0=can、1=plastic_bottle、2=cardboard。示例结构不是真实标注，不直接复制到数据集。

`split_manifest.json` 的 `images` 数组必须逐张覆盖所有数据，以下仅示意字段：

```json
{"images":[
  {"image":"images/train/can001_pose01.jpg","split":"train","session_id":"capture01","object_ids":["can001"]},
  {"image":"images/val/can002_pose01.jpg","split":"val","session_id":"capture02","object_ids":["can002"]}
]}
```

多物体场景列出所有实物编号；空场景使用 `object_ids: []`。训练和验证必须各有三类有效轮廓，不能复制同张图补齐集合。程序检查相同图片哈希、实物编号及场次跨集合泄漏，拒绝缺标签、缺类别、非有限/越界坐标、退化轮廓和清单遗漏。检查不代替人工标注质量复核。

## 启动训练

在 `pynq_km1_vision/` 执行，复用现有 `.venv-rocm`：

```powershell
..\.venv-rocm\Scripts\python.exe -m src.train_segment --dataset ../local-only/recyclables-v1 --check-only
..\.venv-rocm\Scripts\python.exe -m src.train_segment --dataset ../local-only/recyclables-v1 --weights yolo11n-seg.pt --device 0 --epochs 30 --batch 2 --imgsz 640 --native-backend
```

没有标注图像时，检查会报 `No annotated images` 并在加载模型、下载或训练前停止。先采集标注再执行第二行。首次预训练权重会从 Ultralytics 官方发布源下载；离线使用前保留本机权重。`--native-backend` 只影响本进程的 GPU 算子后端选择，不会改为 CPU，不改变系统或其他进程设置。若其他电脑不需要该兼容选项，可省略。

每次训练在独立 `runs/segment/recyclables-<时间>/` 保存配置、分组数据摘要与哈希、`weights/best.pt`、`weights/last.pt` 和 `model.json`，拒绝覆盖同名运行目录；不覆盖旧瓶盖模型。默认不评估测试集。确定训练方案与阈值后，最终一轮训练可加 `--evaluate-test`，在完成验证集选模之后评估封存测试集；需要三类独立测试样本，结果不能再用于调参。

该入口产出 PyTorch 分割权重；ONNX 导出、正式掩码服务、板端调用和抓取集成另行实现，不把旧 v2 检测服务视为已支持新掩码。当前不启用机械动作或修改 `commissioned`。

依据：[Ultralytics 实例分割](https://docs.ultralytics.com/tasks/segment/)、[AMD MIOpen 的 PyTorch 原生 BatchNorm 说明](https://rocm.docs.amd.com/projects/MIOpen/en/docs-7.0.0/how-to/use-nhwc-batchnorm-in-pytorch.html)。本次按已安装 Ultralytics 8.3.230 源码和实测使用 YOLO11n-seg，不随在线文档升级版本。
