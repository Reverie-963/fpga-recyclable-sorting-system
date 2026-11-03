# 免重训新物品接入 Skill

支持“用这几个词识别新物品”或“按这张参考图识别这种物品”，输出独立类别候选包、
轮廓预览、正负样本评估报告和可复制的配置。**不修改现役模型绑定，不启动机械动作。**
免重训能力来自已有YOLOE预训练模型；Skill负责把接入与验证流程做成可复用工具。

## 已实现

- 自定义持久类别ID、文字提示，或多张参考图与框；显式映射到候选箱位/抓取配置ID。
- 文字与视觉提示分别生成冻结的`embeddings.npy`，不反向传播，不保存新的基础权重。
- 原图实例掩码、得分、包围框与PNG轮廓预览，保持640×480消费契约。
- 独立真值评估：每类TP/FP/FN、precision/recall/F1、匹配掩码IoU、正负场景数与延迟。
- 样本泄漏、重复图、无效RLE、过期包身份、错误路由引用与结果篡改检查。
- 导出/复制后验证完整性的候选包；失败保持`preview_only`，识别验证通过最多为
  `evaluated_candidate`。两者均无机械执行权限，也不代表物理验收。
- 无第三方包的完整模拟流程，用于检查包、评估、导出链路。

## 环境

核心合同、回放、导出及测试使用Python3.11+标准库。真实YOLOE步骤复用已安装的独立环境：
Ultralytics8.3.230、PyTorch2.6.0+cpu、对应NumPy/OpenCV及文字模式所需CLIP分词器。
当前只开放已验证的CPU/FP32路径，不自动使用正在服务中的GPU。

准备可信的本机模型文件及runtime配置，格式见
[runtime示例](examples/runtime.example.json)。用实际本机路径替换示例路径，保留对应文件的
真实哈希。runtime文件属于本机配置，不放入共享包。文字构建还需要已下载的
`mobileclip_blt.ts`；视觉模式及加载已有提示缓存时不需要重新调用文字编码器。
工具不下载、不安装或升级包，进程内所有出站网络操作被拒绝。

## 快速试用：无需模型

在本目录执行，每个输出目录必须尚不存在：

```text
python scripts/add_sort_category.py demo --output category-demo
python scripts/add_sort_category.py verify --package category-demo/portable-preview
python scripts/add_sort_category.py replay --package category-demo/candidate --recording category-demo/recording.json --output category-replay
python -m unittest discover -s tests -p "test*.py" -v
```

demo故意包含一个负样本误报：TP=2、FP=1、FN=0，precision=2/3；质量门槛不通过，
仍可导出预览包。模拟时间不是性能成绩。成功完成评估但质量未达标时退出0，报告门槛为false；
无效输入/运行错误退出1，参数错误退出2。评估完成和质量通过是两个不同状态。

## 真实文字提示

以下命令中的`python`应换成已核验的独立ML环境解释器。不要为本技能直接升级原有环境。

```text
python scripts/add_sort_category.py new --package-id cup-v1 --category-id plastic-cup --name 塑料杯 --prompt "plastic cup" --bin-id plastic --grasp-profile existing-cup-profile --output cup-spec.json
python scripts/add_sort_category.py build --spec cup-spec.json --runtime runtime.local.json --output cup-candidate
python scripts/add_sort_category.py preview --package cup-candidate --runtime runtime.local.json --image another-scene.png --output cup-preview
```

`new`中的箱位和抓取配置是用户提供的既有ID，工具只检查其在声明清单中的引用，
不验证实机标定。生成的默认评估策略是团队建议，可在构建前依据任务冻结；不能在
看过验收结果后静默修改阈值。多类别配置参考[文字示例](examples/text-spec.json)。

## 真实参考图提示

```text
python scripts/add_sort_category.py new --package-id part-v1 --category-id sample-part --name 示例零件 --reference reference.png --bbox 100 80 220 240 --reference-group object-reference-a --bin-id plastic --grasp-profile existing-part-profile --output part-spec.json
python scripts/add_sort_category.py build --spec part-spec.json --runtime runtime.local.json --output part-candidate
python scripts/add_sort_category.py preview --package part-candidate --runtime runtime.local.json --image different-scene.png --output part-preview
```

框坐标为原图像素，右下界不包含。参考图随候选包复制并记录文件/解码像素哈希，
原图不改。多参考图分别保留提示索引后映射类别；不把`object0`直接当作材料或箱号。
预览参考图本身会标记`reference_image_match=true`，不能计为独立验证。

## 评估与导出

按照[合同说明](references/contracts.md)准备dataset JSON，显式给出分组、真值与正负场景：

```text
python scripts/add_sort_category.py evaluate --package cup-candidate --runtime runtime.local.json --dataset validation.json --output cup-evaluation
python scripts/add_sort_category.py export --package cup-candidate --evaluation cup-evaluation --output cup-portable
python scripts/add_sort_category.py verify --package cup-portable
```

`evaluate`复用同一已加载模型逐图推理，不包含加载和提示生成耗时；返回的是本机推理及
实例转换时间，不是整机任务时延。原图不隐式缩放，超过640×480直接拒绝。
验证集不得与参考图共享文件、解码图像或声明分组，不接受test/sealed split。
开发集可以检查熟悉样件流程，但永远不能打开质量门槛。分组和标注真实性仍需人工核验。

导出包保留原候选、提示缓存和引用图片，附评估证据及候选路由；不含约60MB基础权重、
约600MB文字编码器、环境、端点或机器runtime。另一台机器提供同哈希基础权重及兼容环境后，
可验证、预览和重新评估。选择旧目录可回看旧配置；不会自动切换生产执行器。

## 失效条件与真实边界

依赖/权重/编码器缺失、哈希或版本不符、无效框/类别/路由、图像过大等均停止，不自动补装。
验证图重复、参考组泄漏、test数据或错误掩码不被接受；不足样本/误检漏检保留在报告，
不为了通过门槛改变事实。网络探测/下载被阻断，不能将已有缓存运行称为冷环境安装成功。

基础模型预训练可能已经包含该物品；“免本队重训”不是“模型从未见过”。视觉外观不能保证
材料成分或回收政策，新样件尺寸/高度/重量/夹持/箱内可见性仍需独立验证。
本工具没有生产部署和机械接口，识别门槛通过也不产生commissioned或动作许可。

本轮验证及来源见[验证快照](references/validation.md)。官方方法说明：
[Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/)。实际实现以已安装8.3.230源码
的get_text_pe/YOLOEVPSegPredictor接口为准，不照搬新版本API或自动更新环境。
