# 输入、状态与证据合同

## 类别spec

字段以`examples/text-spec.json`为完整可解析示例。一个包采用一种提示模式；需要比较
文字/视觉模式时分别建包，不静默混用。`categories`最多20类，文字提示合计最多30个，
视觉参考最多30张。每类均有持久`category_id`及用户声明的bin/grasp引用。

视觉spec用`mode:"visual"`，类别删除`prompts`，并设置references数组；每项要求：

```json
{
  "reference_id":"example-a", "category_id":"plastic-cup",
  "image":"reference.png", "image_sha256":"真实文件的64位小写SHA256",
  "group_id":"physical-object-session-a", "split":"reference",
  "bbox_xyxy":[100,80,220,240]
}
```

以上哈希文字是格式说明，不是可运行哈希；`new --reference`自动算真实文件摘要。
源路径相对spec文件；构建时复制到包内，包中只保留相对路径，补解码BGR哈希与尺寸。
`split`只允许reference/development。每类都必须有参考图。

## runtime

`examples/runtime.example.json`给出已验证版本及官方现有模型哈希。路径相对runtime文件；
也可在本机配置使用绝对路径，但该文件不导出。当前锁定CPU FP32、1–4线程、
Ultralytics8.3.230及PyTorch2.6.0+cpu。不符合时拒绝，不升级、不回退、不改GPU服务。
基础`.pt`必须是可信本地文件；提示缓存用非pickle的NumPy文件，读取禁止pickle。

## Dataset

顶层：`schema:"category-dataset-v1"`、`purpose:"development"|"validation"`、`images:[...]`。
每项是：

```json
{
  "image":"scenes/scene-a.png", "image_sha256":"真实文件的64位小写SHA256",
  "group_id":"independent-object-session-b", "split":"validation",
  "truth":[{"category_id":"plastic-cup","mask_rle":[0,10,307190]}]
}
```

真值RLE按实际图像尺寸生成；此处307200总像素仅展示640×480的格式。负样本用`truth:[]`，
不是省略标注。每类分别统计“有该类真值”的正场景和“无该类真值”的负场景。
一个图像可含多类；其他类别的存在不代表该类不存在的真值已经自动审核。
上限1000张；测试/封存split不支持。文件哈希与解码像素哈希都唯一，分组应按实物/采集
会话设计，不能随意为连续帧换名字冒充独立场景。

匹配先按持久类别分开，以冻结mask IoU阈值建立候选边，再做最大数量一对一匹配。
低于冻结置信度的预测不计检测。输出TP/FP/FN、precision/recall/F1及已匹配实例平均IoU；
这不是COCO mAP。无真阳性/无可用分母时precision或recall为0，不自动判通过。

## 包与状态

`package.json`保存内容指纹、资产哈希、基础权重前后哈希、prompt_map、运行版本及来源；
`spec.json`保存冻结定义；`embeddings.npy`仅为提示向量。视觉参考存在references/。
本队没有基础权重反向传播，未使用任何train/fit入口，基础权重文件前后哈希必须一致。
哈希不能独立证明任意第三方生成的包没有训练，只支持本工具实际路径的证据。

`preview`返回`image_size=[宽,高]`、原图RLE、像素面积和不含右下界bbox；不是COCO压缩RLE。
RLE从背景段开始、行优先、总和等于宽×高，首像素前景时首段为0。
保留原始prompt索引和显式category_id；`proposed_bin_id`是候选去向，`sorting_bin=null`。

state始终preview。evaluation可通过识别质量门槛，但release仅增加
`recognition_state:evaluated_candidate`，不会改package的状态、生产模型绑定或机械权限。
完整链路始终`automatic_motion_allowed=false`、`physical_acceptance_verified=false`。
旧包不覆盖，输出要求新目录。export核对包与评估身份、原始观测摘要并重新计算关键指标。
这些是完整性保护，不是签名、防伪或第三方独立审计。

## 回放与故障

回放记录：`schema`同技能、`package_sha256`、purpose、rows数组。每行需要图像文件/解码
哈希、group_id、split、image_size、truth、predictions及inference_ms；预测含category_id、
confidence、mask_rle。回放来源固定为recorded或synthetic，永远不能打开识别质量门槛。

无效输入/依赖异常在新建输出目录写failure.json；评估中途失败可保留partial-observations.json，
不生成“通过”结论。旧目录拒绝使用，也不会向旧证据目录添加失败文件。候选包不含设备连接、
相机/串口/执行API；动作预检、限位、STOP/pending/所有权等由现役系统保持。
