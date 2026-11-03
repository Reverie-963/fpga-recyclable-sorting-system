# Model identity

现役模型标识yoloe26_reference，是YOLOE-11m的26细类适配与参考图辅助配置；“26”是类别配置标识，不应据此写成另一种已部署基础网络。权重来自10月31日公开素材续训best；类别辅助材质结果与网络原始类别分开保存。

- yoloe26_reference.pt SHA256: 9d5734b34871aa81412b2fa47176d8f926283bb5adfe3a3f80cc58c5b7332534
- reference_bank.json SHA256: 9b6134e32d5c44b3398cdb08c74cdfd915cd46d013d43b9b1ea13920645f5fe8
- upstream configuration SHA256: cb288271841ff81c2dcd44ea6fc9742231e658d0a68f44caa0a970aee15b792e
- batch=1, ROCm GPU FP16, confidence=0.10, input640×480；原始类别共26项，参考库为已知16件物品的64图特征。

模型权重及参考特征库在本机工程包提供，不上传GitHub。没有可替代本权重哈希的公开下载地址，不把官方基础权重下载入口冒充当前微调权重。原训练图全集、Web下载缓存不随包；training_audit.json为已有训练变更证据，不等于本包可零成本重训。

现役运行库在host/runtime/ultralytics_runtime_8_4_193.zip（8.4.193）；Ultralytics原始许可证见third_party/ultralytics_license.txt。模型来源和数据构成应结合其原始许可核对；没有把第三方素材改为本队MIT授权。原始TACO标签映射提交70479ad9adc3da971a8ae75ac97eb1026910eec7，项目源码已保留来源链接。

效果93.68046% Macro-F1、82.25439%前景mIoU属于128张已知物品旧图加3空景的三材质参考辅助复算；不是26细类独立泛化或机械成功率。不得把GPU性能日志与历史CPU保存掩码复算混成一次统一精度实验。


运行库补充：`host/runtime/ultralytics_runtime_8_4_193.zip`保留上游原始文件名，先用`python host/deploy/prepare_runtime.py --output <new_runtime_dir>`解压到新的本地目录，再将该目录加入PYTHONPATH。源目录中的README.md/SKILL.md保留通行格式（官方推荐目录也使用README.md），其他普通目录和文件统一小写；第三方归档内部保留其原始大小写。
