指标见metrics.csv，采用官方七列表头。日志中的历史状态仅是当时快照，本次归档未访问设备或重新确认服务状态。证据副本只作必要的换行/隐私规范化，来源原哈希见provenance/source_manifest.json；不得将被清理的副本称为原文件逐字节未变。原始波形未新增，已补30次试测及28次实际动作条件首动分位数，见../docs/pilot30_update.md；总体尾部缺口仍保留。


运行库补充：`host/runtime/ultralytics_runtime_8_4_193.zip`保留上游原始文件名，先用`python host/deploy/prepare_runtime.py --output <new_runtime_dir>`解压到新的本地目录，再将该目录加入PYTHONPATH。源目录中的README.md/SKILL.md保留通行格式（官方推荐目录也使用README.md），其他普通目录和文件统一小写；第三方归档内部保留其原始大小写。


精度来源补充：93.68046%/82.25439%来自已保存的GPU FP16首轮旧工位输出，其逐图记录、TP/FP/FN及像素分量见quality_gpu_derived.json，131张原测试图/标签另附test_data，不是训练集。更早CPU FP32审计（mIoU约82.19%）作为历史质量记录单独保留，不能替代本组。
