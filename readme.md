# 基于FPGA的可回收物智能分拣系统

本仓库为独立发布的私有代码与测试证据快照。完整代码包下载：[https://github.com/Reverie-963/fpga-recyclable-sorting-system/releases/tag/code-20261103-v1](https://github.com/Reverie-963/fpga-recyclable-sorting-system/releases/tag/code-20261103-v1)。模型权重、配对位流、检查点及第三方归档存于Release附件，逐文件身份见 `provenance/release_assets.json`。下载并解压完整包后再运行下文的完整包校验；Git源码树中的原始包清单包含Release资产，不能在缺失资产时声称全包校验通过。

AMD具身智能选题（指南3.2）初赛工程快照，基础工程快照日期2026-11-02；30次试测证据更新日期2026-11-03（v2）。

这是现有实现和已留存证据的提交整理包。演示视频、两份最终技术报告、海报、作品全貌图、开发板照片与签署扫描件由队伍继续准备，尚未混入旧稿。不能把本包解释为总体成功率、物理响应P99或干净硬件环境已完成验收。

新增30次结果：目标箱接纳25/30（83.33%，测试后补充允许已记录搭边），原严格17/30（56.67%）；28次实际动作条件首动P99 29.987秒。详见`docs/pilot30_update.md`。旧11次探索组保持独立。

指标证据完整性见 `docs/metric_evidence_audit.txt`，逐项映射见 `docs/metric_evidence_audit.csv`。已附复算脚本不代表每项均具备独立现场重测条件。

## 从哪里看

1. `docs/architecture.md`：现役分工与数据流。
2. `docs/interface.md`：通信、帧身份、单位及异常边界。
3. `docs/hardware_setup.md`：器材/接线/供电核对项。
4. `docs/reproduction.md`：离线核对、环境、模型、FPGA重建和监督实机步骤。
5. `metrics/metrics.csv`：官方七列表头，五类核心与两类专项；待测项明确留空。
6. `metrics/logs/`与`metrics/evidence/`：逐请求、逐任务、原报告与照片证据，保留失败及拒绝。
7. `host/model/model.md`：当前本地权重/参考库/配置身份。
8. `docs/submission_gaps.md`：当前缺口，提交前须据实处理。

## 对照官方推荐目录

`fpga/src/`为两颗现役HLS核源码；`fpga/build/`为原联合BD脚本、IP归档、可复核配对bit/HWH与最终检查点；`fpga/report/`为资源、时序、DRC及HLS结果；`fpga/ps_runtime/`为板端冻结实现的只读源码/配置参考副本。PS快照依赖原冻结目录与实测标定，不能直接复制后启动。

`host/app/repository/`保留现有PC/工具代码目录结构（包括历史入口，不表示同时启用）；`host/runtime/`包含测量时实际导入的Ultralytics 8.4.193；`host/deploy/`包含依赖和环境说明；`host/model/`包含当前56 MB权重与参考库，不含训练图像全集。

`skill/`包含两项可复用能力及既有验证结果，未把同工位回放提升为跨板迁移验收。`report/engineering_report.docx`已归档用户提供技术报告的IEEE引用修订版，其余正文保持；门户最终PDF仍待准备。

## 最小离线验证

在本目录执行：

```
python metrics/scripts/recompute_metrics.py
python metrics/scripts/recompute_pilot30.py
python tools/verify_package.py
```

上面仅核对归档文件和已有数据，不连接板卡、不打开串口、不加载模型。实际输出见`provenance/validation/`。

完整构建/新机环境部署/新机位实抓此次未重跑。源工作区包含未提交代码，Git main与此本机快照不等价。来源快照及各文件校验见`provenance/`。不在包内放置真实令牌、密码、SSH连接模块、设备占用/STOP/pending或旧任务恢复环境。


运行库补充：`host/runtime/ultralytics_runtime_8_4_193.zip`保留上游原始文件名，先用`python host/deploy/prepare_runtime.py --output <new_runtime_dir>`解压到新的本地目录，再将该目录加入PYTHONPATH。源目录中的README.md/SKILL.md保留通行格式（官方推荐目录也使用README.md），其他普通目录和文件统一小写；第三方归档内部保留其原始大小写。
