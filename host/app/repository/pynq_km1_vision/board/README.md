# 双核 overlay 驱动与验收入口

按 [构建与板端验收](readme_board_test.md) 生成同一次构建的 `vision.bit` + `vision.hwh`，内含 **mask_geometry_accel** 和 **visual_evidence_accel** 两个 HLS 核。位流位于构建输出目录，不随 Git 分发；本目录中的旧本机配对不自动代表新源码。

## 内容
| 文件 | 说明 |
|---|---|
| `km1_dma_runtime.py` | 双核共用会话、限时等待、复位与缓冲管理 |
| `dual_core_acceptance.py` / `dual_core_fault_test.py` | 逐像素与统计量对比、短包超时及恢复验收 |
| `vision.bit` / `vision.hwh` | 双核比特流 + 硬件描述（mask_0=0x40000000, visual_0=0x40010000, DMA 0x41E0/1/2_0000）|
| `km1_mask_overlay.py` + `km1_mask_selftest.py` | mask 核驱动 + 板上自测（几何矩/包围框/空图/单点）|
| `km1_visual_overlay.py` + `km1_visual_selftest.py` | visual 核驱动 + 板上自测（交并/消失/新增/灰度差分/无变化）|

板卡地址、登录方式和 PYNQ 环境以当前机器配置为准。将配对产物与上述 Python 文件复制到板端独立目录后按验收说明运行。不要覆盖其他任务的工程或分别重复加载两份 overlay 后交替调用。
