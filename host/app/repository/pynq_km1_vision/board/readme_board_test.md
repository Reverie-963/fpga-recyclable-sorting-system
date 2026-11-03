# 双核构建与板端验收

`mask_geometry_accel` 负责掩码清理和整数几何量，`visual_evidence_accel` 负责前后帧差分及掩码交并。两核不执行机械动作，也不替代训练、相机标定或分拣成功率验收。

## 在 Windows 构建

在仓库根目录，用本机工具路径替换参数：

```powershell
./tools/build_dual_core.ps1 -ToolRoot $env:KM1_TOOL_ROOT -OutputDir C:/fpga-build/dual-core
```

使用新建的英文无空格输出目录。脚本依次运行两核 C 仿真、综合、随机背压 RTL 协同仿真、IP 打包和 Vivado 实现。只有实现时序和 DRC 检查通过才把同次 `vision.bit`、`vision.hwh` 复制到输出目录的 `board/`。报告在 `reports/`；HLS 报告在各核 `hls/syn/report/` 和 `hls/sim/report/`。

构建不使用其他人的 `prj_*` 缓存。不能仅对旧综合结果执行 IP 导出后声称验证了新源码。仅重跑协同仿真可用 `-Stage cosim`；只运行已有配对 IP 的实现可用 `-Stage vivado`。

位流不随 Git 分发。领取时索要本次构建清单和 SHA-256，核对两文件，不把旧仓库曾提交的位流当作新版本。当前团队共享下载地址未提供。

## 复制到板端独立目录

复制同次 bit/hwh，以及本目录中的：

- `km1_dma_runtime.py`、`km1_mask_overlay.py`、`km1_visual_overlay.py`
- `km1_mask_selftest.py`、`km1_visual_selftest.py`
- `dual_core_acceptance.py`、`dual_core_fault_test.py`、`dual_core_benchmark.py`

需要板端 PYNQ、NumPy、OpenCV。使用板上现有 PYNQ Python 环境和管理员权限；当前实机 PYNQ 3.1.1 的设备发现需要在该进程中先加载已有 `/etc/profile.d/xrt_setup.sh`。这不需要安装或升级软件。下面命令在该环境中执行：

```sh
python km1_mask_selftest.py
python km1_visual_selftest.py
python dual_core_acceptance.py --count 1280 --output acceptance.json
python dual_core_fault_test.py
python dual_core_benchmark.py
```

这是 FPGA 实物实验，会重载易失性 PL 配置；确保其他任务没有使用该 overlay。测试只回放数组，不访问相机、串口、机械臂或启动存储介质。

验收逐像素比较两核与独立 OpenCV 参考，几何核同时比较全部 6 个原始矩和 bbox；覆盖 63/64/65 像素、非方形、单行/单列、满帧、边界、空图、非二值像素、形态学四种开关、阈值边界和连续交替任务。原始 JSON 保留所有成功案例和首次失败，不隐藏失败。数据是确定性合成数据，不是三类真实样本泛化成绩。

故障测试故意少传一个像素，要求在 0.2 秒任务期限后退出并重载位流；复位总耗时另行记录，之后两核都必须重新通过。复位失败时缓冲被隔离，不能继续使用该会话。

## 软件接入

```python
import numpy as np
from km1_dma_runtime import DualCoreSession
from km1_mask_overlay import MaskAccel
from km1_visual_overlay import VisualAccel

owner = DualCoreSession('vision.bit')
mask = MaskAccel(session=owner)
visual = VisualAccel(session=owner)
frame = np.ones((480, 640), dtype=np.uint8)  # 高、宽
stats, cleaned = mask.run(frame, erode=1, dilate=1, job_id=1)
evidence, changed = visual.run(frame, frame, mode=1, job_id=2)
```

同一会话串行调用，返回普通 NumPy 数组。每像素传输 4 字节；26 位 DMA 长度足够容纳 640×480。驱动读取 HWH 的真实长度位数，旧 14 位硬件会提前拒绝大帧。

接口、边界和恢复约束见 [接口契约](../../docs/design/interface_contract.md)。性能须报告同一 PS 上 CPU 参考与包含分配、拷贝、DMA、寄存器读取的 PL 总耗时；只有数据支持时才声称加速。
