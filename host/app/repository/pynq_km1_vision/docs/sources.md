# 来源与实现依据

本版整理日期：2026-09-22。文档中的赛道判断依据用户提供 PDF，并不表示已向赛事方核验最新补充公告。软件平台信息同时核对了官方资料。

## 本地原始资料

均位于本项目上级目录，没有改动原文件：

- `选题指南/全国大学生嵌入式芯片与系统设计竞赛'2026FPGA赛道选题指南-AMD.pdf`：物理第 5 页介绍 PYNQ-Z2；第 6、8 页说明具身智能平台要求；第 22～25 页对应自主选题组别、工具版本、板上实测、提交物与评分。物理第 23 页已渲染核对。
- `03.焕新机械臂Km1-总线舵机-ESP32/1.教程手册/Km1机械臂使用手册 -ESP32 - ZX.pdf`：物理第 35 页说明 8V 5A 与上电顺序；第 43 页说明 USB 串口与 115200；第 96 页说明多控制源干扰排查。
- `03.焕新机械臂Km1-总线舵机-ESP32/4.源代码程序/KM1机械臂Openmv视觉模块代码 V3.1/kinematics.py`：连杆尺寸、关节角约束与 PWM 映射。本项目以 CPython 重写、取消 pyb 依赖、将运动学计算和串口发送分离，并增加失败检查。
- 同目录 `colorSort.py`：5 号夹爪命令、原始抓取/抬升/放置流程。
- `1.教程手册/视觉指令表.docx`：`#RunStop!` 属于视觉功能指令，不据此推断硬件急停。
- `4.源代码程序/机械臂控制器出厂源码/ESP代码v1.1.mix`：确认提供的是 Mixly ESP32 工程，不能从 `.mix` 扩展名直接声称已经构建烧录 Arduino 固件。

## 官方在线资料

- [PYNQ 支持板卡和镜像](https://www.pynq.io/boards.html)：Z2 镜像入口及 USB HOST 等板卡能力。
- [PYNQ-Z2 启动说明](https://pynq.readthedocs.io/en/v2.6.1/getting_started/pynq_z2_setup.html)：SD、网络与上电方法；链接为有明确版本的指南，具体镜像认证信息以你下载版本为准。
- [AMD UG973 2026.1 支持系统](https://docs.amd.com/r/en-US/ug973-vivado-release-notes-install-license/Supported-Operating-Systems)：本工程固定 Vivado 2026.1 + Vitis HLS 2026.1，核对 x86-64 平台与操作系统要求。
- [AMD 工具系统支持入口](https://www.amd.com/en/support/adaptive-socs-and-fpgas/installer-info-general.html)：按所选工具版本核对 Windows/Linux 支持。
- [PYNQ DMA 说明](https://pynq.readthedocs.io/en/v2.7.0/pynq_libraries/dma.html)：simple DMA 与 AXI Stream 接口。
- [PYNQ buffer](https://pynq.readthedocs.io/en/v2.5/pynq_package/pynq.buffer.html)：allocate、flush、invalidate。
- [PYNQ DMA 源码](https://github.com/Xilinx/PYNQ/blob/master/pynq/lib/dma.py)：完成状态、错误与缓存处理参考。
- [PYNQ 板卡设置](https://github.com/Xilinx/PYNQ/blob/master/docs/source/overlay_design_methodology/board_settings.rst)：Z2 board files 与 PS preset 入口。

## 发布时的许可

本目录未复制厂家大体积资源、模型或完整厂商固件。运动学根据所给原厂实现改写并在此注明来源。若参加开源提交，需核实厂家代码的再分发许可，并为本队原创部分选择许可证；不应直接把整套厂家资料标为 MIT。依赖库与 AMD IP 按各自许可使用。

## 最终YOLO11n方案依据

- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)：检测模型、训练、验证与导出接口。
- [Ultralytics 8.3.230](https://pypi.org/project/ultralytics/8.3.230/)：本工程锁定版本；不将滚动文档的新功能当作已测试功能。
- [PyTorch历史版本](https://pytorch.org/get-started/previous-versions/)：Windows CUDA开发环境采用2.7.1/cu126安装路线，未在用户Windows实机执行。
- [AMD Ryzen Windows兼容矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/windows/windows_compatibility.html)与[Linux兼容矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html)：按借用机实际SKU、系统和驱动核对ROCm支持；NPU部署不是ROCm条款的替代品。
- [Ultralytics许可](https://www.ultralytics.com/license)：依赖与模型涉及AGPL-3.0或商业许可；对外开源提交前应按实际使用方式核对相应义务。

本地指南具身智能章节为PDF物理第13–20页；要求与代码对应见 [competition_requirements.md](competition_requirements.md)。指南版本号保持原文，未取得最新补充公告或赛事方兼容性答复。正式环境版本和硬件实测结果由最终设备记录补全。

## 动态回收分拣新增依据

- [AMD AXI GPIO数据寄存器](https://docs.amd.com/r/en-US/pg144-axi-gpio/AXI-GPIO-Data-Register-GPIOx_DATA)：双通道GPIO与寄存器接口。
- [PYNQ MMIO](https://pynq.readthedocs.io/en/v2.1/pynq_libraries/mmio.html)：通过AXI GP映射访问PL外设。

本版正交解码与快照RTL、PS关联及预测由项目实现，不把官方IP文档当作本工程已经通过综合或硬件验证的证据。
