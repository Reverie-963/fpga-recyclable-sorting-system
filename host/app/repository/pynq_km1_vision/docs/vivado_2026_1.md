# Vivado / Vitis HLS 2026.1 环境与构建

本项目固定使用 **Vivado 2026.1 + Vitis HLS 2026.1**，目标为 TUL PYNQ-Z2 的 `xc7z020clg400-1`。版本记录在 `build/toolchain_version.txt`。版本检查通过只表示工具版本符合要求，不代表综合、时序或上板验收通过。

## 本机已安装：直接检查与使用

2026-10-06 已确认本机程序目录：

- Vivado：`D:\work\AMDDesignTools\2026.1\Vivado\bin\vivado.bat`
- Vitis：`D:\work\AMDDesignTools\2026.1\Vitis\bin\vitis-run.bat`

两者均已实际返回 **2026.1**。普通 PowerShell 在工程目录执行即可，不要求先手工设置 PATH：

```powershell
.\tools\build_windows.ps1 -CheckOnly
```

脚本优先使用 `-ToolRoot`（默认取环境变量 `KM1_TOOL_ROOT`），未指定时查 PATH，再查本机 `D:\work\AMDDesignTools\2026.1`。安装位置不同或 PATH 指向旧版本时，明确指定版本目录：

```powershell
.\tools\build_windows.ps1 -ToolRoot D:\work\AMDDesignTools\2026.1 -CheckOnly
```

本机 `vivado -version` 输出正确版本但退出码为 1；脚本会再启动无工程的 batch/Tcl 核验，只有退出码正常且版本标记匹配才放行。实际构建的失败检查保持有效。该检查不连接板卡、不运行机械臂、不执行综合。

## 其他电脑的安装与手动环境加载

仅在其他电脑尚未安装时，使用 [AMD 2026.1 官方安装器](https://www.amd.com/en/support/downloads/adaptive-socs-and-fpgas/development-tools/2026-1.html)，安装 Vivado、Vitis HLS 和 Zynq-7000 器件支持。本机不需要重复下载安装。仅安装 Lab Edition 不能完成本项目构建。系统要求分别核对 [Vivado UG973](https://docs.amd.com/r/en-US/ug973-vivado-release-notes-install-license/Supported-Operating-Systems) 与 [Vitis UG1742](https://docs.amd.com/r/en-US/ug1742-vitis-release-notes/Installation-Requirements)。

在 AMD 2026.1 工具命令提示符里启动 `powershell -NoProfile`。如果使用普通命令提示符，先 `call` 实际安装目录里的 Vitis 和 Vivado `settings64.bat`，然后启动 PowerShell。不要直接在 PowerShell 中运行 `.bat` 后就假定它能修改父 PowerShell 的环境。

```powershell
vivado -version
vitis-run --version
.\tools\build_windows.ps1 -CheckOnly
```

脚本检查 Vivado、vitis-run 和 v++ 均为2026.1，不回退到旧vitis_hls。2026-10-07实机发现旧Tcl入口漏编译设计文件，现改用生成的build/hls_segment_open.cfg和组件目录build/hls_segment_open。顺序为vitis-run --csim、v++ --compile --mode hls、vitis-run --cosim、vitis-run --package，再运行Vivado。build/hls.tcl已禁用并给出迁移提示。参见[AMD HLS命令说明](https://docs.amd.com/r/en-US/ug1399-vitis-hls/vitis-v-and-vitis-run-Commands)。Vivado Tcl入口也会在创建工程前检查版本。

## Windows 路径与构建

当前工作区 `D:\work\嵌赛_FPGA` 可用于编辑。按原工程的 ASCII 路径要求，正式构建时将整个 `pynq_km1_vision` 文件夹复制到纯英文、无空格的短路径，例如 `D:\fpga\pynq_km1_vision`，每次构建前同步最新源文件。脚本会在构建前检查工程、board files 和 XDC 路径。`-CheckOnly` 不创建或重置 HLS/Vivado 工程。

准备官方 TUL PYNQ-Z2 board files，例如放在 `C:\fpga\board_files`；不能用 PYNQ-Z1 预设替代。切换到英文路径工程后执行：

```powershell
Set-Location D:\fpga\pynq_km1_vision
.\tools\build_windows.ps1 -CheckOnly
.\tools\build_windows.ps1 -BoardRepo C:\fpga\board_files -Kernel segment_open -EncoderXdc .\build\encoder_sen0230_pmodb.xdc
```

示例 XDC 对应已有 SEN0230/Pmod B 方案，使用前按接线说明核对实物。构建顺序为 HLS C 仿真、综合、C/RTL 协同仿真、IP 导出，以及 Vivado 实现和位流生成。`board/vision.bit` 与 `board/vision.hwh` 必须来自同次成功构建；检查 `report/timing.rpt`、`report/utilization.rpt` 和工具日志。

## 2026-10-06 本机检查

已运行更新后的 `tools/build_windows.ps1 -CheckOnly`，Vivado 2026.1 batch/Tcl 启动核验及 Vitis 2026.1 版本查询通过。此前“未找到可运行程序”的记录已失效。尚未执行 HLS 综合、Vivado 实现、器件/board files 完整检查或上板验证，不将版本核验写成构建成功。

2026-10-07硬件状态：microSD已到，输送带未到；KM1（ESP32）、PYNQ-Z2、C270已有到货记录，编码器型号及配件清点待实物确认。现在可写卡启动，并在传送带到货前构建PL、验证相机和编码器电气/计数；最终毫米比例和动态标定待机构安装。完整步骤见[开工手册](commissioning_roadmap.md)。

本次更新当前源码和操作文档；根目录旧 ZIP 与已有历史测试报告保留为原始快照，不能当作本次 2026.1 验证结果。后续请从当前工程构建，避免重新解压旧 ZIP 覆盖配置。

## 2026-10-07 C仿真链接错误修复

首轮报`undefined symbol: segment_open(...)`：生成的csim.mk仅包含hls_open_tb.cpp，图像核源码未参与链接；hls.app设计路径登记异常。仅调整绝对路径及重新创建旧式项目仍复现。改用配置文件入口后，同一份设计和测试程序均进入编译，C仿真返回`CSim done with 0 errors`，随后C综合也通过。工具头文件的GMP宏重定义警告在通过的仿真中仍存在，因此不是本次链接失败的原因。

一键构建仍使用原PowerShell命令，无需重装工具。HLS产物现在位于`build/hls_segment_open/hls`，IP目录为其`impl/ip`；Vivado脚本已同步读取新路径。完整一键流程的后续状态以`report/fpga_build_config_20261007.log`及构建结果记录为准。

本轮后续已完成Verilog协同仿真、IP打包、100MHz布局布线及位流生成；DMA双向DDR映射修复、串行恢复和最终报告见`report/fpga_build_repair_20261007.json`。板端加载仍待完成，编码器输入约束仍待专项验收。
