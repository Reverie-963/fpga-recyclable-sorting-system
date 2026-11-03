# Third-party provenance and declarations

Ultralytics运行库来源为官方Python分发8.4.193，源码头部声明AGPL-3.0，实际许可证原文随包保留；模型为项目适配权重，来源与哈希见host/model/model.md。本队原生代码没有在此整理过程中被重新授权为MIT/Apache。

PyTorch/torchvision与ROCm为环境依赖，原版权与许可证由上游保留；不附安装包。AMD/Xilinx HLS生成IP、板卡定义、Vivado报告与bit/HWH保留原有声明，不把生成IP认作手写RTL。PYNQ运行时为板端依赖。模型训练使用TACO及经整理的公开素材，原素材许可与模型发布条件仍由队伍在签署前确认。

KM1机械臂/ESP32属于厂商产品，现役通过厂商串口协议调用；代码包含项目自己的协议适配与ASRPRO流程，没有重刷ESP32。未把厂商全部扩展例程和安装包当成项目必需源码复制进来。自制源码、标准设备和第三方依赖分开标识。

参赛承诺与授权书必须由全体队员据实阅读并签署，本包不替任何权属方做授权承诺。GitHub仍私有，未进行公开发布。

上游参考：https://github.com/ultralytics/ultralytics 、https://github.com/pytorch/pytorch 、https://github.com/Xilinx/PYNQ 、https://github.com/pedropro/TACO 。
