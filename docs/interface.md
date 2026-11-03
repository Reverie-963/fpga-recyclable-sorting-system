# 接口与异常处理

PC与PS经以太网传输控制请求、状态、无损PNG和实例结果；应用RTT分别统计，不将其叫作单向链路延迟或物理响应。跨主机单调钟不直接相减。物理响应记录使用板端同一单调时钟的任务决策帧与只读采样帧，并给出采样上下界；曝光与USB管线偏差没有独立标定。

实例`image_size=[宽,高]`，`mask_rle`是完整原图逐行、背景段起始、0/1交替的整数游程，总和为宽×高；前景开头首段为0，不是COCO压缩RLE。消费规格最大640×480。Python的bbox右下界不含；历史HLS min/max为包含边界，不能混用。

帧号、流号、BGR哈希、模型/配置/参考库与执行绑定身份要一致；同帧无损PNG和PL统计绑定。固定标记10/11/12/13用原镜头内参求机位，`object_height_mm`是桌面至物体顶部毫米高度，不是夹持深度或像素。

PS到ESP32使用既有ASCII舵机命令协议；接口源码在`host/app/repository/pynq_km1_vision/openmv_gripper/`。协议单位、PWM、限位、时长及实际端口须来自相应实测配置，不以机器枚举顺序猜测端口。现役板端机械串口和ASRPRO语音串口按设备VID/PID分别核对。

保留任务所有权、真实客户端心跳、STOP、pending、未知回执与人工复核。断线不自动重发动作；所有权不静默换主机；错误不改成功。语音未武装时不得自行开启分拣。具体保护见原始源码配套`runtime_admission.md`、`simple_bin_verification.md`、`execution_model_follow.md`和`sorting_station_interface.md`（位于host/app/repository/pynq_km1_vision/docs）。

当前路径核寄存器/控制与位流配对见`fpga/build/paired/manifest.json`与HWH；RGB六区接口、输入字长、请求tag和缓存同步以`fpga/ps_runtime/pynq_km1_vision/src/motion_path_runtime.py`及`motion_path_service.py`为准。不得直接套用历史mask核的AXIS契约。
