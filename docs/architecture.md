# 架构与功能划分

作品针对固定工位金属、塑料与纸类可回收物的受监督识别、抓点计算、搬运和投箱后复查。

现役链路为相机→PYNQ PS采图与PL RGB六区域统计→以太网/本机桥接→AMD Ryzen AI 9 HX 370 / Radeon 890M上的ROCm GPU实例分割与参考辅助材质判断→PYNQ PS同帧几何、坐标/高度转换、任务规划、所有权和账本→PL路径高度批量检查→PS串口→KM1/ESP32执行→新鲜图像复查访问过的箱区。

电脑还负责本地界面与视觉转发。PYNQ PS当前承接原生任务权威，不能因旧KnownItem入口含PC调度而同时启动两个执行器。PL不是普通串口转接：现役包含station_vision_accel与motion_path_accel两颗自定义HLS核，100 MHz，m_axi访问DDR与AXI4-Lite控制；当前联合设计未实例化独立AXI DMA IP。历史mask/DMA核与phase_pixel候选均不计入现役硬件成绩。

图示见`figures/01_system_dataflow.svg`、`figures/02_fpga_architecture.svg`；真实BD见`figures/actual_bd_readable.svg`。图中源区为标定纸与右侧实际登记取物区域，示意框不等于软件精确执行边界；此前把大片灰垫标为源区的照片标注不应沿用。

任务采用初始场景计算并冻结整批路线，按次序执行，末尾用新鲜图像检查访问箱区占用。占用变化并不证明同一物体完整入箱，独立落箱真值另列。软件限制6件/1200秒不是六件物理验收。机械臂无舵机角度读回，UART回显与等待不等于位置反馈。

尚未完成总体成功率、真实运动响应尾延迟、多件不中断整批、无人值守与独立毫米精度验收。当前项目不是SAM2+CLIP集成，不包含NPU推理实装。
