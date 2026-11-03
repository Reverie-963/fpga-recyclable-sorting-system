# Mac协作与CPU复现

**v6任务入口：**两种零件、六种状态，详见[开工手册第8节](commissioning_roadmap.md#8-第六步两种零件六种状态的外观缺陷检测)。根配置尚未自动迁移，先完成第8.2节；4种缺陷可抓取，2种正常不抓，末端直行仍需复核。

Mac使用同一YOLO11n源码、数据集和协议v2，承担数据整理、CPU复现及SSH操作PYNQ。正式赛机为AMD Ryzen AI PC，Windows NVIDIA承担CUDA训练。

需要Mac临时接替上位机时，将板卡网线改接Mac有线口，把该口设为192.168.2.1、掩码255.255.255.0、网关/DNS留空，板保持192.168.2.99；此时Y9000P不在这条直连链路上。Wi-Fi保留下载用途。

在本地英文路径复制项目，使用当前架构对应、依赖支持的Python（建议3.11），新建专用环境：

```bash
cd ~/fpga/pynq_km1_vision
bash tools/setup_macos.sh
.venv-yolo/bin/python -m src.doctor
.venv-yolo/bin/python -m src.train_yolo --data data/defect_dataset.yaml --device cpu --weights yolo11n.pt
export KM1_TOKEN='两端相同的至少24字符口令'
.venv-yolo/bin/python -m src.detect_server --config config/host_cpu.json --host 192.168.2.1
```

defect_dataset.yaml的path改为本机真实绝对路径，服务使用上述固定有线地址。训练采用同一最终模型，CPU只作平台对照，不代表AMD性能。已有Windows训练权重可直接复制data/models/best.pt和model.json，再运行服务，不必重新训练。

上传源码、数据与配置到PYNQ，排除虚拟环境：

```bash
ssh xilinx@192.168.2.99 'mkdir -p /home/xilinx/pynq_km1_vision'
scp -r src config tools sim build docs data board report metrics skill config.json requirements.txt README.md xilinx@192.168.2.99:/home/xilinx/pynq_km1_vision/
```

板端接线、标定、构建产物和最终批次命令与 [Windows手册](windows_final.md) 中PYNQ步骤完全一致。EDA工具使用受支持的x86-64 Windows/Linux构建机，Mac可以远程访问。

最终物理任务已改为轻质零件外观缺陷检测与动态剔除。六状态标签与Windows共用；控制和编码器只在PYNQ运行，Mac不承担动态机械控制。新增配置见 [conveyor_setup.md](conveyor_setup.md)。
