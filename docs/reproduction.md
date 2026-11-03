# 复现与核对

## 1. 离线核对（本包已执行）
解压到英文短路径，运行readme.md列出的两个纯文件验证入口。现役权重、参考库、bit/HWH及IP包均有SHA256；检查结果保存在provenance/validation。无需训练、GPU、板卡或串口。

## 2. 电脑环境
现役Windows 11、Python3.12.10、PyTorch2.12.0+rocm7.14.0、torchvision0.27.0+rocm7.14.0，实际执行Ultralytics8.4.193。host/runtime是当时优先导入的运行库；普通环境中安装的Ultralytics8.3.230不是被测实现。设置PYTHONPATH时把host/runtime及host/app/repository/pynq_km1_vision加入模块搜索路径。依赖全量观察记录在host/deploy/requirements_gpu_observed.txt，原GPU轮子来自AMD发行源，不应直接按普通PyPI CUDA教程安装。

实际版本与驱动见host/deploy/environment.json，板端版本见board_environment.json。原始测量没有逐轮独立记录功耗模式，现读到的Windows方案只作当前环境盘点，不能倒填历史。

30次补测离线复算：在工程根目录运行`python metrics/scripts/recompute_pilot30.py`，仅读包内记录，不启动设备。

## 3. 离线模型调用
权重和参考库已提供，不需要下载训练数据。通过host/model/verify_assets.py验证；host/model/model.md声明来源、数据范围和限制。模型API与当前参考辅助处理在PC源码src/console_candidate_monitor.py、inventory_reference.py；需读取对应execution_profile.json，不能随意改阈值、类别映射或默认高度后沿用本包成绩。完整GPU推理这次未重跑。

## 4. FPGA构建
已有配对产物可先用verify_assets.py作纯文件核验。fpga/build/build_joint.ps1以已归档HLS IP和原BD脚本重建联合工程，保留初次布局失败与物理优化步骤；fpga/build/build_hls.ps1是HLS源级重建入口。脚本只在指定新输出目录构建，不自动上板、不启动机械。Vivado/Vitis HLS2026.1、PYNQ-Z2板卡定义随包提供。此次仅检查脚本语法和来源/IP身份，不做完整综合；重建时仍须实际检查资源、时序、警告、接口和成对产物，不保证二进制逐字节重现。

## 5. 板端部署与监督实机（待队伍在独立环境复核）
PS源码快照保留现役实现，但部署身份、标定绑定、环境、秘密及账本不能直接跨机器复用。请从host/app/repository/tools/stage_board_station.py、stage_board_control.py和board_runtime_bootstrap.py查看分阶段入口，先生成新工位标定与机器绑定，再运行相应只读预检。当前没有可证明从本包自动重建全部现役机器绑定的一键部署入口，此缺口明确保留。

供电/网络就绪后先恢复相机与路径依赖，再启动唯一控制器和PC视觉代理。重启板卡不会自动启动这些应用服务。下次恢复需核对最新保护和恢复记录；不得把原已完成任务重放。相机移动后先用四标记复核机位；当前右侧箱区有出画记录，实抓前改善取景再验证。

初次仅做在线无动作预演；随后操作者确认停稳、张开空爪、样件实测高度、目标箱空、双手撤出且有人监督，再在已核验范围内低速单件验收。UART回显和软件占用检查与独立完整入箱真值分别记录。此流程是下一步要求，不意味着本次重做了实验。

## 6. 第三方与签署
third_party/notice.md保留第三方声明及核对入口。本包不替队伍选择整体开源许可证、不更改仓库可见性、不代签授权。全部队员应使用随管理目录保留的空白模板填写队伍、年份与签名后扫描。


运行库补充：`host/runtime/ultralytics_runtime_8_4_193.zip`保留上游原始文件名，先用`python host/deploy/prepare_runtime.py --output <new_runtime_dir>`解压到新的本地目录，再将该目录加入PYTHONPATH。源目录中的README.md/SKILL.md保留通行格式（官方推荐目录也使用README.md），其他普通目录和文件统一小写；第三方归档内部保留其原始大小写。
