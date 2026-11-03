# PYNQ 加速核自检与性能诊断 Skill

把 DMA 长度检查、双核结果比对、作业身份核对和完整调用计时整理成可移植工具。
面向开发者或智能体；Python 3.11+，核心工具只依赖标准库，不要求安装 PYNQ、NumPy、
Vivado、GPU 环境或连接本队设备。整目录复制后即可离线使用。

## 适用范围与交付

- `SKILL.md`：智能体入口及证据解释规则，目录也可复制到用户的技能目录。
- `scripts/accelerator_check.py`：独立命令行工具，生成 `report.md` 与 `report.json`。
- `examples/dual-core-contract.json`：当前二值掩码双核的可编辑接口描述。
- `examples/renamed-core-contract.json`：不同 IP 名称/小帧尺寸的接口示例。
- `tests/`：不需要硬件的独立参考、元数据、错误与 HTTP 协议测试。
- `references/contracts.md`：配置/记录格式、扩展方法与失效边界。

硬件文件检查可用于其他具有相同 HWH 元数据形式的 PYNQ 工程，IP 名称、参数、寄存器、
传输尺寸均由配置指定。当前算法参考和在线适配器只覆盖：

1. 二值掩码的面积、六项几何矩、包含右下边界的像素包围框，不做腐蚀/膨胀。
2. 两幅同尺寸二值掩码的交集、消失、新增、并集、异或输出及变化包围框。

不是任意 HLS 核的通用正确性证明。其他算法需要增加独立参考和对应适配器。
不连接相机/串口，不改业务控制器，不装驱动，不构建或加载位流。

## 无设备快速演示

以下命令在本目录执行；换平台仍使用相同 Python 命令。每个输出目录必须尚不存在。

```text
python scripts/accelerator_check.py demo --output check-demo
python scripts/accelerator_check.py replay --recording check-demo/synthetic-recording.json --output check-replay
python scripts/accelerator_check.py replay --recording check-demo/deliberate-failure.json --output check-failure
python -m unittest discover -s tests -p "test*.py" -v
```

前两条退出码为 0，第三条故意用旧作业号，应返回 1，报告指出
`statistics_mismatch:job_id`。demo 的 32 个用例包含两种算法、四种尺寸、空图、满图、
最后一个像素及稀疏图，全部明确标记为 `synthetic`。它们的预设时间只演示报告格式，
不能用于硬件成绩。`cases.json` 仅是小规模诊断集，不代替完整硬件边界回归。

## 检查已有硬件文件

先从真实构建记录确认文件来自同一次构建，再冻结身份；不要为缺少构建记录编造 ID。

```text
python scripts/accelerator_check.py profile --bit path/to/vision.bit --hwh path/to/vision.hwh --build-id actual-build-id --contract examples/dual-core-contract.json --output device-profile.json
python scripts/accelerator_check.py inspect --profile device-profile.json --output file-check
```

`profile` 保存文件的实际路径和 SHA-256，**不会自动证明配对关系**；将配置迁移到另一台
机器时可修改路径，但应保留冻结的构建身份/哈希。profile 属于本机配置，不随比赛包分发。
`inspect` 校验冻结哈希、IP 名称/参数、指定寄存器偏移，以及
`width × height × bytes_per_pixel ≤ 2^DMA长度位宽 − 1`。
未在接口描述中声明的属性不属于检查结论。

## 调用已经运行的板端服务

只支持 `sorting-statistics-v1`：`GET /health`、`POST /run`。
详见[接口与记录格式](references/contracts.md)。先从设备权威状态确认独占、空闲及服务身份；
服务的 `/health` 没有机械工位所有权信息，**本工具不能代替这项确认**。
已有设备授权覆盖本次检查时沿用；其他服务占用时只做离线检查，不自动停服或换 overlay。

将现有服务令牌通过环境变量 `PYNQ_CHECK_TOKEN` 提供，不写入参数、示例或报告。
端点由本机实际服务配置提供，工具不会发现或猜测设备地址。

```text
python scripts/accelerator_check.py rpc --profile device-profile.json --endpoint http://device-host:8782 --token-env PYNQ_CHECK_TOKEN --service-kind hardware --allow-run --exclusive-device-confirmed --repeats 3 --output board-check
```

默认使用 demo 生成的 32 个二值算法诊断输入；也可用 `--cases` 提供相同格式的自定义输入。
每个用例先做一次预热，再计时；身份/会话/状态、统计量、输出摘要逐次核对。
默认请求超时 5 秒；`--repeats` 仅允许 1–10，总计最多512次（含预热），默认运行预算60秒，
可用 `--budget-seconds` 设置1–300秒。预算在每次调用前检查，已在途请求最多还需一个请求
超时窗口。遇到第一次异常立即停止，不重试、不恢复设备。
原始相关响应保存在 `measurements.jsonl`，中断或失败仍生成失败报告。
服务本身的驱动异常处理属于该服务，调用前需核对其边界。

计时口径：`reference_ms` 是本机纯 Python 参考与摘要；`packing_ms` 为输入解码/打包；
`service_call_ms` 含 JSON/HTTP、服务排队及 PS/DMA/PL 完整往返；`validation_ms` 为响应核验。
不包含每次前置健康请求，未测独立 PL 核耗时；未独立分解 PS/DMA/网络耗时。
纯 Python 参考不是优化 CPU 基线，因此工具不输出泛化的“FPGA 加速比”。
按算法与尺寸分别统计中位数/P95/最大值，预热排除；不足100项不输出P99。

## 失效条件与结论边界

- bit/HWH 或服务身份不匹配、缺少参数/寄存器、DMA 容量不足：拒绝该检查。
- 模型掩码格式不符合本工具完整原图行优先 RLE，或尺寸超过640×480：拒绝。
- 传输失败、作业号/请求号/会话错误：保留首次失败，不自动重发硬件请求。
- `inspect` 只核文件，不读寄存器，不证明已加载 overlay、引脚、时钟、TLAST 或时序正确。
- 回放文件中 `recorded-hardware` 是记录提供者的声明；回放不提升为本轮板端实测。
- HTTP 服务响应也是远端声明，不能独立排除 CPU 替代实现；仍需板端驱动、硬件产物和
  实验记录交叉验证。报告始终保留这一边界。
- 没有实物抓放、分类准确率、机械精度或跨板迁移验收结论。
- 复用至不同算法/接口时，参考和适配器必须一起修改并独立验证。

## 来源与已验证复用结果

源自本项目的真实观察：旧 DMA 长度上限不能硬编码；AP_DONE/作业号需要关联；错误恢复时
活动缓冲区不能提前释放；相同 PL 核在整帧和小 ROI 的完整调用成本不同。
对应项目来源：`src/km1_dma_runtime.py`、`board/dual_core_acceptance.py`、
`board/dual_core_benchmark.py`、2026-10-31 FPGA阶段优化记录。
上述来源是设计依据，历史板测不是这个新技能包的本轮硬件验收。

本轮实际验证结果见随包携带的[验证快照](references/validation.md)，包括44项标准库测试、
40组既有参考对照、真实本机文件检查及隔离目录复用。完整命令及结果另见项目记录
`docs/team/reviews/2026-10-31-pynq-accelerator-skill.md`。
交付时验证离线脚本、真实本机 HWH 元数据、模拟 HTTP 协议、复制到独立目录后的使用；
不把模拟服务写成真实板测，不把复制目录写成跨板验证。
竞赛前仍需补一次本技能的独占真实服务运行，并由另一环境/操作者复现与保存证据。

AMD赛题3.2.2.5、3.2.5.3接受工具脚本/标准化工作流；本目录提供赛方要求的适用范围、
用法、失效条件和可核验复用说明。是否加分由评审决定，不承诺固定分值。
