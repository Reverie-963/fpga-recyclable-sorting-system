# 最终系统通信协议 v2

**v6任务入口：**两种零件、六种状态，详见[开工手册第8节](commissioning_roadmap.md#8-第六步两种零件六种状态的外观缺陷检测)。根配置尚未自动迁移，先完成第8.2节；4种缺陷可抓取，2种正常不抓，末端直行仍需复核。

上位机只提供检测接口，不接受网络舵机命令。HTTP/JSON通过PYNQ与PC的直连网线传输：板卡192.168.2.99，PC检测服务http://192.168.2.1:8765。Bearer口令来自KM1_TOKEN，至少24字符；不提供公网TLS。服务单请求顺序处理，不堆积帧。2026-10-16 起当前上位机为 HX 370 AI PC，原 Y9000P 已退出使用。沿用上述目标地址，推理后端使用 ROCm；新机须重新验收实际通信及模型/运行指纹，本次文档更新不改变协议格式。

## 请求与返回

- GET /health：protocol=2、task=detect、labels、providers、model_id、runtime_id。
- POST /detect：protocol=2、request_id、frame_id、整帧png（base64）。请求上限8500000字节；图像长宽各不超过1920，像素数不超过1920×1080。
- 响应回显protocol、request_id、frame_id，附image_size `[width,height]`、detections、timing和指纹。
- detection格式：`{"label":"cap_notch","confidence":0.91,"xyxy":[10,20,50,70]}`。坐标为原图像素坐标，不是letterbox空间；必须在图像内。置信分数范围0..1，不与旧CNN的probability_margin混用。
- timing为preprocess_ms、inference_ms、postprocess_ms、total_ms；客户端另记rpc_ms和request_bytes。RPC包含网络、服务与推理，不是纯网络延迟。

YOLO权重SHA256形成model_id；配置摘要形成runtime_id。板端可预设expected指纹，未预设时首次成功响应锁定。批次中模型或推理配置变化均拒绝。服务最近4096个请求编号拒绝重复；板端task_id在发机械动作前排他创建并落盘，不自动重放。

## 时间与异常

帧年龄使用PYNQ的monotonic时钟，从应用完成采集到处理结束计时，不依赖两台机器墙上时间。使用同一采集帧进行PL与YOLO融合；新帧复查时重新取掩码和检测。帧年龄不是传感器曝光时间戳，系统不宣称已建立硬实时上界。

断网、超时、过期、响应ID错误、坐标异常、指纹变化或模型错误都停止批次，不静默退回CPU/旧分类器。低分和没有有效候选可以继续观察。动作前验证不通过或动作后证据不足时停止，人工检查后重新启动。

动作前复查包含串口打开后及移向目标前。已经发出的运动不能被网络失败或Ctrl+C撤销；本工程没有独立硬件急停互锁。分类盒图像变化仅是视觉证据，真实任务成功率仍由录像人工标注。

旧协议v1的/infer只用于历史分类对照，不能与本版检测控制器交叉连接。

## AI PC 本机监视（可选）

可选 AI PC 监视页见[工位视觉监视](vision_monitor.md)：`detect_server --monitor-port 8766` 额外开启仅监听 `127.0.0.1` 的观察服务。页面的 `GET /snapshot` 提供最新一组 JPEG、检测结果、帧/请求编号、服务实例 `stream_id`、单调时钟计算的本机收帧后年龄和刷新率；它是本机预览接口，不是板端控制协议。该年龄不包含板端采图到 AI PC 收帧的时间。快照仅保留最新帧，异步 JPEG 编码的待处理槽也只保留一帧，旧待处理帧会被替换。页面不接收认证口令，不提供机械控制接口，原 `/health` 和 `/detect` 的格式与认证不变。

## v5动态关联

线上检测协议仍为v2：服务只做整帧检测。板端在提交请求时保存该帧几何及track_id，响应只绑定该快照中的唯一对应编号。编号已经失效则丢弃，不能将旧检测框直接应用于新画面。目标的当下位置由编码器与板端几何更新，不要求PC时钟与PYNQ时钟一致。编码器本地协议见 [conveyor_setup.md](conveyor_setup.md)。
