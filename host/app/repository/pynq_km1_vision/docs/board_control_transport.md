# PYNQ 控制代理传输

`tools/board_station_proxy.py --config <现有本机配置> --port 8810`使用现有私有连接模块建立两个独立SSH会话：控制状态/心跳/停止与页面图片分离，沿用原板端认证和执行器归属。配置结构不变，不含重新规划或串口实现。

| 入口/条件 | 行为 |
| --- | --- |
| /operation、/operation/*、/board-control | 使用独立控制会话；锁等待250ms、通道创建1.5秒、HTTP等待4秒 |
| 其他板端显示请求 | 最多3个在途，超额立即503；通道创建3秒、HTTP等待15秒 |
| SSH拒绝/超时/EOF、失联、会话替换 | 返回JSON 503，不自动重连或重发，不回退PC执行 |
| GET /proxy-health | 仅本机同源；transport_contract=isolated-control-v1、automatic_retry=false、control_location=pynq_ps |
| tools/restart_native_board_proxy.py | 默认只读，只有显式--restart才更新唯一匹配的8810代理；busy必须false，原idle/blocked快照必须保持 |

HTTP等待是各阻塞读取的超时，不是整段业务端到端性能保证；控制客户端及板端原有断线停止/租约仍保留。脚本需要paramiko，现场更新工具另外需要psutil。完整原生控制部署、私有连接模块及配置仍来自现有本机/PYNQ，不由这些脚本单独重建。此前被自动审批拒绝的重启不能由代理换入口来绕过；该工具保留供现场操作者使用。

验证及实际上线状态见[本轮记录](../../docs/team/reviews/2026-10-30-board-heartbeat-fix.md)。

2026-10-30补充：原生协调器`serve_board_station.py`同样使用独立控制连接。`BoardLinkMonitor`的只读探测失败只更新部署记录`board_link_health`，不再关闭视觉中继和本地视觉页面；`available=false`明确表示该探测未知或失败，不代替控制租约、现场确认或动作授权。代理转发上游帧号、流号及帧龄头，不生成新鲜度。见[主中控请求失败修复记录](../../docs/team/reviews/2026-10-30-visual-link-lifetime.md)。
