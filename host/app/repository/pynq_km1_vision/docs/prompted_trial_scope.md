# 固定26类的显式受监督试用范围

config/prompted_motion_policy.json中的operator_trial_scope用于记录操作者明确要求的试用范围，独立于历史校准报告eligible_materials/eligible_official_labels。没有该字段时沿用原验证范围。

- kind必须为operator-supervised-trial-scope-v1，operator_statement记录实际请求；model_sha256和taxonomy_sha256绑定当前模型及名称/材质映射。
- official_labels必须来自当前策略且无重复，每类必须有paper/plastic/metal映射；single_manual_attempt_only必须true，physical_validation_completed必须false。
- 实际动作仍走原材料机械配置、尺寸/面积/避让、现场就绪、停止、所有权、原图身份和完整等待检查。

/operation新增prompted_scope_basis、prompted_validated_scope、prompted_physical_validation_completed和prompted_motion_timing_overrides；prompted_trial_materials/labels表示当前可试用范围。任务报告execution_scope记录依据与原验证范围，不把操作者开放当作实物通过。

当前26类试用使用既有20秒目标的motion_timing_overrides，速度参数与实际等待由原距离计时实现解释；该目标不是整轮实测保证。详情见[本轮记录](../../docs/team/reviews/2026-10-30-prompted26-fast-motion.md)。
