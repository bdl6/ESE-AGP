# 20条历史输入与独立预测目标（2026-11-01）

论文协议的max_session_length=20现明确表示历史输入上限。清洗后的会话最后一次交互为目标，输入为目标之前最近20次；SQL中等价地保留最近21次，再将最后一次分离，短会话不补足。
本次只改变长度口径，过滤仍为单轮user→item→session校验，尚未改为迭代到收敛；旧temporal协议保留历史语义。

- 新配置：configs/amazon_cds_paper_history20.yaml
- 新数据：data/processed/amazon_cds_paper_history20_v2/
- 新行为图：data/graphs/paper_history20_v2/behavior.json（533097条边）
- 验收：runs/history20_change_20261101/verification.json
- 预处理日志：runs/history20_change_20261101/prepare.log

训练282311条、测试31368条，逐条按父级商品ID核验，目标和会话划分不变。训练1220条、测试152条历史由19增加到20。商品目录63493不变；数据指纹已改变。
99项测试通过，新增2/20/21/25次会话和自定义长度边界测试。

旧数据、模型、语义图与融合图仍保留在原位置作为历史实验。旧模型指标不能作为新版本指标；尚未重训，也未重新绑定/构建新版本语义图和融合图。实体文本结果有望复用，但必须核验商品身份与证据、生成匹配新指纹的产物，不能直接混用旧图。
