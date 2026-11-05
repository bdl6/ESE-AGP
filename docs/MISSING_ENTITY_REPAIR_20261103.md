# 571件商品实体修复（2026-11-03）

527件空结果包括385件只有标题、142件有其他文本；另44件抽取报错。
全部571件从原文title字段保留product实体，保存精确证据位置和metadata_title_grounded来源，不能把此兜底宣称为LLM抽取成功。没有推断品牌、艺人或关系。
142+44件以4并发低负载定向补跑，启用经过服务验证的response_format=json_object，仍执行实体语义及原文证据校验。单件业务失败不停止其他任务。

补丁与状态：data/entities/catalog_v4_missing_repair_20261103/。
最终实体：data/entities/catalog_v4_repaired_20261103/。
最终图：data/graphs/paper_drop_session_v4_repaired_20261103/。
日志：runs/catalog_missing_repair_20261103.log。
独立官方GraphRAG任务未修改，旧实体结果和旧图未覆盖。

新增实体尚未完成跨商品消歧，暂保留单独规范ID，不凭同名自动合并。标题实体不保证能与其他商品建立边。描述、消歧以及可能未补齐的其他属性仍单独标记pending，产物维持partial，不开始正式训练。

125项测试通过。运行结果以repair_status.json及最终recovery_report.json为准。本次代码未提交推送。
