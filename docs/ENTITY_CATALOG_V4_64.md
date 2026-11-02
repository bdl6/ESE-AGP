# v4全商品实体抽取及语义图（2026-11-02）

配置configs/entity_workflow_catalog_v4_64.yaml；范围为全部12492件目录商品的静态元数据，包括训练未出现商品。不使用测试交互构造行为边。64并发，保持五步及证据校验、重试、截断拆块、失败隔离。

旧万商品任务在消歧期间触及500000次累计请求上限停止，未完成。新任务逐商品核验相同数据指纹、文本、模型与抽取参数后，以旧目录为只读成功缓存来源。每次还需stage、完整提示词和payload匹配，验证缓存摘要和输出结构才复用；失败缓存不复用。请求计数只记录本次新调用，reused_stage_hits单独记录复用次数。

新请求上限200万，候选上限300万；不会将触及限额写成成功。全部商品均会尝试，失败商品隔离，最终如有错误仍为partial。

- tmux：ese-agp-catalog64
- 运行报告：runs/entity_catalog_v4_64_20261102/
- 实体结果及实时状态：data/entities/catalog_v4_64_20261102/
- 语义图：data/graphs/paper_drop_session_v4/semantic.json
- 融合图：data/graphs/paper_drop_session_v4/fused.json
- 启动日志：runs/entity_catalog_v4_64_20261102.log

本次没有推荐模型训练，也未实现自适应图剪枝。语义图包含全目录静态元数据，在完整图标志下要求所有目录商品实体处理成功；任意失败均明确标为部分图。语义候选累加预算5亿，逐来源统计避免构造完整N平方矩阵。
