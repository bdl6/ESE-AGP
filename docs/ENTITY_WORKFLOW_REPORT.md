# 论文五步实体工作流最终验收（v3）

日期：2026-10-30。两个真实小批，共5件不同训练商品。原v1/v2输出均保留。

|样例|商品|规范实体|关系|跨商品合并|语义有向边|累计调用|运行秒|
|---|---:|---:|---:|---:|---:|---:|---:|
|smoke|3|5|5|0|0|17|87.8|
|pair|2|3|4|1|2|16|100.4|

## 完成内容

联合实体/关系抽取、关系强度、跨块跨轮聚合、无新增停止、带引用的分层描述及证据审查、LLM消歧和规范ID语义图已接通。
所有实体/关系证据位置均回到实际文档核验，关系端点均存在于规范实体注册表。
最终测试86项通过；包含R&B/Rhythm and Blues别名合并、拒绝低置信和不同类型合并、跨商品同名不自动合并、分层描述、缓存和限额。
v1/v2真实验收发现厂牌label类型被拒绝、日期尺寸等噪声和无依据描述；v3加入明确类型映射、推荐详情字段筛选、数字实体拒绝、独立描述审查。
所有旧结果保留，没有将失败/较差结果改写为最终结果。

## 边界

只完成5件真实商品的完整流程验收，原简化版100件和全训练目录未按新方法重跑。
这些图标记pilot_or_partial，不能用于完整推荐实验；真实人工30条质量审核未完成。
描述引用存在、LLM审查通过不等于语义准确率100%；候选召回是启发式，不承诺发现全部别名。
关系强度与消歧置信度为模型主观输出，不是校准概率。
按论文步骤自建流程，没有使用官方GraphRAG SDK，不宣称复刻未公开的Agent全部配置。

## 位置

配置：configs/entity_workflow.yaml、configs/entity_workflow_pair.yaml。
结果：data/entities/paper_workflow_smoke_v3/、data/entities/paper_workflow_pair_v3/。
文件：documents.json、joint_chunks.json、entity_registry.json、relations.json、disambiguation.json、entities.json、status.json、stages/。
说明：docs/ENTITY_WORKFLOW.md；机器验收：runs/entity_workflow_verification.json；测试：runs/entity_workflow_tests.log。
