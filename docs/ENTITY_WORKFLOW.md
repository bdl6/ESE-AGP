# 论文五步实体工作流：实现与复核

日期：2026-10-30。主项目 `/data1/rag/zhy/ESE-AGP`。
这是一套按论文所述步骤自建的离线实体工作流，未接入官方 Microsoft GraphRAG SDK，不能称为官方工作流逐项复刻。
旧简化版 `extract`、100件结果和故障记录保持原状；新命令为 `extract-workflow`，输出目录独立。

## 五项补齐

|要求|实现|产物|
|---|---|---|
|实体和关系联合抽取|joint.py：实体类型、名称、原文证据；关系端点、关系描述、strength∈[0,1]|joint_chunks.json和stages/原始响应|
|跨块跨轮聚合|aggregation.py：同商品实体聚合、证据去重、关系强度观测保留|聚合实体、关系与轮次轨迹|
|实体描述生成|workflow.py：LLM按证据生成描述，超过8片段分组归纳和分层合并，再执行描述证据审查|entity_registry.json中的description及evidence_ids|
|实体消歧|disambiguation.py：同类型候选＋LLM双方证据判断，置信度阈值0.9|disambiguation.json中的理由、依据、合并映射|
|规范实体重建语义图|graphs/semantic.py优先使用canonical_id，按共享实体IDF建边|新semantic及fused图，不覆盖旧图|

## 文本与证据

只处理训练交互中实际出现的商品，从已清洗products.parquet取title/features/description/details。
先把details从JSON字符串展开为可阅读字段，避免双重转义造成证据匹配失败。
v3进一步按推荐字段白名单保留Label、Manufacturer、Artist、Genre、Format、Material等，排除日期、尺寸、数量、型号和停产标记；title/features/description仍完整保留。
新工作流处理完整商品文本，不预先截断为6000字符；默认块1600字符、重叠200字符，保存绝对字符位置。
每条证据记录商品ID、块编号、原文start/end、实际原文quote和稳定evidence_id。
允许证据大小写、空白差异，但匹配后恢复原文片段；不允许凭专辑名生成原文不存在的艺人名。
明确将label/record_label类型映射为brand，纯数字不接受为非product实体；原始类型可在阶段响应中追溯。
实体与关系逐项校验并保留rejected记录；原文出现并不保证实体类型或关系解释正确，仍需质量审核。

## 迭代与聚合

每块最多3轮。模型获得本块已有实体名称、类型及关系三元组，只补充遗漏项；没有新实体或新关系时提前停止。
同商品内完全同类型/规范名去重；跨商品即使同名也先保留独立实体，避免同名异义直接误合并。
重叠分块中的同一原文位置只保留一份证据，原始逐块逐轮结果仍保留。
相同关系保留所有strength_observations，汇总强度取最大值，这是明确的工程选择；LLM强度不是校准概率。

## 描述与消歧

每个实体的所有证据分组进入描述阶段；每组最多8片段，之后分层归纳，不只取最前面几段。
每次描述生成后，用独立提示词审查是否引入原文未支持的断言，保存初稿、修订描述、supported判断和理由。该LLM审查仍不能替代人工评估。
描述必须引用实际展示的证据ID；程序能校验引用存在，不能自动证明每句话都被证据充分支持。
候选仅限同类型，小组全比较，大组使用词项/缩写阻塞，并记录候选预算与未处理数；这是启发式候选召回，不保证发现全部同义词。
相似度只用于候选排序，不直接决定合并。
合并须有同类型、same_entity=true、置信度达到阈值、双方至少一条来源和明确理由；不确定结果不合并。
canonical_id由规范名称、类型和成员身份生成，在同一输入/配置下可重复；扩充语料或合并成员后身份可能改变，图与映射必须一起版本化。
规范名称选择组内最长已有名称；所有原名进入aliases，原始实体ID映射保留。
合并后重写关系端点、聚合重复关系；对实际合并组重新生成规范实体描述。

## 关系与语义边的用途

关系三元组、关系强度和实体描述都已存储，不再省略这些论文步骤。
商品语义图仍按论文共享实体IDF公式计算，不把关系strength直接加进商品边权。
这样能够对照原方法，避免同时改变抽取流程和推荐公式。实体描述目前是补充属性，不额外引入向量模型。

## 缓存、限额与失败

每阶段请求按数据身份、模型、提示词、输入和版本生成缓存键，保存原始响应与校验错误。
只有成功阶段才能复用；缓存摘要不匹配直接拒绝。已有成功阶段可跨进程恢复，真正失败的阶段重试。
累计请求尝试先落盘再发送；额度耗尽不再请求。输出目录锁防止并发写入。
服务/格式失败会明确停止工作流，保留已完成阶段；不会把未完成流程写成最终成功。
新工作流使用本机Qwen生成服务，单次超时180秒。没有自动更改模型、升级依赖或请求第三方收费接口。

## 配置和命令

```bash
cd /data1/rag/zhy/ESE-AGP
.venv/bin/python -m pytest -q
.venv/bin/python -m ese_agp extract-workflow --config configs/entity_workflow.yaml
.venv/bin/python -m ese_agp extract-workflow --config configs/entity_workflow_pair.yaml
```

entity_workflow.yaml选原100件中的3篇最短完整文档，最大160次调用；pair配置选择两件真实共享Sony Legacy厂牌的商品，最大80次调用。
这些是小批验收，不是100件或全目录重跑，不因代码补齐就宣布全量计算完成。
旧extract命令和旧entities.json不会被新工作流覆盖。

## 输出阅读顺序

1. documents.json：实际完整文本与商品ID。
2. joint_chunks.json：逐块、逐轮实体/关系、拒绝项与停止轨迹。
3. stages/：每阶段请求缓存、原始模型返回和错误。
4. entity_registry.json：规范实体、aliases、描述与原文证据。
5. disambiguation.json：候选覆盖、每次判断及旧ID到规范ID的映射。
6. relations.json：规范实体之间的关系、强度及证据。
7. entities.json：按商品组织的最终规范实体，供语义图使用。
8. status.json：完成状态、调用量、token、图边数和未审核标记。

## 验收边界

86项测试通过，涵盖大小写证据还原、强度边界、未知端点、跨块证据、无新增停止、分层描述、别名合并与拒绝、同名不自动合并、缓存续跑和调用限额。
v1真实3件流程曾得到5个规范实体、3条关系，但发现厂牌label类型被拒绝、描述存在外部知识补全。旧v1结果保留；v2随后发现管理字段噪声，当前配置已改为v3重新验收。
没有共享规范实体时0条语义边是允许结果，不通过错误合并来制造边。
两组当前真实结果以 `paper_workflow_smoke_v3/status.json`、`paper_workflow_pair_v3/status.json` 及 `ENTITY_WORKFLOW_REPORT.md` 为准。
所有小批图均标记pilot_or_partial，训练加载器默认拒绝用于完整实验。
没有进行真实人工30条审核，不能把字段/证据校验通过写成实体或关系准确率100%。
