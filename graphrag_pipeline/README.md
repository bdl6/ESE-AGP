# 独立官方GraphRAG商品实体与语义图

此目录为全新实现。只读现有products.parquet，不导入ese_agp，不修改原配置/模块/依赖，不写入原data/entities或data/graphs。使用专用.venv；所有运行产物位于本目录runs。

官方GraphRAG固定3.3.0。这里调用官方SDK的分块、实体关系联合抽取、补充抽取、实体及关系描述归纳；随后由独立适配器恢复商品→实体映射，再计算共享实体IDF商品图。没有将原有自定义抽取代码改名后冒充官方SDK。

## 文件

- product_graph/prepare.py：只读元数据，标题/特征/描述/details白名单拼接；保存显式document_id=parent_asin和内部item_id映射，校验输入SHA。
- product_graph/official.py：官方API配置和执行；600 token块、80 token重叠，每块加文档ID以避免相同段落跨商品碰撞；默认64并发。
- product_graph/convert.py：实体text_unit_ids→TextUnit.document_id→parent_asin→item_id；保存引用文本，不把模型描述伪装成原文逐字证据。
- product_graph/semantic.py：共享实体IDF求和、Top-K、按来源归一化；IDF分母/分子范围均为本次覆盖商品，样例明确标记sample。
- product_graph/common.py：JSON、摘要和输入身份检查。
- product_graph/__main__.py：本目录独立命令行。
- tests/：本实现的独立测试，不混入原tests。

## 安装

```bash
cd /data1/rag/zhy/ESE-AGP/graphrag_pipeline
../.venv/bin/python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

已安装环境后可通过requirements.lock.txt锁定完整依赖。不要在原项目.venv安装本目录依赖。

## 使用

```bash
cd /data1/rag/zhy/ESE-AGP/graphrag_pipeline
.venv/bin/python -m product_graph prepare --source ../data/processed/amazon_cds_paper_drop_session_v4 --run runs/demo --limit 3
.venv/bin/python -m product_graph index --run runs/demo --concurrency 2 --gleanings 1
.venv/bin/python -m product_graph convert --run runs/demo
.venv/bin/python -m product_graph build-semantic --run runs/demo --top-k 20
```

准备全目录时将--limit 3换成--all；正式运行可设置--concurrency 64。代码不会自动启动全量任务。--item-ids可以指定当前数据版本的商品ID。
相同设置断点重跑index使用--resume，成功LLM缓存由SDK管理。更换模型/切块/提示词/并发等设置需使用新run目录，不覆盖原输出。

## 新目录内输出

- manifest.json、documents.parquet、product_mapping.json：固定输入、商品映射和数据来源。
- settings.yaml：本轮官方SDK真实配置。
- sdk_output/：官方SDK表，保留原始结果，CSV/Parquet由固定版本后端决定。
- sdk_cache/、sdk_logs/、index_status.json：SDK缓存、日志和工作流状态。
- product_entities.json、entity_registry.json、entity_relationships.json：转换后的商品实体映射、实体字典、实体关系。
- text_unit_provenance.json：证据文本块及来源商品。
- semantic_graph.json、semantic_graph_preview.json：商品有向语义图和小预览。

## 与原自定义实现的区别与限制

1. GraphRAG默认以名称及类型聚合实体；不等于再次调用LLM确认所有别名。此实现没有额外实现R&B与Rhythm and Blues的LLM别名消歧。不能宣称完全复现原保守消歧策略。
2. 3.3.0的finalize_graph会按title去重并生成随机UUID。为避免同名异类型实体丢失，本实现使用官方extract_graph的汇总表，独立分配稳定的(type,title)实体ID，不运行finalize_graph和社区/问答相关流程。
3. 官方关系端点以名称表示；同名多类型端点保留候选ID并标记ambiguous_endpoint，不擅自指定一个实体。
4. max_gleanings=2表示首轮之后最多补充2轮，另可能有是否继续的判断请求；不同于简单固定3次HTTP。
5. 使用官方tuple格式及本目录prompts下的保守提示词，不使用旧流程的entities/relations JSON提示。提示词会复制到每次run并记录SHA，修改后必须开新run。描述是模型归纳，来源文本块可追溯，但不保证每个断言语义正确。
6. 只为商品图使用实体索引，因此不运行社区报告、问答和向量生成；不需要调用embedding/reranker服务。cl100k_base为SDK切块tokenizer，不是Qwen原生token计数。
7. SDK单块错误可能记录日志后继续，本适配器收集GraphRAG ERROR日志并阻止将其标成完整成功。index失败时保留原始表/缓存，但convert拒绝继续，需先检查并重试。
8. 未额外强制无证据记录逐字校验，也未修改SDK的截断处理；不要把旧版严格证据校验的能力归到本实现。
9. 新语义图为独立schema，不能直接假定与原训练加载器兼容；没有修改原模型接入或实现自适应剪枝。

官方资料：https://microsoft.github.io/graphrag/index/default_dataflow/ ，https://microsoft.github.io/graphrag/config/yaml/

## 本次验收

独立测试9项通过。`runs/smoke_v3/`使用3件真实v4商品，得到13个实体、11条关系、2条商品语义边；使用保守提示词后厂牌描述可回到原文角色。SDK索引约48.7秒（共享服务负载未控制，不用于全量性能外推）。相同设置--resume已验证；原有受跟踪文件无改动、原项目模块无导入。详细结果见VALIDATION.json。样例仅说明接口和来源链路可用，不代表全量抽取质量验收。
