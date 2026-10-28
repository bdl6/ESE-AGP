# ESE-AGP 代码设计与实施顺序

日期：2026-10-28。Day02 已在服务器实现真实数据预处理与可选 ID/类别/相对价格融合，43 项测试及 CUDA 前反向通过。全量预处理、两组各20轮训练及独立验证已完成，结果见 `docs/DAY2_REPORT.md`。本文仍包含完整模型的后续设计，不能将全局图/剪枝/多视角当作已实现。

当前继续使用平铺模块便于学习：新增 `download.py / amazon.py / artifacts.py / features.py`，`model.py` 通过 `use_item_features` 切换 ID-only/属性局部图。正式配置为 256 维，tiny 为 32 维。完整全局图与剪枝仍待实现；checkpoint 当前支持推理加载，尚无中断续训命令。

## 1. 代码如何拆分

使用一个 Python 包和统一命令入口，先完成可训练链路，再通过配置逐项增加模块。环境统一使用项目 `.venv`，不使用全局 site-packages。服务器运行与测试由 `.venv/bin/python` 发起，依赖通过 `python -m pip` 安装；`.venv` 加入 Git 忽略规则。

```text
ese-agp/
├── PLAN.md / CODE_DESIGN.md / README.md
├── pyproject.toml / uv.lock / .gitignore
├── configs/                     # 主配置、小样本配置、三组消融配置
├── src/ese_agp/
│   ├── __main__.py / cli.py      # prepare/extract/build-graphs/train/evaluate/export/serve
│   ├── config.py / types.py      # 配置、数据结构、输入验证
│   ├── data/                    # 原始读取、会话切分、映射、Dataset、collate
│   ├── entities/                # 提示词、模型适配器、JSON校验、清洗、缓存
│   ├── graphs/                  # 行为图、语义图、融合图、批内局部图
│   ├── models/                  # embedding、图层、剪枝、多视角、完整模型
│   ├── training/                # 基线、Trainer、指标、checkpoint
│   └── serving/                 # 模型加载、共享预处理、FastAPI、静态演示页
├── tests/                       # 手工小图、梯度、指标、缓存与接口一致性
├── docs/                        # 设计决策、实验说明、面试问答
├── data/                        # raw/interim/processed；不提交原始数据
└── runs/                        # 配置快照、日志、checkpoint、导出产物；大文件不提交
```

模块只做本层工作：数据处理不调用模型；构图不读取测试目标；模型不读取原始文件；Trainer 不自行改切分；服务不重新训练。CLI 负责组合这些模块。

## 2. 数据结构与产物

### 2.1 原始记录到监督样本

- `Interaction`：`user_id, parent_asin, timestamp_ms, original_row`。只从原始评论取这些字段，正文不进入商品特征。
- `Product`：`parent_asin, title, features, description, details, category, price`，来自商品元数据。
- `SessionExample`：`session_id, input_item_ids, target_item_id, start_ms, end_ms`。输入和标签分开保存；训练样本 ID 为 1..N，0 留作 padding/OOV。
- `EntityResult`：商品、文本哈希、提示词版本、模型名、实体类型/规范名/证据、关系、抽取状态和 token 用量。空实体集合与请求失败必须区分。
- `GraphArtifact`：`num_items, edge_index[2,E], edge_weight[E], graph_hash, catalog_hash, split_hash`。边按 `(src,dst)` 排序并合并，0 节点不参加可学习图。
- 训练目录按原始商品 ID 排序固定编号 1..N；分类输出候选只有 N 件，交叉熵标签等于 `target_item_id - 1`。未知目标在评估中使用独立标记，不能误送给交叉熵。

数据准备输出：商品目录、训练/验证/测试 Parquet、ID/类别映射、价格统计、训练热门榜、过滤统计及 manifest。manifest 包含原始文件哈希、时间边界、规则和样本数。后续产物必须匹配同一 manifest。

### 2.2 局部图批处理

批内每个会话单独建立商品节点，重复点击同一商品共享该会话内节点；不同会话即使点击同一商品，也使用不同的批内节点。图之间不连边，商品 embedding 仍共享。

| 字段 | 形状 | 含义 |
|---|---|---|
| `node_item_ids` | `[M+1]` | 第 0 个为虚拟零节点，1..M 为所有会话局部节点的商品 ID |
| `local_edge_index` | `[2,L]` | 批内有向相邻边，仅含有效节点；重复边合并 |
| `position_to_node` | `[B,T]` | 原始输入每个位置对应的批内节点；padding/OOV 指向虚拟节点 0 |
| `position_mask` | `[B,T]` | 该位置是否为已知有效点击，不能仅凭节点编号猜测 |
| `target_class` | `[B]` | 已知训练目标为 0..N-1；未知评估目标单独标记 |
| `has_known_history` | `[B]` | 是否至少有一次已知商品交互，用于统一热门降级 |

先对原始事件取最近的长度上限，再标记 OOV；仅原始相邻且两端已知的位置建边。例如 `[A, unknown, B]` 不能新建 `A→B`。重复物品节点合并仍可产生跨位置的共享，不能宣称 OOV 将图彻底分成互不相连的部分。

训练、离线评估与服务共用同一个序列转局部图函数。全未知输入直接走热门策略，不将全零向量假装成有兴趣的会话。局部聚合后强制虚拟节点为零，池化时同时使用 position mask。

## 3. 模型类与张量流

默认 `d=256, K=3, global_layers=2, local_layers=2`；B 为批量，N 为训练商品数，M 为批内局部节点数，E 为全局边数。

### 3.1 模块接口

| 类/组件 | 输入 → 输出 | 实现内容 |
|---|---|---|
| `ItemEmbedding` | 商品 ID → `[*,d]` | ID/类别/价格嵌入拼接后投影回 d 维；padding 输出为零 |
| `DirectedGraphEncoder` | `[N+1,d]` 和 COO 图 → `[N+1,d]` | 稀疏入边消息传递与独立自节点通路 |
| `AdaptivePruner` | 初始全商品表示、融合图 → `scores[E], gated_weight[E], regularizer` | 独立 prop/recv 图编码器、边点积 sigmoid、软门控 |
| `LocalSessionEncoder` | 批内节点初始表示、局部边 → `[M+1,d]` | 入邻居注意力聚合和自节点通路，层间非线性/dropout |
| `SessionReadout` | 局部/全局节点表示、位置映射和 mask → `[B,d]` | 按点击位置 gather，相加，再按有效位置平均 |
| `MultiViewHead` | `[B,d]` 和候选 ID embedding → `[B,N]` | K 个独立投影，逐候选最大视角分数 |
| `ESEAGP` | `SessionBatch, GraphArtifact` → `ModelOutput` | 组合模块，不访问文件或网络 |

`ModelOutput` 返回原始 `logits[B,N]`、图正则标量以及便于记录的门控均值/视角获胜数。诊断张量只在日志分支 detach，不能截断模型计算。

### 3.2 全局传播方向必须写清楚

边 `A[u,v]` 表示 `u→v`，固定基础权重按源节点出边和归一。每层接收聚合为：

```python
neighbor = sparse_mm(A.transpose(0, 1), H @ W_neighbor)
H_next = activation(neighbor + H @ W_self + bias)
```

可学习边值使用 PyTorch COO；构建后 coalesce，不把带梯度的值转成 NumPy/SciPy。prop 和 recv 参数独立，**它们最后一层使用有符号线性输出**。若两者最后都用 ReLU，点积会非负，门控全都不小于 0.5，使阈值 0.5 的硬剪枝失效。

独立自节点通路不属于剪枝候选集合；训练行为中真实的重复点击自转移可作为候选边保留。语义图排除商品与自己的语义连边。

### 3.3 剪枝和全局编码

```python
H0 = item_embedding(all_item_ids)
H_prop = prop_encoder(H0, fused_graph)
H_recv = recv_encoder(H0, fused_graph)
score = sigmoid((H_prop[src] * H_recv[dst]).sum(-1))
weighted_edges = base_edge_weight * score
H_global = global_encoder(H0, graph_with(weighted_edges))
sparsity_loss = abs(score.mean() - target_keep_score)
```

融合图先去重，每对唯一边只算一次门控。软门控后默认不再次按行归一，避免只有一个邻居时门控被除回 1；这是固定的工程选择。空图的正则返回零，节点仍能走自节点通路。

使用论文未缩放的点积起步，记录 score 饱和比例、节点范数和阈值下实际保留率；若需要缩放或归一修正，另开配置并记录变更，不悄悄改变公式。平均 score 正则与硬边保留率分别记录。

### 3.4 会话表示与多视角打分

```python
H_local = local_encoder(item_embedding(batch.node_item_ids), batch.local_edge_index)
H_global_local = H_global[batch.node_item_ids]
Z_node = H_local + H_global_local
Z_position = Z_node[batch.position_to_node] * batch.position_mask[..., None]
Z_session = Z_position.sum(1) / batch.position_mask.sum(1).clamp_min(1)[:, None]
Z_views = stack([expert(Z_session) for expert in experts], dim=1)  # [B,K,d]
logits = maximum_view_scores(Z_views, id_embedding[1:])          # [B,N]
```

expert 是线性层加 ReLU，K 个 expert 用 `ModuleList` 注册。该结构属于第三章多视角 max 融合，不加入第四章的动态 MoE 权重或熵约束。

先写直接计算 `[B,K,N]` 的参考版本验证数学；正式版本按候选商品维分块，例如每块 2048 件，每块沿 K 取 max，最后 `torch.cat` 为 `[B,N]`。只对拼接后的全部 logits 做一次交叉熵，不能逐块计算交叉熵再平均。分块版能否减少峰值显存由实际测试确认。

## 4. Trainer、实验与服务

### 4.1 训练循环

```python
for epoch in range(max_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad(set_to_none=True)
        output = model(batch, train_graph)  # 本步重新建立可微全局分支
        rec_loss = cross_entropy(output.logits, batch.target_class)
        loss = rec_loss + gamma * output.sparsity_loss
        loss.backward()
        optimizer.step()
    metrics = evaluate_on_validation(model, valid_loader)
    save_if_best_validation(metrics)
    stop_if_patience_reached()
```

- `CrossEntropyLoss` 直接接受原始 logits；训练前不要先 softmax。
- 训练全局分支每步前向，不能 detach 后跨 step 缓存。初版不使用梯度累积，先让梯度和耗时清楚可测。
- 验证在 `eval + no_grad` 下计算冻结全局表示，同一轮验证只算一次。缓存只由明确的评估/服务生命周期持有；optimizer 更新、checkpoint 加载、图变化或模式切换后必须重建。
- 每步记录主损失/正则，定期记录门控饱和、峰值显存和吞吐；每 epoch 记录完整验证指标与耗时。检查非有限 loss，停止并保存诊断，而非输出假结果。
- checkpoint 包含模型/优化器状态、epoch、随机状态及配置/数据/图指纹；载入时验证商品顺序。最终选择依据验证集，测试集只做最终报告。
- 正式指标同时报告全测试分母（OOV 目标计 miss）与已知目标子集，热门降级走同一候选集；OOV 目标不参与 CE。

### 4.2 消融只通过配置切换

- `full`：实体增强＋软剪枝＋K=3。
- `no_entity`：只用行为全局图，保留属性、剪枝和多视角，便于隔离语义边贡献。
- `no_pruning`：融合边直接使用基础权重，去掉剪枝模块及正则。
- `single_view`：K=1，其他部分不变。
- 独立的热门和相邻转移基线共用数据/评估器。字面硬剪枝仅作梯度和稀疏性诊断，不混入主结果。

全部配置输出同一格式的报告，由汇总命令读取日志生成比较表，禁止手工改好看的数字。

### 4.3 推理服务复用训练预处理

导出产物包含模型权重、商品映射、元数据展示字段、全局缓存、热门榜和指纹。`Predictor` 是离线/服务共用入口；`FastAPI` 只负责请求校验和结果序列化。

```json
POST /recommend
{"item_ids": ["ASIN_A", "ASIN_B"], "top_k": 10}

{"items": [{"item_id": "ASIN_C", "score": 1.23}],
 "source": "model", "model_version": "<checkpoint_hash>"}
```

- 参数 `top_k` 取 1..50，实际结果不超过候选数；scores 为排序分数。
- 最近输入截断、OOV 标记、相邻关系判断及重复点击映射与离线完全一致。全部输入未知或为空时 source 为 `popular_fallback`。
- 主服务先使用 CPU 和冻结全局缓存；局部图和全候选打分开销单独测量。`GET /health` 检查产物就绪与指纹，不触发训练。
- 极简 HTML 页面只维护当前浏览序列、展示推荐结果和来源，不增加登录、支付或购物车系统。

## 5. 实现次序与每步完成条件

| 次序 | 先写什么 | 完成条件 | 对应面试问题 |
|---|---|---|---|
| 1 | 配置、数据类型、小型手工数据、HR/MRR | 手算例子与程序一致 | 输入、标签和评估目标是什么？ |
| 2 | 数据清洗、时间切分、映射、Dataset/collate | 无泄漏；OOV 不新建跨位置边；重复点击正确 | 怎样构造会话，为什么这样切分？ |
| 3 | ItemEmbedding、局部图层、单视角头、Trainer | 小数据可过拟合，checkpoint 恢复一致 | 梯度从损失怎样到 embedding？ |
| 4 | 行为图、稀疏全局编码 | 手工方向测试通过，参数得到梯度 | 全局与局部有何不同？ |
| 5 | 实体适配器、校验缓存、语义图融合 | 抽取有证据、成本可追踪、图权重可核对 | 为什么需要 LLM，错误怎么控制？ |
| 6 | 双角色软剪枝、多视角 max | 梯度测试通过；分块与直接版一致 | 剪枝如何可微，多视角如何融合？ |
| 7 | 全量实验、核心消融、结果汇总 | 同协议报告所有结果，包含未提升情况 | 怎样证明每个模块有用？ |
| 8 | 导出、Predictor、API、演示 | 与离线分数一致；fallback 可复现；延迟有记录 | 如何上线、更新和降级？ |
| 9 | 文档、讲稿、两轮模拟 | 能独立画数据流、解释一次反向、修改一个模块 | 你的工作、取舍和局限是什么？ |

首个编码交付只完成 1–3 的可训练小闭环，再继续叠加；每个增量都带“改了什么、张量如何流动、如何验收、你需要口述什么”。

## 6. 必须保留的测试与实现边界

- 手工图验证 `A[src,dst]` 和 `A.T @ H` 方向、权重融合、孤立节点、重复商品及 padding。
- 改动测试标签后训练产物不变；验证/服务 OOV 流程相同，未知目标不误变为候选索引。
- 软门控仅用推荐损失即可给剪枝参数非零梯度；硬阈值的梯度缺失有明确诊断；有符号输出能产生小于 0.5 的门控。
- 小规模直接打分与候选分块版比较 logits、loss 和所有参数梯度，包含 max 并列样例；不能只比较最终 Top-K。
- 保存/加载、评估缓存、在线缓存与未缓存前向一致；改变 checkpoint/映射/图后旧缓存被拒用。
- 数据、模型和 API 细节补齐属于工程复现选择，不宣称作者原实现。第三章的软剪枝、时间切分、具体图层和实体流程差异写入决策记录。

实现依据：项目 `PLAN.md` 与第三章已核查公式。技术接口核对：

- PyTorch COO 稀疏矩阵乘法及反向：https://docs.pytorch.org/docs/2.9/generated/torch.sparse.mm.html
- 交叉熵输入 logits：https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html
- 模型子模块注册：https://docs.pytorch.org/docs/2.9/generated/torch.nn.Module.html

本设计不包含已完成的实验结论。实体服务供应商、URL、模型名与预算仍待补充，相关调用在配置确认后接入。
