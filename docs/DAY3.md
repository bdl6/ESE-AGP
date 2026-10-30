# Day03：训练行为全局图与局部图融合

> 目录已按职责重组：模型在 `src/ese_agp/models/`，训练评估在 `src/ese_agp/training/`，数据在 `src/ese_agp/data/`。旧文件名对照与阅读顺序见 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)。历史报告中的源码名称对应当时版本。


日期：2026-10-29。主项目 `/data1/rag/zhy/ESE-AGP`，解释器 `.venv/bin/python`。

## 1. 今天新增了什么

- `graphs.py`：只从训练会话构建有向相邻行为图，统计次数并按源节点出边归一。
- `global_model.py`：两层稀疏全图传播，接收入邻居信息，加独立自身通路，ELU、dropout。
- `model.py`：`use_global` 开关；对应商品的全局与局部表示相加，仍按原点击位置平均池化。
- `training.py`：统一模型构造、图摘要绑定、GPU分配器显存上限；全局分支每个优化步重新计算。
- `features.py`：保留 legacy 初始化，新增 matched 属性表标准差0.05的独立对照。
- `entities.py`：100商品试抽取离线接口、提示词、分块、缓存键、证据校验；没有发起外部调用。
- `cli.py/config.py`：`build-graphs`、`prepare-entities`、可选graph配置与路径解析。
- `scripts/training/run_day03.py`：四组固定预算实验、代码/配置冻结校验、日志/失败状态、最终统一评估。

尚未实现实体语义图、融合图剪枝、多视角max和在线服务。当前不是完整ESE-AGP。

## 2. 手算行为图

训练完整序列 `[A,B,C]`、`[A,B,D]`、`[A,C]`：

|边|次数|源归一权重|
|---|---:|---:|
|A→B|2|2/3|
|A→C|1|1/3|
|B→C|1|1/2|
|B→D|1|1/2|

训练标签可作为训练序列最后一项参与全局图；每条当前会话的局部输入仍不含标签。
验证/测试序列不参与行为图。`[A,0,B]` 不补 `A→B`；连续重复 `[A,A]` 保留自转移。
图只遍历保留下来的训练样本，不等同于原始文件所有历史事件的图。

## 3. 张量与传播方向

`edge_index[2,E]` 第一行来源u、第二行接收v，`edge_weight[E]` 是源归一权重。
商品编号1..N，0为虚拟节点，显式保留孤立商品。

```python
# A[u,v]代表u→v；PyTorch乘法的行是接收节点，因此使用A.T。
incoming = sparse_coo_tensor(edge_index.flip(0), edge_weight, (N+1,N+1))
neighbor = sparse.mm(incoming, H @ W_neighbor)
H_next = dropout(ELU(neighbor + H @ W_self + bias))
```

全图输入输出为 `[N+1,256]`，稀疏矩阵只有E条边，不创建N平方稠密矩阵。
按源归一不保证接收节点的权重和为1，不要与局部图按接收节点归一的注意力混淆。

局部批有M个节点：全局输出用 `node_item_ids` 取出 `[M+1,256]`，与局部输出相加。
随后通过 `position_to_node[B,T]` 还原位置，按mask平均，再与候选ID表打分。
重复点击同一商品在池化中重复计数，未知和padding不计数。

## 4. 梯度与缓存

训练：每个batch重新执行全图传播，CE→会话表示→全局层→商品embedding。
全局图的基础计数权重固定，GNN参数与embedding可学习。本阶段没有可学习边门控。
不跨优化步缓存/分离全局表示，否则会改变训练算法。

评估：只有 `eval()` 且 `no_grad()` 才缓存全图表示。同一冻结模型的后续batch复用。
任何train/eval模式切换、迁移设备/精度、load_state_dict都会清缓存。
图buffer由模型创建时固定；变更图必须重新构造模型，不支持手工原地修改buffer。
图文件有完整摘要、数据摘要和目录大小校验；checkpoint保存图摘要，错图拒绝加载。

## 5. 四组实验为何这样分

|配置|作用|
|---|---|
|day03_local_id.yaml|仅ID局部图对照|
|day03_global_id.yaml|仅增加行为全局分支|
|day03_attrs_legacy.yaml|原属性初始化局部图|
|day03_attrs_matched.yaml|只统一两张属性表初始化std=0.05|

第一对检查全局分支，第二对检查初始化尺度。属性初始化不是与全局图同时改变。
相同seed下共有ID/局部层/投影初始参数已用测试核对。全局版参数更多，单次收益不能排除参数量影响。
全部使用论文协议当前产物：282311训练、31368测试、63493候选、491类别；固定20轮、seed42、batch100。
没有验证集，selection=last_epoch；不通过测试集选轮次。旧时间切分MRR不能与新协议混比。
本阶段自动执行器只在四组训练均成功后评估固定最终checkpoint，不在训练中查看测试指标。
若后续根据这些测试结果继续改架构，这个测试集就成为开发反馈，须披露并另留最终评估集。

## 6. 服务器命令

```bash
cd /data1/rag/zhy/ESE-AGP
.venv/bin/python -m pytest -q
.venv/bin/python -m ese_agp build-graphs --config configs/day03_global_id.yaml
# 离线准备，不发送请求；已有目录拒绝覆盖。
.venv/bin/python -m ese_agp prepare-entities --config configs/day03_global_id.yaml
# 单组训练：务必选择合适设备，run_dir不可复用旧实验。
CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m ese_agp train --config configs/day03_global_id.yaml
# 执行器已经启动，不要同时再运行！首次全新执行才用下行。
.venv/bin/python scripts/training/run_day03.py
```

运行状态：`runs/day03_jobs_status.json`；每轮记录：`runs/day03_*/metrics.jsonl`；完整输出：`runs/day03_*.log`。
执行器拒绝已有实验目录；当前训练没有断点续训命令。失败后保留证据，再为重跑显式新建目录/配置。
四组全部结束后生成 `docs/DAY3_RESULTS.md`；文件不存在表示尚未完成，不能引用预计指标。
代码快照 `runs/day03_source_snapshot` 与协议 `runs/day03_experiment_protocol.json` 用于复核。

## 7. 实体试抽取进度

`data/entities/pilot_6b03f00c6d5d/requests.jsonl`：100件训练中实际出现商品的文本与分块。
`audit_30.jsonl`：30条待人工审核空模板；`manifest.json`记录未配置状态及0次网络请求。
文本最多6000字符，2000字符分块、200字符重叠；归一只做Unicode/大小写/空白，不自动猜测艺人别名。
提示词要求JSON实体类型、名称和连续原文证据；格式/证据校验并不等同于实体语义准确率，仍需人工审核。
供应商、Base URL、模型名、预算/配额尚缺，故不能完成100件实际抽取、30件人工审核、成本估算或全量缓存。
密钥只在服务器环境变量中配置。取得服务配置后，先实现对应供应商适配器，再执行pilot，不直接批量抽取。

## 8. 面试练习

1. 局部图表达当前会话内的转移；全局图表达其他训练会话中累计的相邻转移，两者共享商品embedding。
2. 图统计只用训练数据，训练标签可参与训练图，测试标签绝不进入构图；当前输入图只含前缀。
3. 用同协议local_id/global_id检验全局分支，不能用不同数据切分的MRR证明收益。
4. 稀疏不是减少商品数，而是只存真实边；训练仍对全商品传播，计算开销需实测。
5. 推荐loss能更新全局参数，不代表一定提升MRR；初始化尺度只是待实验检验的解释。
6. 当前固定计数边不等于后续可学习剪枝；实体抽取仅完成离线准备，不能说已经接入LLM。

动手练习：在手工图增加B→A一次，重新算源归一权重，并指出一层传播中哪些接收节点会变化。
