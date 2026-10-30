# Day02：真实数据与商品属性

> 目录已按职责重组：模型在 `src/ese_agp/models/`，训练评估在 `src/ese_agp/training/`，数据在 `src/ese_agp/data/`。旧文件名对照与阅读顺序见 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)。历史报告中的源码名称对应当时版本。


当前主项目位于服务器 `61.162.193.19_rag:/data1/rag/zhy/ESE-AGP`，后续实现与测试直接在服务器完成。
本阶段只实现局部图基线与商品 ID/类别/相对价格融合；全局行为图、实体语义图、剪枝和多视角仍属于后续阶段。

## 1. 新增代码及阅读顺序

| 文件 | 职责 | 阅读时要理解的问题 |
|---|---|---|
| `src/ese_agp/download.py` | 官方原始文件断点/分段下载、区间校验与 gzip CRC | 如何确认没有下载缺块或混入另一个文件？ |
| `src/ese_agp/amazon.py` | DuckDB 外部处理、会话、时间切分、训练内过滤、Parquet 输出 | 哪一步可以读取验证/测试，哪一步只能读取训练？ |
| `src/ese_agp/artifacts.py` | 紧凑 Arrow 样本、目录/属性/文件指纹检查 | 为什么商品编号和特征不能随意替换？ |
| `src/ese_agp/features.py` | 价格解析、论文相对价格公式、类别/价格 embedding 与融合 | 离散索引和可学习参数有什么区别？ |
| `src/ese_agp/model.py` | 可选属性编码后接原局部 GNN；候选分块点积 | 预测输入融合属性，但候选仍用共享 ID embedding 打分 |
| `src/ese_agp/training.py` | 同一 Trainer 兼容小 JSON 与真实 Parquet 产物 | 为什么完整交叉熵不能按候选块各算一次再平均？ |
| `tests/test_amazon.py` | 真实 schema、缺失/OOV、无泄漏、梯度、保存加载 | 小型手工案例如何检验复杂流水线？ |

## 2. 数据协议

数据为 Amazon Reviews 2023 CDs_and_Vinyl 的官方原始 review/meta JSONL.GZ。交互字段仅为 user_id、parent_asin 和 timestamp（缺失时可使用 sort_timestamp）。它是评论记录构造的代理会话，不是实际浏览点击日志。

固定顺序如下：

1. 校验交互标识和正的毫秒时间戳，保留重复事件；同时间按原文件行序排序。
2. 按用户首次事件为原点切固定 24 小时窗口。原始会话至少两条，先不截断。
3. 用未过滤会话结束时间的 nearest-rank 分位点确定 80%/10%/10% 时间边界；跨界会话整段删除。并列时间与过滤会改变实际比例。
4. 仅训练会话拟合一次 user→item 频次过滤（用户至少 3 次、商品至少 5 次），不是迭代 k-core。商品还须有有效末级类别和正价格。
5. 按原始商品 ID 排序编号为 1..N，0 为 padding/OOV。保留末尾最多 20 次事件，含最后目标；输入最多 19 次。训练目标未知或前缀全未知的样本删除；验证测试保留以计 miss/走热门。
6. 缺元数据/价格/类别的商品不能进入训练目录，但不先删除原始交互位置。例如 [A,unknown,B] 仍为 [A,0,B]，不会被改成 [A,B] 并误连 A→B。训练目标被过滤时不向前换答案。
7. 类别编号与价格统计只对训练目录的唯一商品拟合；商品价格是元数据快照，不代表历史事件时的真实价格。

元数据 parent_asin 重复时取原文件首次记录。商品文本只保留 title/features/description/details；反馈列不进入特征，details 中的 review/rating/rank/bought 键也排除。文本留在 products.parquet 供后续实体抽取，当前不调用任何 LLM。

## 3. 价格公式与 embedding

对应论文第三章式 3-28..31：类内均值 mu、总体标准差 sigma，gamma=sigma*sqrt(3)/pi；f(p)=sigmoid((p-mu)/gamma)，再用 f(min)、f(max) 极差归一到 r，价格等级为 min(floor(r*50),49)。实现中再加 1 编码为 1..50，把 0 留给 padding。零方差/数值退化类目使用中间档 26。

训练目录唯一商品等权贡献价格统计，不按交互频次重复加权；不使用验证/测试商品价格重新估计。

```text
商品 ID → ID embedding ─────┐
类别 ID → 类别 embedding ───┼→ 拼接 [*,3d] → Linear → [*,d] → 局部 GNN
价格档 → 价格 embedding ────┘
```

商品到类别/价格的整数映射注册为 buffer，不求梯度；三张 embedding 表和投影权重随推荐交叉熵共同训练。投影带 bias，因此输出再次按 item_id!=0 置零。属性关闭时保持原 ID-only 模型，并使相同种子下共有参数初始化一致。

## 4. 产物与规模

`data/processed/amazon_cds_v2/` 中包含 train/valid/test.parquet、catalog.json、item_features.json、popularity.json、products.parquet、manifest.json。文件摘要和清单身份共同绑定 checkpoint。

原始大文件由 DuckDB 分块读取/外部处理；训练样本存为紧凑 Arrow 列，按需转成 SessionExample，不把原始评论/商品文本或全部样本展开成 Python 对象列表。当前预处理后的样本列仍驻留内存，不能称为完全流式训练；更大规模可再引入 row-group 缓存或 IterableDataset。

源文件记录完整 SHA256；输出目录在成功后才发布，拒绝覆盖已有产物。下载分段缓存和失败的 building 目录仅用于恢复/诊断，均由 Git 忽略。

## 5. 服务器运行命令

现有服务器 .venv 已可运行。依赖声明和 uv.lock 均已包含 DuckDB/PyArrow；如使用 uv 重建环境，选择 `--extra cu126 --extra dev`，CPU/cu126 互斥。已有环境不用重复创建或切换 Torch 构建。

```bash
cd /data1/rag/zhy/ESE-AGP
.venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m ese_agp doctor
.venv/bin/python -m ese_agp download --config configs/amazon_cds.yaml
.venv/bin/python -m ese_agp prepare --config configs/amazon_cds.yaml
.venv/bin/python -m ese_agp baseline --config configs/amazon_cds.yaml --split valid
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m ese_agp train --config configs/amazon_cds.yaml
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m ese_agp train --config configs/amazon_cds_id_only.yaml
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m ese_agp evaluate --config configs/amazon_cds.yaml --split valid
```

GPU 编号应先检查实际空闲情况。本次执行器会选择有足够空闲显存的卡并记录编号。已有 manifest 时跳过 prepare；要重建请改成新的 data.output。训练会覆盖同名 run_dir，复跑对比请使用新的运行目录。

本阶段只用验证集选模型和比较，真实测试集等最终实验方案确定后显式使用 `--split test`。属性版与 ID-only 使用同一份全量产物、维度、种子、优化器、最多轮数和早停规则；不保证属性版一定更好，结果如实记录在 DAY2_REPORT.md。

## 6. 今天你需要掌握的面试回答

- 为什么 parent_asin 既出现在交互文件又出现在元数据？因为它是关联商品的键，不是用户特征。
- 为什么 ID 和类别都做 embedding？ID 提供商品个体差异，类别让同类商品共享参数与监督信号。
- 为什么价格先分档？把类内相对价格转成离散输入；统计只能用训练商品。
- 为什么 OOV 不直接删除？删除中间事件会伪造相邻边；未知目标删除会改变评估分母。
- 为什么先跑热门/转移？它们提供同协议、可复核的效果参照，复杂度不自动带来收益。
- 为什么不用评论正文做商品语义？本项目只用商品侧描述，避免把行为后的反馈混进特征。
- 新商品没有 ID embedding 怎么办？当前目录固定，新商品映射为 OOV；不能因它有类别/价格就声称已经支持冷启动。

来源：[Amazon 官方说明](https://amazon-reviews-2023.github.io/)、[JSONL 字段](https://amazon-reviews-2023.github.io/data_loading/jsonl.html)、[DuckDB 1.4 JSON 读取](https://duckdb.org/docs/1.4/data/json/loading_json)。价格公式来自用户提供毕业论文第三章；数据时间协议、OOV 位置保留和具体模型层属于已披露的工程选择。

## 7. 一个真实训练样本的动手检查

样本来自 train 第 3 条，仅用于读代码，不用于展示模型效果。商品编号属于当前固定目录。

| 商品编号 | 原始商品 ID | 类别 | 价格 | 类别编号 | 价格档 |
|---:|---|---|---:|---:|---:|
| 5907 | B000002JWU | American Alternative | 11.32 | 20 | 13 |
| 6794 | B000002N74 | Progressive Rock | 11.65 | 337 | 6 |

```text
输入商品编号：[6794, 6794, 5907]
目标商品编号：5907；交叉熵类别下标：5906
node_item_ids：[0, 6794, 5907]
edge_index：([1, 1], [1, 2])
position_to_node：[[1, 1, 2]]
```

本例 T=3，M=2，N=49540。查 ID/类别/价格三张表各得到 [M+1,256]，拼接为 [M+1,768]，投影回 [M+1,256]，再进入两层局部图。按位置取回 [1,T,256] 后平均，得到 [1,256]，最终对全部候选输出 [1,N]。

练习：找出重复出现的商品对应哪个共享局部节点；检查目标是否出现在前缀（出现也允许再次推荐）；确认目标不会被额外添加进输入图。再把一个中间位置临时改为 0，在独立排错代码中观察它两侧的相邻边消失，勿改正式 Parquet 产物。
