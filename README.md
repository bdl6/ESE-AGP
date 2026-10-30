> 当前 `configs/amazon_cds.yaml` / `amazon_cds_id_only.yaml` 已切换为第三章论文的全局过滤、随机会话 9:1 协议。

> 目录已按职责重组：模型在 `src/ese_agp/models/`，训练评估在 `src/ese_agp/training/`，数据在 `src/ese_agp/data/`。旧文件名对照与阅读顺序见 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)。历史报告中的源码名称对应当时版本。


> 2026-10-29 Day03 更新：训练行为图、可选稀疏全局 GNN、属性初始化对照和实体试抽取离线接口已实现。61项测试通过。主配置现为论文随机9:1协议，历史时间协议结果不可混比。正式实验进度见 `runs/day03_jobs_status.json`；讲解见 `docs/DAY3.md`。实体服务尚未配置，未执行抽取、人工审核或付费调用。下文早期设计/结果保留为历史记录。

> 旧时间协议使用 `configs/amazon_cds_temporal.yaml` / `amazon_cds_id_only_temporal.yaml`；历史数据和结果保留。
> 新协议没有独立验证集，固定轮数保存最后一轮；详细步骤与实现选择见 [论文数据协议](docs/PAPER_DATA_PROTOCOL.md)。

# ESE-AGP

融合实体语义增强与自适应图剪枝的会话推荐项目，分阶段复现并配套代码学习与推荐算法实习面试练习。

**当前主项目：服务器 `61.162.193.19_rag:/data1/rag/zhy/ESE-AGP`。** Day02 已实现真实 Amazon 数据适配、Parquet 产物和 ID/类别/相对价格融合，43 项测试通过，CUDA 前反向通过。已完成完整类目下载/预处理、两组各20轮训练及独立验证；真实结果见 `docs/DAY2_REPORT.md`。

当前模型为局部会话图＋单视角基线。全局行为图、实体语义图、可微剪枝、多视角 max 和推理 API 仍待后续实现。`examples/tiny_interactions.jsonl` 仅为人工排错数据；正式输入来自 Amazon Reviews 2023 CDs_and_Vinyl 的官方原始 review/meta 文件。

## 环境

服务器使用项目 `.venv`：Python 3.12.13、PyTorch 2.9.0+cu126。执行命令统一调用 `.venv/bin/python`，不必激活环境。依赖在 `pyproject.toml` 声明、`uv.lock` 锁定，保留 CPU/CUDA 互斥 extra；如使用 uv 重建环境，运行 `uv sync --locked --extra cu126 --extra dev`。

```bash
cd /data1/rag/zhy/ESE-AGP
.venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m ese_agp doctor
```

运行前按实际空闲显存选择 GPU。doctor 检查解释器、CUDA 可用性与一次矩阵前反向；本次另验证了 256 维属性局部 GNN 的 CUDA 梯度。

## 真实数据流程

```bash
.venv/bin/python -m ese_agp download --config configs/amazon_cds.yaml
.venv/bin/python -m ese_agp prepare --config configs/amazon_cds.yaml
.venv/bin/python -m ese_agp baseline --config configs/amazon_cds.yaml --split valid
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m ese_agp train --config configs/amazon_cds.yaml
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m ese_agp train --config configs/amazon_cds_id_only.yaml
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m ese_agp evaluate --config configs/amazon_cds.yaml --split valid
```

已有 `data/processed/amazon_cds_v2/manifest.json` 时跳过 prepare；要重建请改 `data.output` 为新目录。训练会覆盖同名 run_dir，保留实验时先修改运行目录。数据/模型/日志均由 Git 忽略。

完整下载支持连续前缀续传、分段区间/ETag 校验与 gzip CRC。预处理用 DuckDB 限制内存、落盘运算；输出 Parquet 样本，以及商品目录、属性索引、热门榜、商品安全文本、统计和文件指纹。训练样本列使用紧凑 Arrow 数据，按需生成样本对象；原始评论和元数据文本不会整体展开进 Python 列表。样本列当前仍驻留内存，更大规模需进一步扩展读取策略。

## 数据与模型口径

- Amazon 是评论/评分记录，由它构造的是代理会话，不是实际点击日志；商品元数据/价格是快照。
- 以用户首次事件锚定固定 24 小时窗口，至少两次交互；先按原始会话结束时间确定 80%/10%/10% 分位边界，删除跨界会话，再做训练内过滤。
- 用户至少 3 次、商品至少 5 次。商品须有有效末级类别与正价格。目录和类别/价格统计只从训练期拟合，不使用评估商品或反馈列拟合特征。
- 最近最多 20 次含目标，输入最多 19 次。未知交互保留为 0，不补连未知位置两侧；未知评估目标计入全分母 miss，全未知历史走训练热门。
- ID/类别/类内相对价格档位各自查 embedding，拼接投影后输入局部 GNN。价格使用论文式 3-28..31，档位平移到 1..50，0 预留。
- GNN 聚合入邻居并有独立自节点通路；按原点击位置平均，重复商品多次参与池化。候选为全部训练目录，允许已见商品再次推荐，并列按 ID 升序。
- 候选点积分块后拼成全 logits，只计算一次完整交叉熵，不逐块归一化。评价只把 Top-K 排名传回 CPU。
- 主指标 HR@20/MRR@20，另报已知目标子集和 OOV 比例。日常比较使用验证集，测试集只在最终方案确定后显式 `--split test`。
- checkpoint 绑定完整数据/目录/属性身份，支持推理加载；尚无训练断点续跑命令。

## 学习顺序

1. `docs/DAY1.md`：前缀、标签、局部图与训练五步骤。
2. `docs/DAY2.md`：真实数据协议、商品特征、运行步骤和面试问答。
3. `src/ese_agp/data.py` / `evaluation.py`：小数据参考实现、统一指标和基线。
4. `src/ese_agp/features.py` / `model.py`：属性查表、融合、局部图与候选打分。
5. `src/ese_agp/amazon.py` / `artifacts.py`：大文件处理、时间切分、Parquet 与指纹。
6. `src/ese_agp/training.py`：训练、验证、早停、吞吐与保存加载。

代码、配置和测试均有中文说明。完整目标及与论文的差异见 `PLAN.md`、`CODE_DESIGN.md`；软剪枝等后续设计不能当作已经实现的成果。

数据来源：[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)；[原始字段说明](https://amazon-reviews-2023.github.io/data_loading/jsonl.html)。本项目的时间协议与论文随机切分不同，不直接拿论文原表数字排名。

## 当前源码结构

```text
src/ese_agp/
├── cli.py / __main__.py          # 统一命令入口
├── models/                      # 所有推荐模型和可训练图层
│   ├── recommender.py           # LocalSessionRecommender：组合各模块
│   ├── local_graph.py           # 局部图注意力层
│   ├── global_graph.py          # 全局稀疏图层与编码器
│   ├── embeddings.py            # 商品属性 embedding 与投影
│   └── baselines.py             # 热门、相邻转移基线
├── training/                    # 训练与评估
│   ├── trainer.py               # 训练循环、保存实验记录
│   ├── evaluator.py             # 神经模型批量评估
│   ├── metrics.py               # HR/MRR 计算
│   └── checkpoint.py            # 模型构造、校验与加载
├── data/                        # 数据处理
│   ├── sessions.py              # 数据结构、会话划分、局部构图
│   ├── batching.py              # SessionBatch、collate_examples
│   ├── amazon.py                # Amazon 真实数据预处理
│   ├── artifacts.py             # Parquet 产物与目录校验
│   ├── features.py              # 类别/价格统计与分档
│   ├── io.py                    # 预处理数据加载与指纹
│   └── download.py              # 官方数据下载
├── graphs/
│   └── behavior.py              # 行为全局图构建、保存与校验
├── entities/
│   └── extraction.py            # 实体抽取准备与结果校验
└── utils/
    ├── config.py                # YAML 配置、路径解析
    ├── device.py                # 运行设备选择
    ├── io.py                    # JSON 写入
    └── validation.py            # 输入编号校验

scripts/training/run_day03.py     # 四组对照实验执行脚本
tests/{data,models,graphs,entities,training}/
configs/                         # 现有配置路径不变
docs/                            # 学习讲义与实验报告
data/ / runs/                    # 数据、权重、日志产物
```
