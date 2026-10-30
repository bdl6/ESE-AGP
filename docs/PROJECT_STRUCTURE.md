# 项目目录与代码阅读顺序

主项目：`/data1/rag/zhy/ESE-AGP`。本次只调整模块组织与记录源码的路径，不改变模型计算、训练超参数和数据协议。

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

## 原文件现在在哪里

|原文件|现在的位置（相对 src/ese_agp）|
|---|---|
|model.py|models/recommender.py；图层在 models/local_graph.py；批数据在 data/batching.py|
|global_model.py|models/global_graph.py|
|features.py|固定统计在 data/features.py；可训练编码在 models/embeddings.py|
|training.py|training/trainer.py、evaluator.py、checkpoint.py；数据加载拆到 data/io.py|
|evaluation.py|training/metrics.py；基线模型拆到 models/baselines.py|
|data.py|data/sessions.py|
|amazon.py / artifacts.py / download.py|data/ 对应文件|
|graphs.py|graphs/behavior.py|
|entities.py|entities/extraction.py|
|config.py|utils/config.py|
|scripts/run_day03.py|scripts/training/run_day03.py|

## 推荐阅读顺序

1. `data/sessions.py`：输入前缀、标签、局部图各是什么。
2. `data/batching.py`：把多个会话转换为模型张量。
3. `models/embeddings.py`：属性如何变成商品表示。
4. `models/local_graph.py`、`models/global_graph.py`：两类消息传递。
5. `models/recommender.py`：组合模块，得到候选商品分数。
6. `graphs/behavior.py`：历史行为图从哪里来。
7. `training/trainer.py`：一次前向、反向与优化。
8. `training/evaluator.py`、`metrics.py`、`checkpoint.py`：评估与加载。
9. `entities/extraction.py`：实体抽取准备，目前没有实际服务调用。

## 命令与历史产物

统一CLI仍为 `.venv/bin/python -m ese_agp ...`，所有现有 YAML 路径不变。
直接运行下载模块改为 `.venv/bin/python -m ese_agp.data.download --help`。
测试仍为 `.venv/bin/python -m pytest -q`；可单独执行 `tests/models/` 等目录。
Day03 执行脚本的新路径为 `scripts/training/run_day03.py`；它仍拒绝覆盖已有运行目录，不要因目录重构重新启动已完成的训练。
旧的内部导入如 `ese_agp.model` 已迁移，个人脚本需按上表更新；类名、参数名和state_dict键保持不变。
已有权重、实验报告、配置、数据和原始源码快照保留原路径。六组历史权重的全候选输出已与重构前逐元素比较。
历史实验的源码摘要描述当时的代码，不改写这些记录；新训练自动记录当前目录下全部Python源码。
历史Day03冻结协议仍为 `runs/day03_experiment_protocol.json`；新入口使用单独的 `runs/day03_refactored_protocol.json`。
验收证据：`runs/layout_tests.log`、`runs/layout_verification.json`；变更前备份路径见 `runs/layout_backup.json`。
