# 第一阶段验收记录

日期：2026-10-27。项目：`E:\ese-agp`。本记录属于人工数据排错验证，不是论文复现效果或正式实验结论。

## 已验证

- Python：`E:\ese-agp\.venv\Scripts\python.exe`，3.12.4，`sys.prefix != sys.base_prefix`。
- PyTorch：2.9.0+cpu；CPU 矩阵前反向梯度有限。CUDA runtime 为空，CUDA unavailable，GPU 环境尚未验收。
- `python -m pytest -q`：**30 passed in 9.60s**。覆盖会话窗口、跨界删除、训练内过滤、OOV 不桥接、局部图隔离、重复点击、自转移、手算指标、基线训练范围、入边方向、多邻居注意力梯度、dummy 零梯度、小数据过拟合、checkpoint 分数一致与指纹校验。
- 已实际运行 `doctor → prepare → baseline → train → evaluate`，另显式执行人工数据的 `baseline/evaluate --split test`。
- 依赖和项目代码均装在 venv；`.venv`、处理数据、日志和 checkpoint 由 Git 忽略。

## 人工数据运行证据

输入为 `examples/tiny_interactions.jsonl`：60 条事件、20 个会话、3 件商品，训练/验证/测试为 16/2/2。种子 42，局部图 32 维、2 层、单视角、batch 16、Adam 学习率 0.01；验证 MRR@20 在第 7 轮首次达到 1，连续 8 轮未提升后于第 15 轮停止，选中第 7 轮 checkpoint。

| 调试测试集 | HR@20 | MRR@20 |
|---|---:|---:|
| 热门基线 | 1.0000 | 0.6667 |
| 转移基线 | 1.0000 | 1.0000 |
| 局部图基线 | 1.0000 | 1.0000 |

只有 3 个候选和 2 个测试样本，这张表仅检查链路，不能用于说明泛化提升；转移基线已经达到相同排名结果。训练 loss 从约 1.1004 降至 0.0250，不代表选中的最佳模型就是最后一轮。

数据指纹：`05486ad058c10011abc321c0a30b34a3a9d2511a1a14c192a69153ee8db4b906`。原始人工文件 SHA256：`8eba4d478ac452c348754d79813b3037dbb320776de1d10b1bf8eb3fba31e4b8`。

原始证据在本机 `runs/tiny/`：`metrics.jsonl`、`best.pt`、`report.json`、`environment.json`、`config.json`、`baselines_valid.json`、`baselines_test.json`、`evaluation_valid.json`、`evaluation_test.json`；处理数据在 `data/processed/tiny.json`。这些可重新生成的文件不提交 Git。

## 仍需完成

1. GPU：官方 CUDA 12.6 wheel 下载持续停在 24MB，已停止；当前 CPU 环境可用于学习与小数据排错。正式图模型训练前需安装 CUDA wheel，检查实际 GPU 前反向、显存与吞吐，并重新锁定依赖。
2. 数据：真实 Amazon 下载、元数据清洗、外部排序/Parquet、商品类别与价格特征尚未实现；当前读取器仅适用于小型 JSONL。
3. 模型：行为全局图、实体语义图、软剪枝、多视角 max、正式消融和推理 API 尚未实现。
4. 工程：当前保存 optimizer/RNG 状态，但只有推理加载入口，尚无中断续训命令。

后续每完成一阶段追加真实验收记录，不以这份人工数据记录替代全量结果。
