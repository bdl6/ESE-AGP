# 第三章论文数据协议

依据论文 3.3.2（正文 29—31 页）：先全局过滤，再按会话随机 9:1 划分。
这是对论文明确规则的实现，并不保证复原作者未公开的随机划分和全部数据细节。

## 处理顺序

1. 用 parent_asin 对齐交互与元数据，类别取最后一个有效叶子，价格须为有效正数。
2. 删除元数据缺失、类别无效、价格无效的交互，不保留 OOV 占位。
3. 全局统计用户交互，保留至少 3 次交互的用户。
4. 以保留用户的首个有效交互为锚点，按固定 24 小时窗口切分，删除长度小于 2 的会话。
5. 在上述会话中全局统计商品频次，删除少于 5 次的商品，再删除长度不足 2 的会话。
6. 保留每个会话最近 20 次交互，最后一项作为标签，前面最多 19 项作为输入。
7. 用全部保留序列建立商品目录、类别与类别内相对价格的 50 桶特征。
8. 按 split_seed=42 的稳定 SHA256 排序实现会话随机置换，前 floor(0.9*N) 为训练，其余测试。
9. valid.parquet 保留空表以兼容产物格式；不偷偷从测试集取验证数据。

## 未公开细节与实现选择

- 论文称“迭代筛选”，但未给迭代停止条件。这里执行正文列明的一轮用户、商品、会话校验，未增加反复 k-core。
- 论文没有给随机种子/随机库。本实现记录 SHA256 会话置换和种子，不能声称与作者逐条样本相同。
- 重复事件保留；同时间戳按源文件行顺序；重复商品元数据取首条。这些延续现有实现。
- 目录内商品不一定在训练交互出现。manifest 分别统计 OOV 目标和 targets_unseen_in_train。
- 候选目录/属性统计使用全局数据，是论文协议选择；热门榜、转移统计和梯度训练仍只使用训练会话。
- 论文没有给单独验证集；当前训练 selection=last_epoch，固定 20 轮保存最终一轮，不用测试指标早停/选轮次。
- 本次只改数据协议和与两切分兼容的训练选择；模型仍为局部 GNN，未实现完整 ESE-AGP。
- 使用当前已下载的 Amazon Reviews 2023 全类目原始文件；论文的精确文件版本、过滤实现和最终样本清单仍需作者材料核对。

## 配置和命令

项目目录：`/data1/rag/zhy/ESE-AGP`，解释器：`.venv/bin/python`。

```bash
.venv/bin/python -m ese_agp prepare --config configs/amazon_cds.yaml
# 同一数据的 ID-only 配置，不需要再次 prepare。
CUDA_VISIBLE_DEVICES=<空闲GPU编号> .venv/bin/python -m ese_agp train --config configs/amazon_cds_id_only.yaml
CUDA_VISIBLE_DEVICES=<空闲GPU编号> .venv/bin/python -m ese_agp train --config configs/amazon_cds.yaml
# 固定方案训练结束后才显式评估测试集；没有验证集，不能使用默认 --split valid。
.venv/bin/python -m ese_agp baseline --config configs/amazon_cds.yaml --split test
CUDA_VISIBLE_DEVICES=<空闲GPU编号> .venv/bin/python -m ese_agp evaluate --config configs/amazon_cds.yaml --split test
```

新数据：`data/processed/amazon_cds_paper_v1/`。
新运行目录：`runs/amazon_cds_paper_local_attrs/`、`runs/amazon_cds_paper_local_id/`。
历史时间协议配置：`configs/amazon_cds_temporal.yaml`、`configs/amazon_cds_id_only_temporal.yaml`。
历史数据 `amazon_cds_v2` 和运行结果未覆盖。已有 checkpoint 不能用于新目录，加载器会拒绝指纹不匹配。

## 比较原则

只能在同一数据指纹、候选目录和划分内比较模型。论文表 3-2 的 CDs_and_Vinyl 为
47,767 个序列、11,268 个商品、361 类别；本次实际规模在生成后的 manifest 中报告。
新旧数据规模不一致不应通过随意抽样凑数。测试样本的完整性检查与指标评估是不同操作。


## 本次全量处理结果

| 项目 | 数量 |
|---|---:|
| 训练会话 | 282,311 |
| 测试会话 | 31,368 |
| 商品目录 | 63,493 |
| 类别 | 491 |
| 测试目标不在目录 | 0 |
| 测试目标对应商品未在训练交互出现 | 45（0.1435%） |
| 目录中未在训练交互出现的商品 | 57 |

处理耗时约 115.3 秒，数据指纹 `6b03f00c6d5d47388513806084ecabef29f4089679b758fcb3175f17cfdf0479`。
全部会话切分无重叠、无重复 session_id，输入长度 1—19，编号和特征表对齐。
49 项测试通过，真实训练样本的模型前反向有限；未启动新的正式训练，也未计算正式测试集推荐指标。

实际总会话数 313,679 与论文 47,767 不同；本实现遵循正文明确的处理规则，
不能声称严格复原论文数据。需进一步核对作者原始文件范围、预处理代码和“迭代筛选”的确切实现。
完整机器验收结果见 `runs/paper_protocol_verification.json`。
