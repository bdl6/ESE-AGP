# Day03 验收与运行记录

> 目录已按职责重组：模型在 `src/ese_agp/models/`，训练评估在 `src/ese_agp/training/`，数据在 `src/ese_agp/data/`。旧文件名对照与阅读顺序见 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)。历史报告中的源码名称对应当时版本。


日期2026-10-29。实现已完成，正式实验是否完成以机器状态为准；本记录不预填MRR。

## 已验收

- 新增行为图、全局GNN、可选局部/全局融合、缓存生命周期与图身份校验。
- 61项测试通过，包括原有数据协议、局部模型及新增全局/实体准备测试。
- 全训练数据图：63493商品，532021有向唯一边，606368相邻转移，57个孤立商品。
- 数据指纹：`6b03f00c6d5d47388513806084ecabef29f4089679b758fcb3175f17cfdf0479`。
- 图指纹：`5361a0e4f8259bdfa74b81dba3bc451954fab5aeee451f69d43df7c1a3b41a83`。
- 全目录256维两层全局模型实际完成5次CUDA优化，全部参数梯度有限；不代表完整训练结果。
- 参数量16846592；短测峰值分配显存830.6MiB；预热后约0.103秒/步。
- 相同冻结batch首次eval约0.0213秒，缓存后约0.0027秒；这是单batch短测，不是服务延迟或吞吐承诺。
- 粗估20轮约97分钟，不含完整数据取样、保存、额外负载变化；探测用共享GPU7，正式用共享GPU4/5，实际耗时以日志为准。
- 实体准备100件、审核空模板30件，网络调用0次，未完成实际抽取/人工审核。

## 正式对照执行

两组对照均固定20轮/seed42，同一论文9:1数据，不使用测试集早停。
GPU5执行global_id；GPU4依次执行local_id、attrs_legacy、attrs_matched。
PyTorch分配器显存比例上限分别0.10/0.06，不是整卡预留；不终止其他任务。
首次启动在显存限制API收到无编号cuda设备时失败，发生在模型训练之前；修复明确设备编号后重新启动，错误证据在 `runs/day03_startup_failure/`。

最新进度：`runs/day03_jobs_status.json`；四组成功后统一显式测试评估并生成 `docs/DAY3_RESULTS.md`。
未生成结果文件前不宣称全局图或初始化修订带来提升。测试结果用于后续设计反馈时须披露其开发用途。

## 未完成与限制

实体服务配置缺失，不能完成抽取质量审核/费用估算/批量缓存。语义图、剪枝、多视角与推理服务属于后续工作。
这是Amazon评论交互构造的代理会话，商品属性是快照；论文原始文件版本仍未知，规模与论文表不一致。
初始化对照仅改变属性表尺度；若无提升不能证明属性无用。单种子结果不能声称统计显著。

## 证据

`runs/day03_tests.log`、`runs/day03_resource_probe.json`、`runs/day03_experiment_protocol.json`、
`runs/day03_source_snapshot/`、`data/graphs/paper_behavior_v1.json`、
`data/entities/pilot_6b03f00c6d5d/manifest.json`。
