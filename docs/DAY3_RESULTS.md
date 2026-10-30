# Day03 正式实验结果

> 目录已按职责重组：模型在 `src/ese_agp/models/`，训练评估在 `src/ese_agp/training/`，数据在 `src/ese_agp/data/`。旧文件名对照与阅读顺序见 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)。历史报告中的源码名称对应当时版本。


同一论文随机9:1数据，seed=42，各固定20轮，保存最后一轮。单种子结果不代表统计显著。

|模型|HR@20|MRR@20|训练秒数|峰值分配显存MiB|
|---|---:|---:|---:|---:|
|day03_local_id|0.083556|0.039124|494.3|358.7|
|day03_global_id|0.099433|0.046417|1253.5|855.7|
|day03_attrs_legacy|0.077882|0.026571|510.8|365.2|
|day03_attrs_matched|0.089932|0.037479|514.7|365.2|

对照一仅比较 local_id/global_id；对照二仅比较 attrs_legacy/attrs_matched。不能跨数据协议比较旧数值。
所有数字来自 runs 对应目录；完整图模型、实体语义、剪枝、多视角仍未完成。
