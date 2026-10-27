# ESE-AGP

第三章会话推荐模型的分阶段工程复现，同时用于学习 PyTorch 和准备推荐算法实习面试。

**当前是第一阶段：局部会话图基线。** 已实现小数据会话准备、时间切分、训练商品映射、热门/转移基线、局部图注意力模型、全候选交叉熵训练和 checkpoint 评估。商品属性、行为全局图、实体语义图、自适应剪枝和多视角 max 尚待后续实现。

`examples/tiny_interactions.jsonl` 是人工构造的调试数据，不能作为 Amazon、论文复现效果或业务收益的证据。当前读取器把小型 JSONL 装入内存；真实 Amazon 数据的流式下载、元数据清洗、外部排序和 Parquet 处理属于下一阶段。

## 环境

Windows / Python 3.12，项目环境使用标准库 `venv`，目录为 `.venv`。在项目根目录执行 PowerShell 命令：

```powershell
# 仅在尚无 .venv 时创建
python -m venv .venv

# 已创建环境可直接从这里开始，无需激活或更改系统执行策略
# 第一阶段先用 CPU 完成小数据验证
.\.venv\Scripts\python.exe -m pip install torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m ese_agp doctor
```

安装使用 [PyTorch 官方源](https://pytorch.org/get-started/previous-versions/)。本次 CUDA 12.6 大包下载持续停滞，第一阶段改用 CPU；GPU 环境验收仍未完成。`doctor` 检查实际解释器、环境隔离、CUDA 可用性以及一次矩阵前反向。包本身不依赖 CUDA，`device: auto` 会选择可用设备。

`requirements.lock` 记录本次实际安装的 CPU 环境依赖；重建时运行 `.\.venv\Scripts\python.exe -m pip install -r requirements.lock` 和 `.\.venv\Scripts\python.exe -m pip install -e . --no-deps`。它是本机 Python/Windows 环境快照，不承诺跨平台通用。

正式训练前安装 CUDA 版本并重新验收：`.\.venv\Scripts\python.exe -m pip install torch==2.9.0+cu126 --index-url https://download.pytorch.org/whl/cu126`，然后运行 `doctor`，要求 `cuda_available` 为 true。更换后应重新生成环境锁文件，不能继续套用 CPU 的 torch 锁定行。

## 跑通第一阶段

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ese_agp prepare --config configs/tiny.yaml
.\.venv\Scripts\python.exe -m ese_agp baseline --config configs/tiny.yaml
.\.venv\Scripts\python.exe -m ese_agp train --config configs/tiny.yaml
.\.venv\Scripts\python.exe -m ese_agp evaluate --config configs/tiny.yaml
```

训练和基线命令默认只看验证集。实验方案确定后，分别执行 `baseline --config configs/tiny.yaml --split test` 与 `evaluate --config configs/tiny.yaml --split test` 生成测试报告；人工调试数据可以完整走一遍，真实数据不要反复查看测试集调参。

`prepare` 输出 `data/processed/tiny.json`；基线结果、每轮日志、最佳模型及评估报告写入 `runs/tiny/`。数据与 checkpoint 绑定同一指纹；更换商品编号或切分后，旧 checkpoint 会被拒绝加载。目前支持保存和推理加载，尚无中断续训命令。重复运行训练会覆盖同名运行目录，保留实验时先修改配置中的 `run_dir`。

可选激活环境：`.\.venv\Scripts\Activate.ps1`。不激活时，继续使用上面的明确解释器路径即可。

## 数据与评估口径

输入每行包含 `user_id`、`item_id`、`timestamp`，时间戳单位为毫秒。按用户首次事件为原点划分固定 24 小时窗口，保留最近 20 次事件，最后一件是目标。时间切分边界由未过滤会话结束时间确定，跨界会话删除；用户/商品频次与目录只从训练期拟合。小样本配置阈值为 1，主实验计划的阈值仍为用户 3、商品 5。

商品目录编号 `1..N`，`0` 表示 padding/OOV。前缀 `[A, unknown, B]` 不会建立 `A→B`；重复商品共享会话内节点，池化时按实际点击位置重复计数。验证/测试未知目标计 miss，另报已知目标指标；全未知前缀走训练热门榜。候选为全部训练目录，允许推荐已见商品，并列按商品 ID 升序。

模型输出 `B×N` 原始 logits，标签是 `item_id - 1`，直接送入 [CrossEntropyLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html)。邻居注意力只聚合入边，使用 [scatter_reduce](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.scatter_reduce_.html) 实现稳定分组 softmax，另有独立自节点通路。

主指标为 HR@20/MRR@20，附带 HR@1、OOV 比例和样本数。人工数据候选很少，HR@20 很容易达到 1，不能据此证明模型强于基线；这里用于检查实现，泛化结论必须等待真实数据和消融实验。

## 阅读顺序

1. `src/ese_agp/data.py`：交互如何变成前缀、目标和局部图。
2. `src/ese_agp/evaluation.py`：先手算 HR/MRR，再看热门与转移策略。
3. `src/ese_agp/model.py`：节点 → 入邻居聚合 → 按位置平均 → 候选打分。
4. `src/ese_agp/training.py`：原始 logits、交叉熵、反向、更新、验证和保存。
5. `docs/DAY1.md`：第一阶段学习任务与面试问答。

完整范围、论文差异与后续顺序见 `PLAN.md` 和 `CODE_DESIGN.md`。本项目的可微软剪枝、时间评估协议和具体图层属于明确披露的复现选择；完整模型尚未完成，不使用论文原表数字充当本项目结果。
