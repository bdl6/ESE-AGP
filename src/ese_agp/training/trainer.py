"""推荐模型的训练循环、实验环境记录和checkpoint保存。"""
import hashlib
import json
import os
import random
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from ese_agp.data.sessions import PreparedData
from ese_agp.data.io import data_fingerprint
from ese_agp.data.batching import collate_examples
from ese_agp.utils.device import choose_device
from ese_agp.utils.io import write_json
from ese_agp.training.checkpoint import create_model, load_model
from ese_agp.training.evaluator import evaluate_model

def train(data: PreparedData, config: dict) -> dict:
    """训练局部图基线，按验证 MRR 或固定轮数最后一轮返回报告。

    Args:
        data: 已切分、过滤和编号的 PreparedData；训练目标必须为已知商品。
        config: 含 model/training/data 的配置字典，路径已经 load_config 解析。
    Returns:
        最佳轮次、验证指标、数据指纹、耗时及峰值显存等字典，同时写入 run_dir。

    validation_mrr: epochs 为最多轮数，patience 控制早停。
    last_epoch: 固定 epochs 轮，不使用测试集或早停；报告的 valid/best_valid_mrr 为 None。
    本函数不使用测试集选模型；同一 run_dir 会覆盖上次日志/模型，比较实验需换目录。
    """
    cfg = config["training"]
    # 论文 9:1 产物没有验证集，显式使用固定轮数的最后一轮；绝不以测试集选模型。
    selection = cfg.get("selection", "validation_mrr")
    if selection not in {"validation_mrr", "last_epoch"}:
        raise ValueError("selection must be validation_mrr or last_epoch")
    if not data.train or not data.catalog:
        raise ValueError("Training needs nonempty train and catalog")
    if selection == "validation_mrr" and not data.valid:
        raise ValueError("Validation selection needs nonempty valid; paper 9:1 uses selection=last_epoch")
    if any(ex.target_item_id <= 0 for ex in data.train):
        raise ValueError("Unknown training targets cannot enter cross entropy")
    batch_size, epochs, patience = cfg["batch_size"], cfg["epochs"], cfg["patience"]
    if min(batch_size, epochs, patience, cfg["top_k"]) < 1 or cfg["learning_rate"] <= 0:
        raise ValueError("Training counts and learning rate must be positive")
    seed = cfg["seed"]
    # 固定 Python/PyTorch 随机种子，方便复核；CUDA 的部分归约操作仍可能有微小波动。
    random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device(cfg["device"])
    if device.type == "cuda":
        if 'cuda_memory_fraction' in cfg:
            fraction = cfg['cuda_memory_fraction']
            if not 0 < fraction <= 1: raise ValueError('cuda_memory_fraction must be in (0,1]')
            torch.cuda.set_per_process_memory_fraction(fraction, device.index if device.index is not None else torch.cuda.current_device())
        torch.cuda.reset_peak_memory_stats(device)
    # 这是小图排错版本，限制 CPU 线程数，减少过大线程池的调度开销。
    torch.set_num_threads(min(4, torch.get_num_threads()))
    # 模型规模由训练目录确定；Adam 管理 embedding、图层和预测头的所有已注册参数。
    model = create_model(data, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    # 把本次配置、环境与数据身份落盘，让 checkpoint 和结果能追溯到同一份产物。
    fingerprint = data_fingerprint(data)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", {
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": seed, "data_fingerprint": fingerprint,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "source_sha256": {str(source.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(source.read_bytes()).hexdigest()
                          for source in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))},
        "determinism": "seeded; CUDA scatter/index reductions may be nondeterministic",
        "graph_fingerprint": model.graph_fingerprint,
        "model_stage": "local_plus_behavior_global" if config["model"].get("use_global") else "local_graph_single_view_with_attributes" if config["model"].get("use_item_features") else "local_graph_single_view_id_only",
    })
    # best_score 记录最好验证 MRR；stale 记录连续未提高轮数，供早停判断。
    best_score, best_epoch, stale = -1.0, -1, 0
    beginning = time.perf_counter()
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(epochs):
            epoch_started = time.perf_counter()
            # 验证函数会切到 eval，所以每个新 epoch 开始都需恢复训练态 dropout。
            model.train()
            order = list(range(len(data.train)))
            # 每轮用可复核的不同顺序打乱训练样本，不改变预处理产物本身。
            random.Random(seed + epoch).shuffle(order)
            loss_sum = 0.0
            for start in range(0, len(order), batch_size):
                rows = [data.train[i] for i in order[start:start + batch_size]]
                batch = collate_examples(rows, device)
                # 训练步骤 1：清掉上一批梯度；PyTorch 的梯度默认累积，否则会错误叠加。
                optimizer.zero_grad(set_to_none=True)
                # 训练步骤 2：前向得到 [B,N] raw logits，每行对应一个会话的全候选分数。
                logits = model(batch)
                # 训练步骤 3：计算全候选交叉熵；不用先 softmax，标签要从商品编号减 1。
                loss = F.cross_entropy(logits, batch.target_item_ids - 1)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss in epoch {epoch + 1}")
                # 训练步骤 4：链式法则反传，从 loss 到预测头、图层和 ID embedding。
                loss.backward()
                # 限制总梯度范数，缓解梯度爆炸；遇到非有限梯度直接报错，不保存假结果。
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                # 训练步骤 5：Adam 根据梯度更新参数。
                optimizer.step()
                # loss 默认按批平均，乘实际批大小再累计，防止最后一个小批次权重过高。
                loss_sum += loss.item() * len(rows)
            # 一轮训练结束，只在验证集计算排名指标；验证过程中不构建反向计算图。
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_seconds = time.perf_counter() - epoch_started
            metrics = evaluate_model(model, data.valid, data.popularity, device, batch_size, cfg["top_k"]) if data.valid else None
            record = {"epoch": epoch + 1, "train_ce": loss_sum / len(data.train), "valid": metrics,
                      "elapsed_seconds": time.perf_counter() - beginning,
                      "train_seconds": train_seconds, "train_examples_per_second": len(data.train)/train_seconds,
                      "epoch_seconds": time.perf_counter()-epoch_started}
            log.write(json.dumps(record, allow_nan=False) + "\n")
            # 每轮立即刷盘，运行中即可查看 metrics.jsonl，不必等待整个训练结束。
            log.flush()
            print(json.dumps(record, allow_nan=False), flush=True)
            score = epoch + 1 if selection == "last_epoch" else metrics[f"mrr@{cfg['top_k']}"]
            # validation_mrr 严格提高才保存；last_epoch 的 score 随轮次递增，因此每轮覆盖并最终保留最后一轮。
            if score > best_score:
                best_score, best_epoch, stale = score, epoch + 1, 0
                # 保存参数、优化器、随机状态与数据/配置身份；当前加载接口只恢复推理模型。
                torch.save({
                    "schema_version": 1, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "epoch": best_epoch,
                    "model_config": config["model"], "num_items": len(data.catalog),
                    "data_fingerprint": fingerprint, "catalog": data.catalog,
                    "config": config, "selection": selection, "graph_fingerprint": model.graph_fingerprint,
                    "best_valid_mrr": best_score if selection == "validation_mrr" else None,
                    "torch_rng_state": torch.get_rng_state(), "python_rng_state": random.getstate(),
                    "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                }, run_dir / "best.pt")
            else:
                stale += 1
            # 达到 patience 后早停；训练 loss 继续下降也不绕过验证集选择规则。
            if stale >= patience:
                break
    # 加载所选 checkpoint：验证最优或固定最后一轮；测试集始终只供显式评估。
    model = load_model(run_dir / "best.pt", data, device)
    report = {
        "stage": "synthetic_debug_local_baseline" if data.manifest.get("synthetic_debug") else "local_baseline",
        "best_epoch": best_epoch, "selection": selection,
        "best_valid_mrr": best_score if selection == "validation_mrr" else None,
        "valid": evaluate_model(model, data.valid, data.popularity, device, batch_size, cfg["top_k"]) if data.valid else None,
        "graph_fingerprint": model.graph_fingerprint,
        "data_fingerprint": fingerprint, "elapsed_seconds": time.perf_counter() - beginning,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }
    write_json(run_dir / "report.json", report)
    return report
