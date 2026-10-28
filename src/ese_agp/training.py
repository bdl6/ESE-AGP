"""局部图基线的训练、评估与模型加载；候选目录由所选数据协议确定。

每轮打乱/前向/反向更新；有验证集时按验证 MRR 选模，论文两切分时固定轮数保存最后一轮。
本模块不读取原始交互、不重新切分数据；测试集仅由显式评估命令使用。
"""

import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import PreparedData, SessionExample
from .evaluation import evaluate_rankings
from .model import LocalSessionRecommender, collate_examples


def prepared_payload(data: PreparedData) -> dict:
    """把预处理 dataclass 转成可写 JSON 的字典，并附上数据格式版本号。"""
    return {"schema_version": 1, **asdict(data)}


def data_fingerprint(data: PreparedData) -> str:
    """对完整预处理产物生成 SHA256 指纹，绑定样本、目录编号、切分和规则清单。

    排序字典键并去掉非必要空白，使序列化格式变化不轻易改变指纹。
    样本序列的顺序仍然有意义，改变目录或数据后不应沿用旧 checkpoint。
    """
    # 真实数据身份由已校验的文件清单给出，避免展开全量 Arrow 样本重新序列化。
    if hasattr(data, "fingerprint"):
        return data.fingerprint
    raw = json.dumps(prepared_payload(data), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    """创建父目录并用 UTF-8 写可读 JSON；拒绝 NaN/Infinity，避免生成无效报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_prepared(path: str | Path) -> PreparedData:
    """读取 prepare 命令的 JSON，恢复三份 SessionExample 列表和 PreparedData。

    JSON 数组会读成 list，这里把输入前缀还原为 tuple，保持数据结构一致。
    """
    if Path(path).is_dir():
        from .artifacts import PreparedArtifact
        return PreparedArtifact(path)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    # 数据结构升级时应显式处理版本，不能静默把不兼容的文件当旧格式读取。
    if payload.pop("schema_version", None) != 1:
        raise ValueError("Unsupported prepared dataset schema")
    for split in ("train", "valid", "test"):
        payload[split] = [SessionExample(**{**row, "input_item_ids": tuple(row["input_item_ids"])}) for row in payload[split]]
    return PreparedData(**payload)


def choose_device(requested: str) -> torch.device:
    """把 auto/cpu/cuda 配置转换为 torch.device。

    auto 按 CUDA 实际可用性选择；显式要求 cuda 但不可用时直接报错，避免误用 CPU。
    """
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this venv")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)


@torch.no_grad()
def evaluate_model(model, examples, popularity, device, batch_size=64, k=20):
    """评估神经模型并返回全样本/已知目标的 HR、MRR、OOV 比例等。

    Args:
        model: LocalSessionRecommender，参数已放到 device。
        examples: 验证或测试样本，允许未知目标和全未知历史。
        popularity: 只从训练集得到的热门商品 ID 排名，供无有效历史时降级。
        device: 批数据张量所在设备。
        batch_size: 每次前向处理的会话数。
        k: 排名截断位置，默认 20；候选不足 k 时返回全部候选。
    Returns:
        指标字典；额外的 model_known_history_ce 仅统计“目标已知且历史有效”的样本。

    装饰器 no_grad 关闭梯度记录，model.eval 关闭训练态 dropout，二者作用不同。
    全未知历史使用热门排名，未知目标在整体 HR/MRR 的分母中保留并计 miss。
    """
    model.eval()
    rankings, targets = [], []
    total_loss, known_count = 0.0, 0
    for start in range(0, len(examples), batch_size):
        rows = examples[start:start + batch_size]
        batch = collate_examples(rows, device)
        logits = model(batch)
        # CE 只诊断神经模型实际适用的样本；热门降级样本与未知标签均不混入该损失。
        known = (batch.target_item_ids > 0) & batch.has_known_history
        if known.any():
            # 商品编号 1..N 转为类别下标 0..N-1；先累计样本 loss 总和，再统一除以数量。
            total_loss += F.cross_entropy(logits[known], batch.target_item_ids[known] - 1, reduction="sum").item()
            known_count += int(known.sum())
        # 候选列原本按商品 ID 升序，stable=True 使分数并列时保持 ID 升序；+1 还原商品 ID。
        ordered = (torch.argsort(logits, dim=1, descending=True, stable=True)[:, :k] + 1).cpu().tolist()
        for row, has_history in zip(ordered, batch.has_known_history.cpu().tolist()):
            # 全未知历史即使能算出 logits 也没有有效兴趣信号，最终排名使用训练热门榜。
            rankings.append(row[:k] if has_history else popularity[:k])
        targets.extend(ex.target_item_id for ex in rows)
    # 交给与基线共用的评估器，确保候选、未知目标分母和指标定义一致。
    result = evaluate_rankings(rankings, targets, k=k)
    result["hr@1"] = evaluate_rankings(rankings, targets, k=1)["hr@1"]
    result["model_known_history_ce"] = total_loss / known_count if known_count else None
    result["num_model_loss_examples"] = known_count
    return result


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
        torch.cuda.reset_peak_memory_stats(device)
    # 这是小图排错版本，限制 CPU 线程数，减少过大线程池的调度开销。
    torch.set_num_threads(min(4, torch.get_num_threads()))
    # 模型规模由训练目录确定；Adam 管理 embedding、图层和预测头的所有已注册参数。
    model = LocalSessionRecommender(len(data.catalog), **config["model"],
                                    item_features=getattr(data, "item_features", None)).to(device)
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
        "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                          for name in ("model.py", "features.py", "training.py", "evaluation.py", "artifacts.py", "amazon.py")},
        "determinism": "seeded; CUDA scatter/index reductions may be nondeterministic",
        "model_stage": "local_graph_single_view_with_attributes" if config["model"].get("use_item_features") else "local_graph_single_view_id_only",
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
                    "config": config, "selection": selection,
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
        "data_fingerprint": fingerprint, "elapsed_seconds": time.perf_counter() - beginning,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }
    write_json(run_dir / "report.json", report)
    return report


def load_model(path: str | Path, data: PreparedData, device: torch.device):
    """加载权重到指定设备并切换为评估状态，返回模型对象。

    先校验格式、商品编号和完整数据指纹，防止把旧权重配上另一套商品映射。
    这是推理加载入口；虽然保存了优化器/RNG，当前仍没有训练断点续跑命令。
    """
    # 先读到 CPU 再统一迁移设备；weights_only 限制反序列化对象种类。
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["schema_version"] != 1:
        raise ValueError("Unsupported checkpoint schema")
    if checkpoint["catalog"] != data.catalog or checkpoint["data_fingerprint"] != data_fingerprint(data):
        raise ValueError("Checkpoint and dataset/catalog fingerprints do not match")
    model = LocalSessionRecommender(checkpoint["num_items"], **checkpoint["model_config"],
                                    item_features=getattr(data, "item_features", None)).to(device)
    # 加载的是可学习参数；模型结构由保存的配置重新构造，返回前切为 eval。
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model
