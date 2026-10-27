"""Small, inspectable full-catalog training loop for the local graph baseline."""

import hashlib
import json
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
    return {"schema_version": 1, **asdict(data)}


def data_fingerprint(data: PreparedData) -> str:
    raw = json.dumps(prepared_payload(data), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_prepared(path: str | Path) -> PreparedData:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.pop("schema_version", None) != 1:
        raise ValueError("Unsupported prepared dataset schema")
    for split in ("train", "valid", "test"):
        payload[split] = [SessionExample(**{**row, "input_item_ids": tuple(row["input_item_ids"])}) for row in payload[split]]
    return PreparedData(**payload)


def choose_device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this venv")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)


@torch.no_grad()
def evaluate_model(model, examples, popularity, device, batch_size=64, k=20):
    model.eval()
    rankings, targets = [], []
    total_loss, known_count = 0.0, 0
    for start in range(0, len(examples), batch_size):
        rows = examples[start:start + batch_size]
        batch = collate_examples(rows, device)
        logits = model(batch)
        known = (batch.target_item_ids > 0) & batch.has_known_history
        if known.any():
            total_loss += F.cross_entropy(logits[known], batch.target_item_ids[known] - 1, reduction="sum").item()
            known_count += int(known.sum())
        # Candidate columns are in ascending item-ID order; stable ties retain it.
        ordered = (torch.argsort(logits, dim=1, descending=True, stable=True) + 1).cpu().tolist()
        for row, has_history in zip(ordered, batch.has_known_history.cpu().tolist()):
            rankings.append(row[:k] if has_history else popularity[:k])
        targets.extend(ex.target_item_id for ex in rows)
    result = evaluate_rankings(rankings, targets, k=k)
    result["hr@1"] = evaluate_rankings(rankings, targets, k=1)["hr@1"]
    result["model_known_history_ce"] = total_loss / known_count if known_count else None
    result["num_model_loss_examples"] = known_count
    return result


def train(data: PreparedData, config: dict) -> dict:
    cfg = config["training"]
    if not data.train or not data.valid or not data.catalog:
        raise ValueError("Training needs nonempty train/valid and a nonempty training catalog")
    if any(ex.target_item_id <= 0 for ex in data.train):
        raise ValueError("Unknown training targets cannot enter cross entropy")
    batch_size, epochs, patience = cfg["batch_size"], cfg["epochs"], cfg["patience"]
    if min(batch_size, epochs, patience, cfg["top_k"]) < 1 or cfg["learning_rate"] <= 0:
        raise ValueError("Training counts and learning rate must be positive")
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device(cfg["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # Large default CPU thread pools are slower for these tiny hand-built graphs.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = LocalSessionRecommender(len(data.catalog), **config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = data_fingerprint(data)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", {
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": seed, "data_fingerprint": fingerprint,
        "determinism": "seeded; CUDA scatter/index reductions may be nondeterministic",
        "model_stage": "local_graph_single_view_id_only",
    })
    best_score, best_epoch, stale = -1.0, -1, 0
    beginning = time.perf_counter()
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(epochs):
            model.train()
            order = list(range(len(data.train)))
            random.Random(seed + epoch).shuffle(order)
            loss_sum = 0.0
            for start in range(0, len(order), batch_size):
                rows = [data.train[i] for i in order[start:start + batch_size]]
                batch = collate_examples(rows, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch)
                loss = F.cross_entropy(logits, batch.target_item_ids - 1)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss in epoch {epoch + 1}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                loss_sum += loss.item() * len(rows)
            metrics = evaluate_model(model, data.valid, data.popularity, device, batch_size, cfg["top_k"])
            record = {"epoch": epoch + 1, "train_ce": loss_sum / len(data.train), "valid": metrics,
                      "elapsed_seconds": time.perf_counter() - beginning}
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps(record, allow_nan=False), flush=True)
            score = metrics[f"mrr@{cfg['top_k']}"]
            if score > best_score:
                best_score, best_epoch, stale = score, epoch + 1, 0
                torch.save({
                    "schema_version": 1, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "epoch": best_epoch,
                    "model_config": config["model"], "num_items": len(data.catalog),
                    "data_fingerprint": fingerprint, "catalog": data.catalog,
                    "config": config, "best_valid_mrr": best_score,
                    "torch_rng_state": torch.get_rng_state(), "python_rng_state": random.getstate(),
                    "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                }, run_dir / "best.pt")
            else:
                stale += 1
            if stale >= patience:
                break
    # Reload the selected checkpoint; test is reserved for explicit evaluation.
    model = load_model(run_dir / "best.pt", data, device)
    report = {
        "stage": "synthetic_debug_local_baseline" if data.manifest.get("synthetic_debug") else "local_baseline",
        "best_epoch": best_epoch, "best_valid_mrr": best_score,
        "valid": evaluate_model(model, data.valid, data.popularity, device, batch_size, cfg["top_k"]),
        "data_fingerprint": fingerprint, "elapsed_seconds": time.perf_counter() - beginning,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }
    write_json(run_dir / "report.json", report)
    return report


def load_model(path: str | Path, data: PreparedData, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["schema_version"] != 1:
        raise ValueError("Unsupported checkpoint schema")
    if checkpoint["catalog"] != data.catalog or checkpoint["data_fingerprint"] != data_fingerprint(data):
        raise ValueError("Checkpoint and dataset/catalog fingerprints do not match")
    model = LocalSessionRecommender(checkpoint["num_items"], **checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model
