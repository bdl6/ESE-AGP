import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="ESE-AGP stepwise reproduction (stage 1: local baseline)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check the active Python and a CUDA/CPU forward-backward pass")
    for name in ("prepare", "baseline", "train", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="configs/tiny.yaml")
        if name in {"baseline", "evaluate"}:
            command.add_argument("--split", choices=("valid", "test"), default="valid")
    args = parser.parse_args()
    import torch
    if args.command == "doctor":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(8, 8, device=device, requires_grad=True)
        (x @ x.T).square().mean().backward()
        result = {
            "python": sys.executable, "python_version": platform.python_version(),
            "venv_isolated": sys.prefix != sys.base_prefix,
            "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "backward_finite": bool(torch.isfinite(x.grad).all()),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    from .data import load_interactions_jsonl, prepare
    from .evaluation import PopularityBaseline, TransitionBaseline, evaluate_rankings
    from .training import choose_device, evaluate_model, load_model, load_prepared, prepared_payload, train, write_json
    config = load_config(args.config)
    cfg = config["data"]
    output = Path(cfg["output"])
    if args.command == "prepare":
        source = Path(cfg["input"])
        # Deliberately small-data reader. Amazon external sort/Parquet comes in the next increment.
        rows = load_interactions_jsonl(source)
        data = prepare(rows, **{key: value for key, value in cfg.items() if key not in {"input", "output"}})
        data.manifest.update({"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                              "source_name": source.name, "synthetic_debug": source.name == "tiny_interactions.jsonl"})
        write_json(output, prepared_payload(data))
        print(json.dumps(data.manifest, ensure_ascii=False, indent=2))
        return
    data = load_prepared(output)
    if args.command == "baseline":
        results = {}
        k = config["training"]["top_k"]
        for cls in (PopularityBaseline, TransitionBaseline):
            model = cls.fit(data.train, len(data.catalog))
            results[cls.__name__] = {}
            for split in (args.split,):
                examples = getattr(data, split)
                rankings = [model.recommend(ex.input_item_ids, k) for ex in examples]
                results[cls.__name__][split] = evaluate_rankings(rankings, [ex.target_item_id for ex in examples], k)
        write_json(Path(config["training"]["run_dir"]) / f"baselines_{args.split}.json", results)
    elif args.command == "train":
        results = train(data, config)
    else:
        device = choose_device(config["training"]["device"])
        model = load_model(Path(config["training"]["run_dir"]) / "best.pt", data, device)
        results = evaluate_model(model, getattr(data, args.split), data.popularity, device,
                                 config["training"]["batch_size"], config["training"]["top_k"])
        write_json(Path(config["training"]["run_dir"]) / f"evaluation_{args.split}.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
