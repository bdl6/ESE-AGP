"""命令行入口：只负责组合配置、数据、模型与评估模块，不在这里实现算法。

运行方式：python -m ese_agp doctor/prepare/baseline/train/evaluate。
默认配置为 configs/tiny.yaml，默认评估 valid，测试需显式指定 --split test。
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

from ese_agp.utils.config import load_config


def main():
    """解析子命令并调度执行，终端输出 JSON，训练/评估还会保存对应文件。

    doctor 检查解释器与前反向；prepare 产出处理数据；baseline 跑非神经基线；
    train 训练并保存最佳验证模型；evaluate 加载模型，对指定切分独立评估。
    """
    parser = argparse.ArgumentParser(description="ESE-AGP stepwise reproduction (stage 1: local baseline)")
    # 子命令存入 args.command，每个命令单独声明自己需要的参数。
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check the active Python and a CUDA/CPU forward-backward pass")
    for name in ("download", "prepare", "build-graphs", "prepare-entities", "baseline", "train", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="configs/amazon_cds.yaml" if name == "download" else "configs/tiny.yaml")
        # 默认只看验证集，避免日常调参反复查看测试结果。
        if name in {"baseline", "evaluate"}:
            command.add_argument("--split", choices=("valid", "test"), default="valid")
    args = parser.parse_args()
    # 放在参数解析之后导入，使 --help 不必加载较重的 PyTorch。
    import torch
    if args.command == "doctor":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 用一个很小的可求导矩阵做环境检查；这不是模型训练或吞吐基准测试。
        x = torch.randn(8, 8, device=device, requires_grad=True)
        # 形成一个标量损失再反传，检查所选设备是否能完成前向和有限梯度计算。
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
    from ese_agp.data.sessions import load_interactions_jsonl, prepare
    from ese_agp.models.baselines import PopularityBaseline, TransitionBaseline
    from ese_agp.training.metrics import evaluate_rankings
    from ese_agp.utils.device import choose_device
    from ese_agp.training.evaluator import evaluate_model
    from ese_agp.training.checkpoint import load_model
    from ese_agp.data.io import load_prepared, prepared_payload
    from ese_agp.training.trainer import train
    from ese_agp.utils.io import write_json
    # 统一入口解析路径；后续模块只接收处理好的配置和数据对象。
    config = load_config(args.config)
    cfg = config["data"]
    output = Path(cfg["output"])
    if args.command == "download":
        from ese_agp.data.download import download_file, URLS
        for key, filename in (("reviews", "CDs_and_Vinyl.jsonl.gz"), ("metadata", "meta_CDs_and_Vinyl.jsonl.gz")):
            download_file(URLS[filename], cfg[key])
        return
    if args.command == "prepare" and cfg.get("format") == "amazon2023":
        from ese_agp.data.amazon import prepare_amazon
        options = {key: value for key, value in cfg.items() if key not in {"format", "reviews", "metadata", "output"}}
        manifest = prepare_amazon(cfg["reviews"], cfg["metadata"], cfg["output"], **options)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    if args.command == "prepare":
        source = Path(cfg["input"])
        # 当前读取器针对小数据；Amazon 的外部排序与 Parquet 流程将在后续加入。
        rows = load_interactions_jsonl(source)
        # 文件路径不属于会话算法参数，其余 data 配置以关键字参数传给 prepare。
        data = prepare(rows, **{key: value for key, value in cfg.items() if key not in {"input", "output"}})
        # 保存原文件摘要/名字以追溯来源；tiny 文件明确标记为人工排错数据。
        data.manifest.update({"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                              "source_name": source.name, "synthetic_debug": source.name == "tiny_interactions.jsonl"})
        write_json(output, prepared_payload(data))
        print(json.dumps(data.manifest, ensure_ascii=False, indent=2))
        return
    # 除 prepare 外都复用同一份处理数据，不能各自重新切分/编号。
    data = load_prepared(output)
    if args.command in {"baseline", "evaluate"} and not getattr(data, args.split):
        # 论文 9:1 没有 valid，拒绝输出空集合的误导性指标；测试需用户显式指定。
        raise ValueError("Selected split is empty. Paper 9:1 has no valid split; use --split test only for final evaluation.")
    if args.command == "prepare-entities":
        from ese_agp.entities.extraction import prepare_entity_pilot
        results = prepare_entity_pilot(data, output.parent.parent / 'entities' / ('pilot_' + data.fingerprint[:12]))
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    if args.command == "build-graphs":
        from ese_agp.graphs.behavior import build_behavior_graph, save_graph, load_graph
        path = Path(config['graph']['path'])
        # 已有图须校验一致，不能悄悄覆盖另一次实验。
        graph = load_graph(path, data) if path.exists() else build_behavior_graph(data)
        if not path.exists(): save_graph(graph, path)
        print(json.dumps({k: v for k, v in graph.items() if k not in {'edge_index', 'edge_weight', 'counts'}}, indent=2))
        return
    if args.command == "baseline":
        results = {}
        k = config["training"]["top_k"]
        for cls in (PopularityBaseline, TransitionBaseline):
            # 热门/转移统计只拟合训练样本，验证或测试仅用于产生排名与计分。
            model = cls.fit(data.train, len(data.catalog))
            results[cls.__name__] = {}
            for split in (args.split,):
                examples = getattr(data, split)
                rankings = [model.recommend(ex.input_item_ids, k) for ex in examples]
                results[cls.__name__][split] = evaluate_rankings(rankings, [ex.target_item_id for ex in examples], k)
        write_json(Path(config["training"]["run_dir"]) / f"baselines_{args.split}.json", results)
    elif args.command == "train":
        # 训练细节封装在 training.train，这里不再实现一套训练循环。
        results = train(data, config)
    else:
        device = choose_device(config["training"]["device"])
        # 独立评估加载磁盘里的最佳模型，并校验它与当前数据指纹是否匹配。
        model = load_model(Path(config["training"]["run_dir"]) / "best.pt", data, device)
        results = evaluate_model(model, getattr(data, args.split), data.popularity, device,
                                 config["training"]["batch_size"], config["training"]["top_k"])
        write_json(Path(config["training"]["run_dir"]) / f"evaluation_{args.split}.json", results)
    # 统一输出可读 JSON；拒绝非有限数值，避免把异常结果当正常指标。
    print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
