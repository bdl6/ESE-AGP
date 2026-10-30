"""构造模型、校验数据与图身份并加载checkpoint。"""
from pathlib import Path
import torch
from ese_agp.data.sessions import PreparedData
from ese_agp.data.io import data_fingerprint
from ese_agp.models.recommender import LocalSessionRecommender

def create_model(data, config, device, expected_graph=None):
    """统一训练/加载入口，只有启用全局分支才读取并校验行为图。"""
    graph = None
    if config['model'].get('use_global'):
        from ese_agp.graphs.behavior import load_graph
        graph = load_graph(config['graph']['path'], data)
        if expected_graph is not None and graph['fingerprint'] != expected_graph:
            raise ValueError('Checkpoint graph fingerprint mismatch')
    return LocalSessionRecommender(len(data.catalog), **config['model'],
                                  item_features=getattr(data, 'item_features', None), graph=graph).to(device)


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
    if checkpoint['model_config'].get('use_global') and not checkpoint.get('graph_fingerprint'):
        raise ValueError('Global checkpoint missing graph fingerprint')
    model = create_model(data, {**checkpoint.get('config', {}), 'model': checkpoint['model_config']},
                         device, expected_graph=checkpoint.get('graph_fingerprint'))
    # 加载的是可学习参数；模型结构由保存的配置重新构造，返回前切为 eval。
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model
