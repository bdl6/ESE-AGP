"""训练行为图：相邻有向计数、源行归一、版本与数据指纹校验。"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch


def graph_fingerprint(graph):
    """摘要覆盖图规则、数据身份及全部边；排除摘要字段自身。"""
    body = {k: v for k, v in graph.items() if k != 'fingerprint'}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_behavior_graph(data):
    """只遍历训练序列（含训练标签），OOV 两端跳过且不跨位置补边。"""
    from ese_agp.data.io import data_fingerprint
    n = len(data.catalog)
    counts = Counter()
    for ex in data.train:
        sequence = (*ex.input_item_ids, ex.target_item_id)
        if any(not isinstance(i, int) or not 0 <= i <= n for i in sequence):
            raise ValueError('Invalid training item ID')
        counts.update((u, v) for u, v in zip(sequence, sequence[1:]) if u and v)
    outgoing = Counter()
    for (u, v), count in counts.items(): outgoing[u] += count
    edges = sorted(counts)
    graph = {'schema_version': 1, 'kind': 'train_adjacent_directed_source_normalized',
             'num_items': n, 'data_fingerprint': data_fingerprint(data),
             'edge_index': [[u for u, v in edges], [v for u, v in edges]],
             'counts': [counts[e] for e in edges],
             'edge_weight': [counts[u, v] / outgoing[u] for u, v in edges],
             'num_transitions': sum(counts.values()), 'num_edges': len(edges),
             'num_isolated': n - len({i for e in edges for i in e})}
    graph['fingerprint'] = graph_fingerprint(graph)
    return graph


def save_graph(graph, path):
    """写独立图产物；不可覆盖历史实验，必须为新图使用新文件名。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(graph, stream, ensure_ascii=False, allow_nan=False)


def load_graph(path, data):
    """拒绝文件篡改、目录错配、非法边和不一致的归一计数。"""
    from ese_agp.data.io import data_fingerprint
    graph = json.loads(Path(path).read_text())
    if graph.get('schema_version') != 1 or graph.get('fingerprint') != graph_fingerprint(graph):
        raise ValueError('Graph fingerprint mismatch')
    if graph['data_fingerprint'] != data_fingerprint(data) or graph['num_items'] != len(data.catalog):
        raise ValueError('Graph and dataset fingerprints do not match')
    src, dst = graph['edge_index']
    edges = list(zip(src, dst))
    if not len(src) == len(dst) == len(graph['counts']) == len(graph['edge_weight']) == graph['num_edges']:
        raise ValueError('Graph edge array lengths differ')
    if edges != sorted(set(edges)) or any(not 1 <= i <= graph['num_items'] for e in edges for i in e):
        raise ValueError('Invalid graph edge IDs/order')
    totals = Counter()
    for u, c in zip(src, graph['counts']):
        if not isinstance(c, int) or c < 1: raise ValueError('Invalid edge count')
        totals[u] += c
    for u, c, w in zip(src, graph['counts'], graph['edge_weight']):
        if abs(w - c/totals[u]) > 1e-8: raise ValueError('Graph weights not source-normalized')
    return graph


def graph_tensors(graph):
    """边编号保留 1..N，第 0 行为空；张量由模型 buffer 管理设备迁移。"""
    return (torch.tensor(graph['edge_index'], dtype=torch.long).reshape(2, -1),
            torch.tensor(graph['edge_weight'], dtype=torch.float32))
