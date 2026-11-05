"""论文 3.2.4 的自适应图剪枝，输入商品特征与融合图，输出剪枝图及结构损失。"""
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class PruningResult:
    edge_index: Tensor
    edge_weight: Tensor
    scores: Tensor
    keep_mask: Tensor
    structure_loss: Tensor


def normalized_incoming(edge_index: Tensor, edge_weight: Tensor, num_nodes: int) -> Tensor:
    """按式 (3-10) 加自环并归一化。矩阵行是接收者，度取加权入度。"""
    items = torch.arange(1, num_nodes, device=edge_index.device)
    self_loops = torch.stack((items, items))
    indices = torch.cat((edge_index.flip(0), self_loops), dim=1)
    weights = torch.cat((edge_weight, edge_weight.new_ones(len(items))))
    degree = weights.new_zeros(num_nodes).index_add(0, indices[0], weights)
    # 商品节点加自环后度至少为 1。PAD 无连边，保持零行。
    scale = degree.clamp_min(1).rsqrt()
    normalized = scale[indices[0]] * weights * scale[indices[1]]
    return torch.sparse_coo_tensor(indices, normalized, (num_nodes, num_nodes)).coalesce()


class PruningGCNLayer(nn.Linear):
    """单层图卷积，计算归一化邻接矩阵 @ (H @ W)。"""

    def __init__(self, dim: int):
        # 复用 Linear 的权重初始化和 weight 参数名，兼容已有 checkpoint。
        super().__init__(dim, dim, bias=False)

    def forward(self, features: Tensor, incoming: Tensor) -> Tensor:
        transformed = super().forward(features)
        return torch.sparse.mm(incoming, transformed)


class PruningGCN(nn.Module):
    """堆叠图卷积层，每层聚合之后应用 ReLU。"""

    def __init__(self, dim: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList(PruningGCNLayer(dim) for _ in range(layers))

    def forward(self, features: Tensor, incoming: Tensor) -> Tensor:
        for gcn in self.layers:
            features = gcn(features, incoming).relu()
        return features


class AdaptiveGraphPruner(nn.Module):
    """传播/接收分支独立学习。阈值与结构目标由调用方分别指定。"""

    def __init__(self, dim: int, *, threshold: float, rho: float, layers: int = 2):
        super().__init__()
        if dim < 1 or layers < 1:
            raise ValueError('dim and layers must be positive')
        if not (0 <= threshold <= 1 and 0 <= rho <= 1):
            raise ValueError('threshold and rho must be in [0, 1]')
        self.threshold = threshold
        self.rho = rho
        self.propagation = PruningGCN(dim, layers)
        self.reception = PruningGCN(dim, layers)

    def forward(self, features: Tensor, edge_index: Tensor, edge_weight: Tensor) -> PruningResult:
        """features 第 0 行是 PAD，edge_index 两行依次为源商品和目标商品。"""
        if features.ndim != 2 or len(features) < 1:
            raise ValueError('Expected features with shape [num_items + 1, dim]')
        if edge_weight.ndim != 1 or edge_index.shape != (2, len(edge_weight)) or edge_index.dtype != torch.long:
            raise ValueError('Expected long edge_index [2, E] and edge_weight [E]')
        if torch.any((edge_index < 1) | (edge_index >= len(features))):
            raise ValueError('Edges must reference real items, excluding padding')
        incoming = normalized_incoming(edge_index, edge_weight.to(features.dtype), len(features))
        prop = self.propagation(features, incoming)
        recv = self.reception(features, incoming)
        source, target = edge_index
        scores = (prop[source] * recv[target]).sum(dim=-1).sigmoid()
        keep = scores >= self.threshold
        # 结构正则读取筛选前的连续评分。空图没有可约束的边。
        structure_loss = (scores.mean() - self.rho).abs() if scores.numel() else scores.sum()
        return PruningResult(edge_index[:, keep], edge_weight[keep], scores, keep, structure_loss)
