"""稀疏全局 GNN：u→v 按源归一，v 接收入边消息，孤立节点保留自身通路。"""
import torch
from torch import nn
from torch.nn import functional as F


class DirectedGraphLayer(nn.Module):
    """不生成 E×d 消息张量，用 COO 稀疏乘法聚合所有商品。"""
    def __init__(self, dim, dropout):
        """邻居与自身参数独立，默认 ELU 与局部图保持一致。"""
        super().__init__()
        self.neighbor = nn.Linear(dim, dim, bias=False)
        self.self_path = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, incoming):
        # incoming 是全局行为图的稀疏邻接矩阵，形状为 [商品数 + 1, 商品数 + 1] h 是所有商品当前的向量表示，形状为 [商品数 + 1, 向量维度]
        """incoming[v,u]=w_uv；输入和输出均为 [N+1,d]。"""
        # 先对每个源节点的表示做线性变换，再按入边权重汇总到目标节点。
        neighbor_features = self.neighbor(h)
        neighbor_messages = torch.sparse.mm(incoming, neighbor_features)

        # 自身通路单独学习，保证没有入边的商品也能保留自身信息。
        self_messages = self.self_path(h)
        combined = neighbor_messages + self_messages
        activated = F.elu(combined)
        result = self.dropout(activated)

        # 商品 ID 0 留给 PAD；无论聚合和线性层产生什么值，都将其输出置零。
        non_padding_mask = torch.arange(len(h), device=h.device).ne(0)[:, None]
        return result * non_padding_mask


class DirectedGraphEncoder(nn.Module):
    """多层全图传播，每步优化均重新计算，保持 embedding 和图层的梯度。"""
    def __init__(self, dim, layers, dropout):
        """层数必须为正；每层最多增加一跳接收范围。"""
        super().__init__()
        if layers < 1: raise ValueError('global_layers must be positive')
        self.layers = nn.ModuleList(DirectedGraphLayer(dim, dropout) for _ in range(layers))

    def forward(self, h, edge_index, edge_weight):
        # h	[商品数 + 1, d]	所有商品当前的向量；第 0 行留给 PAD
        # edge_index	[2, 边数]	全局行为图的边；第一行是起点 ID，第二行是终点 ID
        # edge_weight	[边数]	每条边对应的权重，顺序与 edge_index 的列一一对应
        """翻转 COO 行列实现 A.T@H，不构造商品数平方稠密矩阵。"""
        incoming = torch.sparse_coo_tensor(edge_index.flip(0), edge_weight, (len(h), len(h))).coalesce()
        for layer in self.layers:
            h = layer(h, incoming)
        return h
