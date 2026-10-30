"""局部会话图的入邻居注意力传播层。"""
import torch
from torch import Tensor, nn
from torch.nn import functional as F

class IncomingAttentionLayer(nn.Module):
    """一层有向图注意力：节点只接收自己的入邻居消息。

    对 u→v 的边计算权重，将变换后的 h_u 加权累加到 v，再加独立自身通路。
    输出为 dropout(ELU(自身通路 + 入邻居聚合))，不构造 M×M 稠密邻接矩阵。
    """

    def __init__(self, dim: int, dropout: float):
        """dim 为输入/输出维度，dropout 为训练时随机置零的比例。"""
        super().__init__()
        # 邻居消息与自身通路参数独立，孤立节点也能通过自身通路保留信息。
        self.message = nn.Linear(dim, dim, bias=False)
        self.self_path = nn.Linear(dim, dim)
        # 两个打分器分别读取来源和接收节点表示，组合成每条边的标量分数。
        self.source_score = nn.Linear(dim, 1, bias=False)
        self.target_score = nn.Linear(dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor, edge_index: Tensor) -> Tensor:
        """输入 h[M+1,d]、edge_index[2,L]，返回同形状的新节点表示。"""
        # src/dst 均为 [L]，表示本批局部节点索引，不是原始商品 ID。
        src, dst = edge_index
        # 每个节点的消息先变换一次得到 [M+1,d]，再按边索引读取。
        messages = self.message(h)
        incoming = torch.zeros_like(h)
        if src.numel():
            # score[L] 是未归一化注意力，归一分组依据接收节点 dst。
            score = F.leaky_relu(
                self.source_score(messages[src]).squeeze(-1)
                + self.target_score(messages[dst]).squeeze(-1), negative_slope=0.2
            )
            # 对每个接收节点分别做稳定 softmax：exp(e_uv-max_v)/sum_{x→v} exp(e_xv-max_v)。
            # 减去最大值避免 exp 溢出；只 detach 平移量，score 本身仍保留训练梯度。
            maxima = torch.full((len(h),), -torch.inf, device=h.device, dtype=h.dtype)
            maxima.scatter_reduce_(0, dst, score.detach(), reduce="amax", include_self=True)
            exp_score = (score - maxima[dst]).exp()
            denominator = torch.zeros(len(h), device=h.device, dtype=h.dtype).index_add(0, dst, exp_score)
            attention = exp_score / denominator[dst].clamp_min(torch.finfo(h.dtype).tiny)
            # 每条边先得到 [d] 维加权消息，再按 dst 累加；多个入邻居汇入同一节点。
            incoming = incoming.index_add(0, dst, messages[src] * attention[:, None])
        # 无入边时 incoming=0，该节点仍保留独立的 self_path。
        updated = self.dropout(F.elu(self.self_path(h) + incoming))
        # 带 bias 的线性层会使零输入产生非零输出，因此每层强制虚拟节点重新归零。
        return updated * (torch.arange(len(h), device=h.device) != 0)[:, None]

class LocalGraphEncoder(nn.ModuleList):
    """按顺序运行多层局部图注意力；保持旧 checkpoint 的 layers.* 参数名。"""

    def __init__(self, dim: int, layers: int, dropout: float):
        """每层独立创建参数；零层仅供隔离读出逻辑的测试使用。"""
        if layers < 0:
            raise ValueError("local layers must be nonnegative")
        super().__init__(IncomingAttentionLayer(dim, dropout) for _ in range(layers))

    def forward(self, h: Tensor, edge_index: Tensor) -> Tensor:
        """在同一批会话图上逐层传播，返回 [M+1,d] 节点表示。"""
        for layer in self:
            h = layer(h, edge_index)
        return h
