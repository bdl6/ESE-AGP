"""第一阶段局部会话图基线：ID embedding → 入邻居注意力 → 位置平均 → 单视角打分。

Day02 可选融合商品 ID、类别和相对价格；全局图、实体语义与剪枝尚待接入。
形状约定：B=会话数，M=批内有效局部节点数，L=边数，T=最长前缀长度，
N=训练商品数，d=向量维度。0 号局部节点为整个批次共享的虚拟零节点。
"""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import SessionExample, prefix_to_local_graph
from .features import ItemAttributeEncoder


@dataclass
class SessionBatch:
    """一次模型前向需要的批数据，所有字段都是同设备的 PyTorch 张量。

    局部节点编号和商品 ID 是两套编号：不同会话中的同一商品有不同局部节点，
    但仍查询同一行商品 embedding。位置映射用于还原重复点击的位置。
    """
    # [M+1]：每个批内局部节点对应的商品 ID，第 0 项为 padding/OOV。
    node_item_ids: Tensor
    # [2,L]：第 0 行为 src、第 1 行为 dst，每列表示有向边 src→dst。
    edge_index: Tensor
    # [B,T]：每个原始点击位置对应的局部节点，未知和补齐位置都指向 0。
    position_to_node: Tensor
    # [B,T]：已知真实交互位置为 True，其余位置不参加平均池化。
    position_mask: Tensor
    # [B]：保存商品 ID，计算交叉熵时再减 1；0 是未知目标，不能参与 CE。
    target_item_ids: Tensor
    # [B]：是否至少有一次已知交互；全未知历史在评估器中降级为热门。
    has_known_history: Tensor


def collate_examples(examples: Sequence[SessionExample], device: torch.device) -> SessionBatch:
    """把非空样本列表拼成互不连通的会话图，并转换成指定 CPU/GPU 上的张量。

    Args:
        examples: 含输入前缀与下一商品标签的 SessionExample 列表。
        device: 输出张量的设备，必须与模型参数一致。
    Returns:
        SessionBatch；节点、边、位置映射、mask 和标签的形状见数据类字段注释。
    """
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    # 只用输入前缀构图，不能把预测目标加入当前会话的输入图。
    graph = prefix_to_local_graph([ex.input_item_ids for ex in examples])
    return SessionBatch(
        node_item_ids=torch.tensor(graph.node_item_ids, dtype=torch.long, device=device),
        # 即使没有边也保持 [2,0] 的形状，以便后续安全拆出 src 和 dst。
        edge_index=torch.tensor(graph.edge_index, dtype=torch.long, device=device).reshape(2, -1),
        position_to_node=torch.tensor(graph.position_to_node, dtype=torch.long, device=device),
        position_mask=torch.tensor(graph.position_mask, dtype=torch.bool, device=device),
        target_item_ids=torch.tensor([ex.target_item_id for ex in examples], dtype=torch.long, device=device),
        has_known_history=torch.tensor(graph.has_known_history, dtype=torch.bool, device=device),
    )


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


class LocalSessionRecommender(nn.Module):
    """局部图＋单视角的下一商品预测器，返回全部训练目录候选的原始分数。

    这是完整 ESE-AGP 之前的学习基线，先验证数据、图层、损失和优化器能否连通。
    """
    def __init__(self, num_items: int, dim: int = 32, layers: int = 2, dropout: float = 0.0,
                 use_item_features: bool = False, candidate_chunk_size: int = 2048, item_features=None):
        """创建局部图预测器。

        num_items 不含 0；dim 为向量维度，layers 为图层数，dropout 为训练置零比例。
        use_item_features 控制是否融合类别/价格，启用时 item_features 必须来自同一目录。
        candidate_chunk_size 控制一次点积的候选块大小，最终仍返回全目录 logits。
        """
        super().__init__()
        if num_items < 1 or dim < 1 or layers < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid model dimensions/dropout")
        if candidate_chunk_size < 1:
            raise ValueError("candidate_chunk_size must be positive")
        if use_item_features and item_features is None:
            raise ValueError("Item features are required by this model configuration")
        self.num_items = num_items
        self.candidate_chunk_size = candidate_chunk_size
        # 查找表形状 [N+1,d]；商品编号为 1..N，padding_idx 阻止查询第 0 行产生梯度。
        self.embedding = nn.Embedding(num_items + 1, dim, padding_idx=0)
        # ModuleList 注册每层参数，确保优化器和 state_dict 能找到它们。
        self.layers = nn.ModuleList(IncomingAttentionLayer(dim, dropout) for _ in range(layers))
        self.view = nn.Linear(dim, dim)
        nn.init.normal_(self.embedding.weight, std=0.05)
        # 手动正态初始化覆盖了整张表，因此把 padding 行恢复为零；初始化不需要求导。
        with torch.no_grad():
            self.embedding.weight[0].zero_()
        # 在共有参数初始化后创建属性层，方便相同种子下与 ID-only 对照。
        # 属性索引为 buffer，类别/价格 embedding 与投影权重则共同接受推荐梯度。
        self.attribute_encoder = ItemAttributeEncoder(dim, num_items, item_features) if use_item_features else None

    def forward(self, batch: SessionBatch) -> Tensor:
        """输入 SessionBatch，输出 logits[B,N]；第 j 列对应商品 ID j+1。"""
        # 1. 商品查表得到初始节点向量 [M+1,d]。
        h = self.embedding(batch.node_item_ids)
        if self.attribute_encoder is not None:
            h = self.attribute_encoder(batch.node_item_ids, h)
        # 2. 沿会话局部图逐层聚合，节点数量和向量维度保持不变。
        for layer in self.layers:
            h = layer(h, batch.edge_index)
        # 3. 按原点击位置取回 [B,T,d]；重复点击同一商品会多次取出同一节点。
        positions = h[batch.position_to_node] * batch.position_mask[..., None]
        # 分母为每个会话的有效点击次数 [B,1]；clamp 防止全未知会话除以零。
        counts = batch.position_mask.sum(1).clamp_min(1)[:, None]
        # 按位置平均得到会话表示 [B,d]，不能改成只对唯一商品节点平均。
        session = positions.sum(1) / counts
        # 4. 一个独立投影视角 [B,d]，后续再扩展论文的多视角 max。
        view = F.relu(self.view(session))
        # 5. 与候选 ID embedding 共享参数做点积：[B,d] @ [d,N] → [B,N]。
        # 排除 0 号候选；直接返回 raw logits，不提前 softmax，供交叉熵内部处理。
        # 分块沿候选维计算，再拼成全候选 logits；训练仍统一计算一次完整交叉熵。
        # 先统一 split，反向时统一拼接候选梯度，避免每个独立切片反复创建全表梯度。
        candidates = self.embedding.weight[1:].split(self.candidate_chunk_size, dim=0)
        return torch.cat([view @ block.T for block in candidates], dim=1)
