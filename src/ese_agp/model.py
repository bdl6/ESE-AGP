"""Local graph baseline: ID embedding -> incoming attention -> mean -> one view.

The global graph, item attributes and adaptive pruning belong to later increments.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import SessionExample, prefix_to_local_graph


@dataclass
class SessionBatch:
    node_item_ids: Tensor
    edge_index: Tensor
    position_to_node: Tensor
    position_mask: Tensor
    target_item_ids: Tensor
    has_known_history: Tensor


def collate_examples(examples: Sequence[SessionExample], device: torch.device) -> SessionBatch:
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    graph = prefix_to_local_graph([ex.input_item_ids for ex in examples])
    return SessionBatch(
        node_item_ids=torch.tensor(graph.node_item_ids, dtype=torch.long, device=device),
        edge_index=torch.tensor(graph.edge_index, dtype=torch.long, device=device).reshape(2, -1),
        position_to_node=torch.tensor(graph.position_to_node, dtype=torch.long, device=device),
        position_mask=torch.tensor(graph.position_mask, dtype=torch.bool, device=device),
        target_item_ids=torch.tensor([ex.target_item_id for ex in examples], dtype=torch.long, device=device),
        has_known_history=torch.tensor(graph.has_known_history, dtype=torch.bool, device=device),
    )


class IncomingAttentionLayer(nn.Module):
    """Each destination receives a softmax-weighted sum of its incoming neighbors."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.message = nn.Linear(dim, dim, bias=False)
        self.self_path = nn.Linear(dim, dim)
        self.source_score = nn.Linear(dim, 1, bias=False)
        self.target_score = nn.Linear(dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index
        messages = self.message(h)
        incoming = torch.zeros_like(h)
        if src.numel():
            score = F.leaky_relu(
                self.source_score(messages[src]).squeeze(-1)
                + self.target_score(messages[dst]).squeeze(-1), negative_slope=0.2
            )
            # Stable segmented softmax, grouped by receiving (destination) node.
            maxima = torch.full((len(h),), -torch.inf, device=h.device, dtype=h.dtype)
            maxima.scatter_reduce_(0, dst, score.detach(), reduce="amax", include_self=True)
            exp_score = (score - maxima[dst]).exp()
            denominator = torch.zeros(len(h), device=h.device, dtype=h.dtype).index_add(0, dst, exp_score)
            attention = exp_score / denominator[dst].clamp_min(torch.finfo(h.dtype).tiny)
            incoming = incoming.index_add(0, dst, messages[src] * attention[:, None])
        updated = self.dropout(F.elu(self.self_path(h) + incoming))
        # Bias and activation must never turn the shared dummy node into a click.
        return updated * (torch.arange(len(h), device=h.device) != 0)[:, None]


class LocalSessionRecommender(nn.Module):
    def __init__(self, num_items: int, dim: int = 32, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        if num_items < 1 or dim < 1 or layers < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid model dimensions/dropout")
        self.num_items = num_items
        self.embedding = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.layers = nn.ModuleList(IncomingAttentionLayer(dim, dropout) for _ in range(layers))
        self.view = nn.Linear(dim, dim)
        nn.init.normal_(self.embedding.weight, std=0.05)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, batch: SessionBatch) -> Tensor:
        h = self.embedding(batch.node_item_ids)
        for layer in self.layers:
            h = layer(h, batch.edge_index)
        positions = h[batch.position_to_node] * batch.position_mask[..., None]
        counts = batch.position_mask.sum(1).clamp_min(1)[:, None]
        session = positions.sum(1) / counts
        view = F.relu(self.view(session))
        # Columns 0..N-1 correspond to catalog IDs 1..N. Return raw logits for CE.
        return view @ self.embedding.weight[1:].T
