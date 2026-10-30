"""把会话输入拼成局部图批次，供训练和推理共同使用。"""
from dataclasses import dataclass
from typing import Sequence
import torch
from torch import Tensor
from ese_agp.data.sessions import SessionExample, prefix_to_local_graph

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
