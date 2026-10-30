"""推荐模型：组合商品embedding、局部/可选全局图与单视角打分。"""
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ese_agp.data.batching import SessionBatch
from ese_agp.models.local_graph import LocalGraphEncoder
from ese_agp.models.embeddings import ItemAttributeEncoder

class LocalSessionRecommender(nn.Module):
    """局部图＋单视角的下一商品预测器，返回全部训练目录候选的原始分数。

    这是完整 ESE-AGP 之前的学习基线，先验证数据、图层、损失和优化器能否连通。
    """
    def __init__(self, num_items: int, dim: int = 32, layers: int = 2, dropout: float = 0.0,
                 use_item_features: bool = False, candidate_chunk_size: int = 2048, item_features=None,
                 attribute_init: str = "legacy", use_global: bool = False, global_layers: int = 2, graph=None):
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
        # 编码器持有各层参数，沿用 layers.* 键名以兼容已有 checkpoint。
        self.layers = LocalGraphEncoder(dim, layers, dropout)
        self.view = nn.Linear(dim, dim)
        nn.init.normal_(self.embedding.weight, std=0.05)
        # 手动正态初始化覆盖了整张表，因此把 padding 行恢复为零；初始化不需要求导。
        with torch.no_grad():
            self.embedding.weight[0].zero_()
        # 在共有参数初始化后创建属性层，方便相同种子下与 ID-only 对照。
        # 属性索引为 buffer，类别/价格 embedding 与投影权重则共同接受推荐梯度。
        self.attribute_encoder = ItemAttributeEncoder(dim, num_items, item_features, attribute_init) if use_item_features else None
        self.global_encoder = None
        self._global_cache = None
        self.graph_fingerprint = None
        if use_global:
            from ese_agp.models.global_graph import DirectedGraphEncoder
            from ese_agp.graphs.behavior import graph_tensors
            if graph is None or graph['num_items'] != num_items:
                raise ValueError("Matching behavior graph required")
            edges, weights = graph_tensors(graph)
            # 图由单独产物+摘要绑定；不复制到每轮 checkpoint，加载时重新校验。
            self.register_buffer("global_edges", edges, persistent=False)
            self.register_buffer("global_weights", weights, persistent=False)
            self.graph_fingerprint = graph['fingerprint']
            self.global_encoder = DirectedGraphEncoder(dim, global_layers, dropout)

    def train(self, mode=True):
        """任何模式切换都清除全局缓存，避免优化之后读取过期表示。"""
        self._global_cache = None
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        """迁移设备/精度时丢弃旧缓存；图 buffer 随模块自动迁移。"""
        self._global_cache = None
        return super()._apply(fn, recurse)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """加载 checkpoint 必须重建与新权重对应的全局表示。"""
        self._global_cache = None
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def encode_global(self):
        """训练不缓存；仅 eval+no_grad 允许复用冻结全图结果。"""
        cache_allowed = not self.training and not torch.is_grad_enabled()
        if cache_allowed and self._global_cache is not None:
            return self._global_cache
        ids = torch.arange(self.num_items + 1, device=self.embedding.weight.device)
        h = self.embedding(ids)
        if self.attribute_encoder is not None: h = self.attribute_encoder(ids, h)
        h = self.global_encoder(h, self.global_edges, self.global_weights)
        if cache_allowed: self._global_cache = h
        return h

    def forward(self, batch: SessionBatch) -> Tensor:
        """输入 SessionBatch，输出 logits[B,N]；第 j 列对应商品 ID j+1。"""
        # 1. 商品查表得到初始节点向量 [M+1,d]。
        h = self.embedding(batch.node_item_ids)
        if self.attribute_encoder is not None:
            h = self.attribute_encoder(batch.node_item_ids, h)
        # 2. 沿会话局部图逐层聚合，节点数量和向量维度保持不变。
        h = self.layers(h, batch.edge_index)
        if self.global_encoder is not None:
            # 局部编号映射回共享商品 ID 后取全局表示，再逐节点相加。
            h = h + self.encode_global()[batch.node_item_ids]
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
