"""可训练的商品属性embedding与投影层。"""
import torch
from torch import nn

class ItemAttributeEncoder(nn.Module):
    """查类别/价格表，与已有 ID 向量拼接再投影；三张表都接受推荐损失梯度。"""
    def __init__(self, dim, num_items, features, initialization="legacy"):
        """features 必须来自与模型同一训练目录，编号 0 的属性必须为零。"""
        super().__init__()
        categories = features['category_ids']
        prices = features['price_bin_ids']
        bins = features['num_price_bins']
        if len(categories) != num_items + 1 or len(prices) != num_items + 1:
            raise ValueError('Feature arrays do not match the catalog')
        if categories[0] != 0 or prices[0] != 0:
            raise ValueError('Padding feature indices must be zero')
        if any(x < 1 for x in categories[1:]) or any(not 1 <= x <= bins for x in prices[1:]):
            raise ValueError('Known products must have valid feature indices')
        # buffer 随模型迁移设备/保存，但商品→属性编号本身不求梯度。
        self.register_buffer('category_ids', torch.tensor(categories, dtype=torch.long))
        self.register_buffer('price_bin_ids', torch.tensor(prices, dtype=torch.long))
        self.category_embedding = nn.Embedding(max(categories) + 1, dim, padding_idx=0)
        self.price_embedding = nn.Embedding(bins + 1, dim, padding_idx=0)
        self.projection = nn.Linear(3 * dim, dim)
        # legacy 保留 Day02；matched 仅修改属性表初始化尺度，便于控制变量。
        if initialization not in {"legacy", "matched"}:
            raise ValueError("attribute_init must be legacy or matched")
        if initialization == "matched":
            for embedding in (self.category_embedding, self.price_embedding):
                nn.init.normal_(embedding.weight, std=0.05)
                with torch.no_grad(): embedding.weight[0].zero_()

    def forward(self, item_ids, id_vectors):
        """item_ids[*] 与 id_vectors[*,d] → 融合表示[*,d]；padding 输出严格为零。"""
        category = self.category_embedding(self.category_ids[item_ids])
        price = self.price_embedding(self.price_bin_ids[item_ids])
        result = self.projection(torch.cat([id_vectors, category, price], dim=-1))
        return result * item_ids.ne(0).unsqueeze(-1)
