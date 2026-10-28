"""商品属性编码：训练目录内拟合类别/相对价格，并与 ID 向量端到端训练。"""
from collections import defaultdict
import math
import re
import statistics

import torch
from torch import nn


def parse_price(value):
    """解析美元数值/字符串；拒绝区间、负数、布尔、缺失和非有限价格。"""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip().replace(',', '')
    if not re.fullmatch(r'\$?\s*\d+(?:\.\d+)?', text):
        return None
    price = float(text.replace('$', '').strip())
    return price if math.isfinite(price) and price > 0 else None


def _sigmoid(value):
    """截断指数的输入，防止极端价格引起浮点溢出。"""
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def relative_price_bin(price, stats, num_bins=50):
    """按论文式 3-28..31 将类内价格映射为 1..P；0 留给 padding。

    gamma=std*sqrt(3)/pi；先 sigmoid，再按类内最小/最大变换值归一。
    零方差或数值退化类目放中间档；统计只来自训练商品，不按事件次数加权。
    """
    if num_bins < 1:
        raise ValueError('num_bins must be positive')
    scale = stats['std'] * math.sqrt(3) / math.pi
    if scale < 1e-12:
        return num_bins // 2 + 1
    low = _sigmoid((stats['min'] - stats['mean']) / scale)
    high = _sigmoid((stats['max'] - stats['mean']) / scale)
    if high - low < 1e-12:
        return num_bins // 2 + 1
    ratio = (_sigmoid((price - stats['mean']) / scale) - low) / (high - low)
    ratio = max(0.0, min(1.0, ratio))
    return min(int(ratio * num_bins), num_bins - 1) + 1


def fit_item_features(products, num_bins=50):
    """输入训练目录商品 (item_id, category, price)，输出与 0..N 严格对齐的属性。

    category/price 必须已清洗；唯一商品只贡献一次价格统计，避免热门商品主导均价。
    返回 JSON 可保存的列表/字典，供模型注册为固定索引 buffer，而非可学习参数。
    """
    rows = sorted(products, key=lambda row: row[0])
    if [row[0] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('Product IDs must be unique and contiguous from 1')
    grouped = defaultdict(list)
    for _, category, price in rows:
        if not category or parse_price(price) is None:
            raise ValueError('Training product attributes must be valid')
        grouped[category].append(float(price))
    mapping = {category: i + 1 for i, category in enumerate(sorted(grouped))}
    stats = {category: {'mean': statistics.mean(prices), 'std': statistics.pstdev(prices),
                        'min': min(prices), 'max': max(prices)}
             for category, prices in sorted(grouped.items())}
    return {'schema_version': 1, 'category_mapping': mapping, 'num_price_bins': num_bins,
            'category_ids': [0] + [mapping[category] for _, category, _ in rows],
            'price_bin_ids': [0] + [relative_price_bin(float(price), stats[category], num_bins)
                                    for _, category, price in rows],
            'price_stats': stats,
            'formula': 'paper_3_28_to_31_population_std_bins_shifted_by_1'}


class ItemAttributeEncoder(nn.Module):
    """查类别/价格表，与已有 ID 向量拼接再投影；三张表都接受推荐损失梯度。"""
    def __init__(self, dim, num_items, features):
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

    def forward(self, item_ids, id_vectors):
        """item_ids[*] 与 id_vectors[*,d] → 融合表示[*,d]；padding 输出严格为零。"""
        category = self.category_embedding(self.category_ids[item_ids])
        price = self.price_embedding(self.price_bin_ids[item_ids])
        result = self.projection(torch.cat([id_vectors, category, price], dim=-1))
        return result * item_ids.ne(0).unsqueeze(-1)
