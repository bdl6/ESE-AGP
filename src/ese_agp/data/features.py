"""商品属性预处理：生成类别/相对价格统计和固定索引，不含可训练模型。"""
from collections import defaultdict
import math
import re
import statistics



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
