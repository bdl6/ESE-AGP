"""只拟合训练交互的热门与相邻转移推荐基线。"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence
from ese_agp.utils.validation import _integer, _catalog_item

@dataclass(frozen=True)
class PopularityBaseline:
    """训练期热门基线：按商品出现次数降序，次数并列按商品 ID 升序。

    本项目每个会话只产生一个样本，所以统计完整训练序列=输入前缀+训练标签。
    若以后扩增成一个会话多个前缀样本，直接复用此统计会重复计数，需要明确改协议。
    """

    num_items: int
    # 按商品 ID 下标索引的计数，长度 N+1，第 0 项不参加推荐。
    counts: tuple[int, ...]
    # 预先排好的商品 ID 列表，热门推荐时直接切片，无需每次重排。
    _ranking: tuple[int, ...]

    @classmethod
    def fit(cls, train: Iterable[Any], num_items: int) -> PopularityBaseline:
        """从 train 样本统计已知商品出现次数并缓存排名，返回基线实例。

        num_items 为训练目录大小，不含 0；此处不读取任何验证/测试标签。
        """
        num_items = _integer(num_items, "num_items", minimum=1)
        counts = [0] * (num_items + 1)
        for example in train:
            # 训练标签属于过去已观察到的训练行为，因此可以参与训练期的热门统计。
            sequence = [*example.input_item_ids, example.target_item_id]
            for value in sequence:
                item_id = _catalog_item(value, num_items)
                if item_id:
                    counts[item_id] += 1
        # 负计数实现降序，ID 实现确定性的并列规则；范围 1..N 排除未知候选。
        ranking = tuple(sorted(range(1, num_items + 1), key=lambda item: (-counts[item], item)))
        return cls(num_items=num_items, counts=tuple(counts), _ranking=ranking)

    def recommend(self, prefix: Sequence[int], top_k: int) -> list[int]:
        """校验前缀后返回固定热门榜前 top_k 件；不屏蔽已见商品，也不改变热门分数。"""
        top_k = _integer(top_k, "top_k", minimum=1)
        # 热门策略不使用前缀内容，但仍校验编号范围，与其他模型保持输入约定一致。
        for value in prefix:
            _catalog_item(value, self.num_items)
        return list(self._ranking[:top_k])


@dataclass(frozen=True)
class TransitionBaseline:
    """相邻转移基线：用前缀最后一个位置的商品，查询训练期的有向下一跳计数。

    [A,0,B] 不记录 A→B，也不记录涉及 0 的边；空历史或末位置为 0 时走热门。
    末项已知时，对全部候选按转移计数降序排序；零计数项也按商品 ID 升序补齐。
    已知末项没有后继证据时是全零分数排名，不会自动切换为热门。
    """

    num_items: int
    popularity: PopularityBaseline
    # 来源商品 → {接收商品:训练转移次数}，方向不能颠倒。
    transitions: dict[int, dict[int, int]]

    @classmethod
    def fit(cls, train: Iterable[Any], num_items: int) -> TransitionBaseline:
        """只用完整训练会话统计有向相邻边次数，并保存训练热门基线供降级。

        先把可迭代对象变成列表，确保热门计数和转移计数都能遍历同一份样本。
        """
        num_items = _integer(num_items, "num_items", minimum=1)
        examples = list(train)
        popularity = PopularityBaseline.fit(examples, num_items)
        transitions: dict[int, Counter[int]] = defaultdict(Counter)
        for example in examples:
            sequence = [
                _catalog_item(value, num_items)
                for value in [*example.input_item_ids, example.target_item_id]
            ]
            # 严格按原始相邻位置统计，只有两端都是已知商品才计入；真实自转移保留。
            for source, target in zip(sequence, sequence[1:]):
                if source and target:
                    transitions[source][target] += 1
        return cls(
            num_items=num_items,
            popularity=popularity,
            transitions={source: dict(counts) for source, counts in transitions.items()},
        )

    def recommend(self, prefix: Sequence[int], top_k: int) -> list[int]:
        """返回给定输入前缀的前 top_k 个候选 ID，允许推荐已见商品。

        只读取原前缀的最后位置；不能删除末尾 OOV 再向前找商品，否则改变了预测策略。
        """
        top_k = _integer(top_k, "top_k", minimum=1)
        items = [_catalog_item(value, self.num_items) for value in prefix]
        # 末尾未知时不向前跨位置寻找另一个末项，统一交给训练热门榜。
        if not items or items[-1] == 0:
            return self.popularity.recommend(items, top_k)
        # 对所有训练目录商品排序，包括没有观测后继的零分候选，不过滤已见商品。
        counts = self.transitions.get(items[-1], {})
        # 正转移计数永远排在零分候选前；只排序非零后继，再按 ID 补零分项。
        # 与全目录排序完全同口径，但避免每个样本都对 N 个商品排序。
        ranking = sorted(counts, key=lambda item: (-counts[item], item))[:top_k]
        if len(ranking) < min(top_k, self.num_items):
            for item in range(1, self.num_items + 1):
                if item not in counts:
                    ranking.append(item)
                    if len(ranking) >= top_k:
                        break
        return ranking
