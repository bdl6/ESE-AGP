"""所有模型共用的排名指标，以及只拟合训练数据的热门/相邻转移基线。

候选商品为训练目录 1..N，0 是 padding/OOV，不能出现在推荐列表中。
允许推荐历史中已见的商品；同一列表内部不能重复 ID，不能事后改变候选口径。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Iterable, Sequence


def _integer(value: Any, name: str, *, minimum: int) -> int:
    """校验非 bool 的整数及下界，并转成 Python int；同时接受 numbers.Integral 类型。"""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _catalog_item(value: Any, num_items: int) -> int:
    """检查编号在 0..N 范围内；0 可作为输入 OOV，但不会成为推荐候选。"""
    item_id = _integer(value, "item ID", minimum=0)
    if item_id > num_items:
        raise ValueError(f"item ID {item_id} is outside catalog 0..{num_items}")
    return item_id


def evaluate_rankings(
    rankings: Iterable[Sequence[int]],
    targets: Iterable[int],
    k: int = 20,
) -> dict[str, float | int]:
    """按真实目标在 Top-K 中的位置计算下一商品预测指标。

    Args:
        rankings: 每个样本按预测分数排序的商品 ID 序列，序列内部不能重复或包含 0。
        targets: 与 rankings 一一对应的真实商品 ID；0 代表不在训练目录内的未知目标。
        k: 排名前 k 个位置参与计分，k 必须为正整数。
    Returns:
        hr@k = 命中样本数 / 全部样本数；
        mrr@k = 命中样本的排名倒数之和 / 全部样本数；
        known_hr/known_mrr 仅对目标已知的子集计算，另报 OOV 比例和样本数。

    例如目标排名为 1、2、未知，k=2 时整体 HR=2/3，MRR=(1+1/2+0)/3。
    未知目标必须在整体分母计 miss，不能直接删掉使指标虚高；返回值是比例而非百分数。
    空集合约定指标为 0，同时返回数量，避免把无样本误读成正常有效实验。
    """

    k = _integer(k, "k", minimum=1)
    ranking_list = list(rankings)
    target_list = list(targets)
    if len(ranking_list) != len(target_list):
        raise ValueError("rankings and targets must contain the same number of examples")

    hits = 0
    reciprocal_rank_sum = 0.0
    known_count = 0
    for ranking, target_value in zip(ranking_list, target_list):
        target = _integer(target_value, "target item ID", minimum=0)
        items = [_integer(item, "ranked item ID", minimum=1) for item in ranking]
        if len(set(items)) != len(items):
            raise ValueError("each ranking must contain unique item IDs")
        # 未知目标不加 hits/排名倒数，但仍留在末尾 total 的全样本分母中。
        if target == 0:
            continue
        known_count += 1
        try:
            # 列表下标从 0 开始，真实排名从 1 开始；不在 Top-K 就没有本次命中。
            rank = items[:k].index(target) + 1
        except ValueError:
            continue
        hits += 1
        # 排第 1 名贡献 1，排第 2 名贡献 1/2；越靠前贡献越高。
        reciprocal_rank_sum += 1.0 / rank

    total = len(target_list)
    oov_count = total - known_count
    return {
        f"hr@{k}": hits / total if total else 0.0,
        f"mrr@{k}": reciprocal_rank_sum / total if total else 0.0,
        f"known_hr@{k}": hits / known_count if known_count else 0.0,
        f"known_mrr@{k}": reciprocal_rank_sum / known_count if known_count else 0.0,
        "oov_target_rate": oov_count / total if total else 0.0,
        "num_examples": total,
        "num_known_targets": known_count,
        "num_oov_targets": oov_count,
    }


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
