"""所有模型共用的HR/MRR排名指标。"""
from typing import Iterable, Sequence
from ese_agp.utils.validation import _integer

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
