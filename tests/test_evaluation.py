"""用可手算排名验证 HR/MRR、OOV 分母、并列规则和非神经基线的训练范围。"""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from ese_agp.evaluation import (
    PopularityBaseline,
    TransitionBaseline,
    evaluate_rankings,
)


def example(prefix, target):
    """构造只含前缀/目标属性的轻量样本，隔离测试评估逻辑，不依赖数据准备过程。"""
    return SimpleNamespace(input_item_ids=prefix, target_item_id=target)


def test_hand_calculated_metrics_keep_oov_targets_in_full_denominator():
    """四个样本中含一个未知目标，手算 HR@2=0.5、MRR@2=0.375，并核对已知子集。"""
    metrics = evaluate_rankings(
        [[2, 1, 3], [2, 3, 1], [1, 2, 3], [3, 1, 2]],
        [2, 1, 0, 1],
        k=2,
    )
    assert metrics["hr@2"] == 0.5
    assert metrics["mrr@2"] == 0.375
    assert metrics["known_hr@2"] == pytest.approx(2 / 3)
    assert metrics["known_mrr@2"] == 0.5
    assert metrics["oov_target_rate"] == 0.25
    assert metrics["num_examples"] == 4
    assert metrics["num_known_targets"] == 3
    assert metrics["num_oov_targets"] == 1


def test_boundary_short_rankings_and_known_target_outside_top_k():
    """检验恰好排第 K 名算命中、排 K+1 不算，以及空/短推荐列表的处理。"""
    metrics = evaluate_rankings([[3, 1, 2], [], [1]], [1, 2, 1], k=2)
    assert metrics["hr@2"] == pytest.approx(2 / 3)
    assert metrics["mrr@2"] == 0.5
    assert evaluate_rankings([[3, 1, 2]], [2], k=2)["hr@2"] == 0.0


def test_empty_and_all_oov_sets_are_explicitly_finite():
    """无样本或目标全未知时返回有限指标和准确数量，不能出现除零或 NaN。"""
    empty = evaluate_rankings([], [], k=20)
    assert empty["num_examples"] == 0
    assert all(value == 0 for value in empty.values())
    unknown = evaluate_rankings([[1], []], [0, 0], k=20)
    assert unknown["hr@20"] == unknown["known_hr@20"] == 0.0
    assert unknown["mrr@20"] == unknown["known_mrr@20"] == 0.0
    assert unknown["oov_target_rate"] == 1.0


@pytest.mark.parametrize(
    "rankings,targets,k",
    [([[1]], [], 1), ([[0]], [1], 1), ([[1, 1]], [1], 2), ([[1]], [-1], 1), ([[1]], [1], 0)],
)
def test_invalid_metric_inputs_fail_instead_of_silently_changing_protocol(rankings, targets, k):
    """参数化覆盖样本数量不匹配、非法候选、重复候选、负目标和非正 K。"""
    with pytest.raises(ValueError):
        evaluate_rankings(rankings, targets, k)


def test_popularity_counts_complete_training_session_and_stable_ties():
    """热门计数含训练目标，重复行为逐次计数；并列按 ID，且不排除已见商品。"""
    train = [example([2, 2], 1), example([3], 1)]
    model = PopularityBaseline.fit(train, num_items=4)
    assert model.counts == (0, 2, 2, 1, 0)
    assert model.recommend([2], top_k=20) == [1, 2, 3, 4]
    # 已浏览商品仍可再次成为下一商品，不在评估时擅自删掉这些候选。
    assert model.recommend([1, 2], top_k=2) == [1, 2]


def test_transition_is_directed_counts_training_target_and_breaks_at_oov():
    """验证转移方向、训练标签可计数，以及 [A,0,B] 不会贡献 A→B。"""
    train = [example([1, 0, 2], 3), example([1], 4), example([1], 4)]
    model = TransitionBaseline.fit(train, num_items=4)
    assert model.transitions == {2: {3: 1}, 1: {4: 2}}
    assert model.recommend([1], top_k=4) == [4, 1, 2, 3]
    assert model.recommend([2], top_k=1) == [3]
    assert 1 not in model.transitions.get(4, {})
    assert 2 not in model.transitions[1]


def test_transition_count_ties_use_item_id_and_never_filter_seen_items():
    """同分后继按 ID 排列，重复推荐已见商品与真实自转移保持有效。"""
    model = TransitionBaseline.fit([example([3], 2), example([3], 1)], num_items=3)
    assert model.recommend([3, 1, 3], top_k=3) == [1, 2, 3]
    assert TransitionBaseline.fit([example([1], 1)], 2).recommend([1], 1) == [1]


def test_transition_unknown_last_position_uses_popular_without_oov_compaction():
    """末位置未知时走热门，不删除 OOV 后向前找末项；已知孤立末项按全零分数排。"""
    model = TransitionBaseline.fit([example([1, 2, 2], 2)], num_items=3)
    assert model.recommend([], 3) == [2, 1, 3]
    assert model.recommend([1, 0], 3) == [2, 1, 3]
    assert model.recommend([0, 1], 3) == [2, 1, 3]
    # 已知末项若没有出边，所有转移分数为零，按 ID 排名，不自动混入热门分数。
    assert model.recommend([3], 3) == [1, 2, 3]


def test_heldout_targets_do_not_update_fitted_baselines_or_training_inputs():
    """改变评估标签只改变计分，不能反过来改基线统计、预测排名或训练输入。"""
    train = [example([1], 2), example([1], 2)]
    original_train = deepcopy(train)
    popular = PopularityBaseline.fit(iter(train), 3)
    transition = TransitionBaseline.fit(iter(train), 3)
    predictions = [transition.recommend([1], 3)]
    counts_before = popular.counts
    transitions_before = deepcopy(transition.transitions)
    first = evaluate_rankings(predictions, [2], k=1)
    second = evaluate_rankings(predictions, [3], k=1)
    assert first["hr@1"] == 1.0
    assert second["hr@1"] == 0.0
    assert transition.recommend([1], 3) == predictions[0] == [2, 1, 3]
    assert popular.counts == counts_before == (0, 2, 2, 0)
    assert transition.transitions == transitions_before == {1: {2: 2}}
    assert [vars(item) for item in train] == [vars(item) for item in original_train]


def test_fit_and_recommend_reject_out_of_catalog_ids():
    """超出训练目录、负编号和非法 K 应报错，避免训练与预测悄悄使用不同候选。"""
    with pytest.raises(ValueError):
        PopularityBaseline.fit([example([1], 4)], 3)
    with pytest.raises(ValueError):
        TransitionBaseline.fit([example([-1], 1)], 3)
    model = TransitionBaseline.fit([], 3)
    with pytest.raises(ValueError):
        model.recommend([4], 1)
    with pytest.raises(ValueError):
        model.recommend([1], 0)
