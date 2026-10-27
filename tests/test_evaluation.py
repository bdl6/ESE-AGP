"""Hand-checkable tests for ranking protocols and baseline training boundaries."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from ese_agp.evaluation import (
    PopularityBaseline,
    TransitionBaseline,
    evaluate_rankings,
)


def example(prefix, target):
    return SimpleNamespace(input_item_ids=prefix, target_item_id=target)


def test_hand_calculated_metrics_keep_oov_targets_in_full_denominator():
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
    metrics = evaluate_rankings([[3, 1, 2], [], [1]], [1, 2, 1], k=2)
    assert metrics["hr@2"] == pytest.approx(2 / 3)
    assert metrics["mrr@2"] == 0.5
    assert evaluate_rankings([[3, 1, 2]], [2], k=2)["hr@2"] == 0.0


def test_empty_and_all_oov_sets_are_explicitly_finite():
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
    with pytest.raises(ValueError):
        evaluate_rankings(rankings, targets, k)


def test_popularity_counts_complete_training_session_and_stable_ties():
    train = [example([2, 2], 1), example([3], 1)]
    model = PopularityBaseline.fit(train, num_items=4)
    assert model.counts == (0, 2, 2, 1, 0)
    assert model.recommend([2], top_k=20) == [1, 2, 3, 4]
    # Already seen items remain eligible, including repeated next-item behavior.
    assert model.recommend([1, 2], top_k=2) == [1, 2]


def test_transition_is_directed_counts_training_target_and_breaks_at_oov():
    train = [example([1, 0, 2], 3), example([1], 4), example([1], 4)]
    model = TransitionBaseline.fit(train, num_items=4)
    assert model.transitions == {2: {3: 1}, 1: {4: 2}}
    assert model.recommend([1], top_k=4) == [4, 1, 2, 3]
    assert model.recommend([2], top_k=1) == [3]
    assert 1 not in model.transitions.get(4, {})
    assert 2 not in model.transitions[1]


def test_transition_count_ties_use_item_id_and_never_filter_seen_items():
    model = TransitionBaseline.fit([example([3], 2), example([3], 1)], num_items=3)
    assert model.recommend([3, 1, 3], top_k=3) == [1, 2, 3]
    assert TransitionBaseline.fit([example([1], 1)], 2).recommend([1], 1) == [1]


def test_transition_unknown_last_position_uses_popular_without_oov_compaction():
    model = TransitionBaseline.fit([example([1, 2, 2], 2)], num_items=3)
    assert model.recommend([], 3) == [2, 1, 3]
    assert model.recommend([1, 0], 3) == [2, 1, 3]
    assert model.recommend([0, 1], 3) == [2, 1, 3]
    # A known item with no outgoing observations has all-zero transition scores.
    assert model.recommend([3], 3) == [1, 2, 3]


def test_heldout_targets_do_not_update_fitted_baselines_or_training_inputs():
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
    with pytest.raises(ValueError):
        PopularityBaseline.fit([example([1], 4)], 3)
    with pytest.raises(ValueError):
        TransitionBaseline.fit([example([-1], 1)], 3)
    model = TransitionBaseline.fit([], 3)
    with pytest.raises(ValueError):
        model.recommend([4], 1)
    with pytest.raises(ValueError):
        model.recommend([1], 0)
