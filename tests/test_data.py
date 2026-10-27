from pathlib import Path

import pytest

from ese_agp.data import DAY_MS, Interaction, load_interactions_jsonl, prepare, prefix_to_local_graph


def _session(user, day, items):
    return [Interaction(user, item, day * DAY_MS + offset) for offset, item in enumerate(items)]


def test_fixed_windows_are_anchored_not_inactivity_sessions():
    interactions = [
        Interaction("u", "A", 0),
        Interaction("u", "B", DAY_MS - 1),
        Interaction("u", "A", DAY_MS + 1),
        Interaction("u", "B", 2 * DAY_MS - 1),
    ]
    result = prepare(interactions, min_user_events=1, min_item_events=1)
    assert len(result.train) == 2
    assert [(s.session_start, s.session_end) for s in result.train] == [
        (0, DAY_MS - 1), (DAY_MS + 1, 2 * DAY_MS - 1)
    ]


def test_raw_boundaries_are_kept_before_truncation_and_crossing_is_removed():
    interactions = [
        Interaction("a", "A", 0), Interaction("a", "B", 10),
        Interaction("b", "A", 0), Interaction("b", "B", 20),
        Interaction("crossing", "A", 15), Interaction("crossing", "A", 25),
        Interaction("crossing", "B", 30),
        Interaction("d", "A", 35), Interaction("d", "B", 40),
    ]
    result = prepare(interactions, min_user_events=1, min_item_events=1,
                     max_session_length=2, train_ratio=.5, valid_ratio=.25)
    assert result.manifest["train_end"] == 20
    assert result.manifest["valid_end"] == 30
    assert result.manifest["crossing_sessions_removed"] == 1
    assert {s.user_id for s in result.train} == {"a", "b"}
    assert [s.user_id for s in result.test] == ["d"]
    assert result.train[1].session_start == 0


def test_train_filters_never_move_target_and_oov_positions_do_not_bridge():
    interactions = _session("u0", 0, ["A", "rare-prefix", "B", "A"])
    interactions += _session("u1", 1, ["A", "B", "rare-target"])
    for day in range(2, 8):
        interactions += _session(f"u{day}", day, ["A", "B", "A"])
    interactions += _session("valid", 8, ["A", "new", "B", "valid-target"])
    interactions += _session("test", 9, ["new", "test-target"])
    result = prepare(interactions, min_user_events=1, min_item_events=2)
    assert result.catalog == {"A": 1, "B": 2}
    first = result.train[0]
    assert first.input_item_ids == (1, 0, 2)
    assert first.target_item_id == 1
    assert "u1" not in {s.user_id for s in result.train}
    assert result.manifest["training_unknown_target_removed"] == 1
    assert result.valid[0].input_item_ids == (1, 0, 2)
    assert result.valid[0].target_item_id == 0
    assert result.test[0].input_item_ids == (0,)
    assert result.test[0].target_item_id == 0
    assert prefix_to_local_graph([first.input_item_ids]).edge_index == ([], [])


def test_heldout_item_changes_cannot_change_training_catalog_or_popularity():
    original = []
    for day in range(10):
        original += _session(f"u{day}", day, ["A", "B", "A"])
    changed = [
        Interaction(row.user_id, "never-trained" if row.user_id in {"u8", "u9"} else row.item_id,
                    row.timestamp)
        for row in original
    ]
    first = prepare(original)
    second = prepare(changed)
    assert first.catalog == second.catalog
    assert first.train == second.train
    assert first.popularity == second.popularity
    assert first.manifest["train_item_counts"] == second.manifest["train_item_counts"]
    assert first.manifest["train_end"] == second.manifest["train_end"]


def test_user_filter_is_fitted_only_on_train_and_max_length_includes_target():
    rows = []
    for day in range(10):
        items = ["A", "B"] if day == 0 else ["A", "B", "C", "A"]
        rows += _session(f"u{day}", day, items)
    result = prepare(rows, min_user_events=3, min_item_events=1, max_session_length=3)
    assert "u0" not in {s.user_id for s in result.train}
    assert result.manifest["training_user_sessions_removed"] == 1
    assert len(result.train[0].input_item_ids) == 2
    assert result.train[0].target_item_id == result.catalog["A"]
    assert result.train[0].session_start == DAY_MS
    assert result.train[0].session_end == DAY_MS + 3
    # Unseen heldout user IDs do not cause their evaluation sessions to vanish.
    assert [s.user_id for s in result.valid] == ["u8"]
    assert [s.user_id for s in result.test] == ["u9"]


def test_local_graph_is_disjoint_and_preserves_repeats_masks_and_self_edges():
    graph = prefix_to_local_graph([[1, 2, 1, 1], [1], [1, 0, 2], []])
    assert graph.node_item_ids == [0, 1, 2, 1, 1, 2]
    assert graph.edge_index == ([1, 1, 2], [1, 2, 1])
    assert graph.position_to_node == [[1, 2, 1, 1], [3, 0, 0, 0], [4, 0, 5, 0], [0, 0, 0, 0]]
    assert graph.position_mask == [
        [True, True, True, True], [True, False, False, False],
        [True, False, True, False], [False, False, False, False]
    ]
    assert graph.has_known_history == [True, True, True, False]
    assert prefix_to_local_graph([[], [0]]).node_item_ids == [0]


def test_tiny_fixture_is_trainable_with_known_heldout_targets():
    fixture = Path(__file__).parents[1] / "examples" / "tiny_interactions.jsonl"
    result = prepare(load_interactions_jsonl(fixture))
    assert result.catalog == {"A": 1, "B": 2, "C": 3}
    assert [len(result.train), len(result.valid), len(result.test)] == [16, 2, 2]
    assert all(example.target_item_id > 0 for split in (result.train, result.valid, result.test)
               for example in split)
    assert set(result.popularity) == {1, 2, 3}


def test_invalid_inputs_have_actionable_errors(tmp_path):
    with pytest.raises(ValueError, match="timestamp"):
        Interaction("u", "A", True)
    with pytest.raises(ValueError, match="ratios"):
        prepare([Interaction("u", "A", 0)], train_ratio=.9, valid_ratio=.2)
    with pytest.raises(ValueError, match="nonnegative"):
        prefix_to_local_graph([[1, -1]])
    path = tmp_path / "bad.jsonl"
    path.write_text('{"user_id":"u","timestamp":0}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_interactions_jsonl(path)
