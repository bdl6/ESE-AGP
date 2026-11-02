"""数据处理的手工样例：检验固定窗口、统一词表、全局过滤和局部图结构。"""
from pathlib import Path

import pytest

from ese_agp.data.sessions import DAY_MS, Interaction, load_interactions_jsonl, prepare, prefix_to_local_graph


def _session(user, day, items):
    """生成某用户第 day 天的有序事件；每次事件间隔 1 毫秒，方便手算。"""
    return [Interaction(user, item, day * DAY_MS + offset) for offset, item in enumerate(items)]


def test_fixed_windows_are_anchored_not_inactivity_sessions():
    """验证按首次事件锚定的 24h 窗口，而非相邻交互的空闲超时切分。"""
    interactions = [
        Interaction("u", "A", 0),
        Interaction("u", "B", DAY_MS - 1),
        Interaction("u", "A", DAY_MS + 1),
        Interaction("u", "B", 2 * DAY_MS - 1),
    ]
    result = prepare(interactions, min_user_events=1, min_item_events=1)
    assert len(result.train)+len(result.test) == 2
    assert [(s.session_start, s.session_end) for s in sorted([*result.train,*result.test],key=lambda x:x.session_start)] == [
        (0, DAY_MS - 1), (DAY_MS + 1, 2 * DAY_MS - 1)
    ]










def test_local_graph_is_disjoint_and_preserves_repeats_masks_and_self_edges():
    """手算批内局部图，检查图间隔离、重复节点映射、真实自环、OOV 和补齐 mask。"""
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
    """检验人工样例产生 16/2/2 切分和 A/B/C 稳定编号，可供后续训练排错。"""
    fixture = Path(__file__).parents[2] / "examples" / "tiny_interactions.jsonl"
    result = prepare(load_interactions_jsonl(fixture))
    assert result.catalog == {"A": 1, "B": 2, "C": 3}
    assert [len(result.train), len(result.valid), len(result.test)] == [16, 2, 2]
    assert all(example.target_item_id > 0 for split in (result.train, result.valid, result.test)
               for example in split)
    assert set(result.popularity) == {1, 2, 3}


def test_invalid_inputs_have_actionable_errors(tmp_path):
    """非法时间戳、切分比例、负编号或缺字段坏行，应报告可定位的异常。"""
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


def test_debug_vocabulary_contains_test_only_items_without_deleting_history():
    """测试独有商品仍有合法ID，不删除中间点击或重新连接历史。"""
    rows=[]
    for day in range(10):rows+=_session(f'u{day}',day,['A','B','C'])
    before=prepare(rows,min_user_events=1,min_item_events=1)
    heldout=before.test[0].user_id
    changed=[Interaction(r.user_id,'ONLY_TEST' if r.user_id==heldout and r.item_id=='B' else r.item_id,r.timestamp) for r in rows]
    result=prepare(changed,min_user_events=1,min_item_events=1)
    ex=next(x for x in result.test if x.user_id==heldout)
    assert 'ONLY_TEST' in result.catalog
    assert ex.input_item_ids==(result.catalog['A'],result.catalog['ONLY_TEST'])
    assert all(result.catalog['ONLY_TEST'] not in (*x.input_item_ids,x.target_item_id) for x in result.train)
    assert all(i>0 for split in [result.train,result.valid,result.test] for x in split for i in (*x.input_item_ids,x.target_item_id))


def test_debug_global_filter_and_history_length_match_current_rule():
    """频次在全局迭代，输入长度不包含独立预测目标。"""
    rows=[]
    for day in range(10):rows+=_session(f'u{day}',day,['A','B','C','A','B'])
    result=prepare(rows,min_user_events=3,min_item_events=5,max_session_length=3)
    for x in [*result.train,*result.valid,*result.test]:
        assert x.input_item_ids==(result.catalog['B'],result.catalog['C'],result.catalog['A'])
        assert x.target_item_id==result.catalog['B']
    assert result.manifest['filter_rounds'][-1]['converged']
