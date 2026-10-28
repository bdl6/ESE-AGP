"""局部图模型的计算和梯度检查；用小图、简单向量与保存加载对照定位问题。"""
import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ese_agp.data import PreparedData, SessionExample
from ese_agp.model import IncomingAttentionLayer, LocalSessionRecommender, collate_examples
from ese_agp.training import data_fingerprint, evaluate_model, load_model, load_prepared, prepared_payload, write_json


def example(prefix, target=1):
    """构造一个已编号监督样本；用户和时间仅填占位值，不参与神经模型输入。"""
    return SessionExample(input_item_ids=tuple(prefix), target_item_id=target,
                          user_id="u", session_start=0, session_end=1)


def test_incoming_direction_and_dummy_zero():
    """固定线性层为可手算参数，检验 1→2 只有节点 2 收到节点 1 消息，虚拟节点为零。"""
    layer = IncomingAttentionLayer(2, 0.0)
    with torch.no_grad():
        layer.message.weight.copy_(torch.eye(2))
        layer.self_path.weight.zero_()
        layer.self_path.bias.zero_()
        layer.source_score.weight.zero_()
        layer.target_score.weight.zero_()
    h = torch.tensor([[0., 0.], [1., 2.], [3., 4.]])
    output = layer(h, torch.tensor([[1], [2]]))
    torch.testing.assert_close(output, torch.tensor([[0., 0.], [0., 0.], [1., 2.]]))


def test_readout_counts_repeated_positions_and_masks_oov():
    """绕过图层、用简单非负向量手算池化，检验重复点击计数以及 OOV/补齐不计数。"""
    model = LocalSessionRecommender(2, dim=2, layers=1)
    # 移除图层并将视角设为恒等映射，只检验位置读出；非负 embedding 使 ReLU 不改数值。
    model.layers = nn.ModuleList()
    model.view = nn.Identity()
    with torch.no_grad():
        model.embedding.weight.copy_(torch.tensor([[0., 0.], [1., 0.], [0., 1.]]))
    batch = collate_examples([example([1, 2, 1]), example([1, 0, 2]), example([0])], torch.device("cpu"))
    torch.testing.assert_close(model(batch), torch.tensor([[2/3, 1/3], [.5, .5], [0., 0.]]))


def test_multiple_incoming_neighbors_train_attention_scores():
    """令节点 3 同时接收节点 1/2 的不同消息，验证注意力打分参数得到有限非零梯度。"""
    layer = IncomingAttentionLayer(2, 0.0)
    with torch.no_grad():
        layer.message.weight.copy_(torch.eye(2))
        layer.self_path.weight.zero_()
        layer.self_path.bias.zero_()
        layer.source_score.weight.copy_(torch.tensor([[.3, -.2]]))
        layer.target_score.weight.zero_()
    h = torch.tensor([[0., 0.], [1., 0.], [0., 2.], [0., 0.]])
    # 需要同一接收节点有多个入邻居；单入邻居 softmax 恒为 1，无法检验注意力学习。
    output = layer(h, torch.tensor([[1, 2], [3, 3]]))
    output[3, 0].backward()
    assert 0 < output[3, 0] < 1
    assert torch.isfinite(layer.source_score.weight.grad).all()
    assert layer.source_score.weight.grad.abs().sum() > 0


def test_unknown_history_uses_popularity_and_excludes_model_loss():
    """全未知/空历史必须走热门排名；不能把对应神经 logits 的 CE 混入模型损失统计。"""
    model = LocalSessionRecommender(2, dim=4, layers=1)
    metrics = evaluate_model(model, [example([0], 1), example([], 2)], [2, 1], torch.device("cpu"), k=1)
    assert metrics["hr@1"] == .5
    assert metrics["model_known_history_ce"] is None
    assert metrics["num_model_loss_examples"] == 0


def test_batch_graphs_are_independent_and_padding_has_no_gradient():
    """合批与逐条前向分数应一致；真实商品 embedding 有梯度，padding 行无梯度。"""
    torch.manual_seed(42)
    model = LocalSessionRecommender(3, dim=8).eval()
    examples = [example([1, 2, 1], 3), example([3, 0, 2], 1)]
    together = model(collate_examples(examples, torch.device("cpu")))
    separately = torch.cat([model(collate_examples([ex], torch.device("cpu"))) for ex in examples])
    torch.testing.assert_close(together, separately)
    F.cross_entropy(together, torch.tensor([2, 0])).backward()
    assert torch.isfinite(model.embedding.weight.grad).all()
    assert model.embedding.weight.grad[1:].abs().sum() > 0
    assert model.embedding.weight.grad[0].count_nonzero() == 0


def test_tiny_data_can_overfit_full_candidate_cross_entropy():
    """三条简单映射应能被拟合，证明损失到参数的训练链路可用；不代表真实泛化能力。"""
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = LocalSessionRecommender(3, dim=16, layers=1)
    batch = collate_examples([example([1], 2), example([2], 3), example([3], 1)], torch.device("cpu"))
    # 商品目标是 2/3/1，对应交叉熵类别下标 1/2/0。
    target = torch.tensor([1, 2, 0])
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    initial_loss = F.cross_entropy(model(batch), target).item()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch), target)
        loss.backward()
        optimizer.step()
        if loss.item() < .01:
            break
    model.eval()
    assert F.cross_entropy(model(batch), target).item() < min(.03, initial_loss / 10)
    assert model(batch).argmax(1).tolist() == target.tolist()


def test_checkpoint_scores_match_and_rejects_different_data(tmp_path):
    """保存加载后分数完全一致；交换商品编号时应拒绝旧 checkpoint，防止推荐错商品。"""
    data = PreparedData(train=[example([1], 2)], valid=[], test=[], catalog={"A": 1, "B": 2},
                        popularity=[1, 2], manifest={"synthetic_debug": True})
    config = {"dim": 8, "layers": 1, "dropout": 0.0}
    model = LocalSessionRecommender(2, **config).eval()
    path = tmp_path / "best.pt"
    torch.save({"schema_version": 1, "model_state": model.state_dict(), "num_items": 2,
                "model_config": config, "catalog": data.catalog, "data_fingerprint": data_fingerprint(data)}, path)
    restored = load_model(path, data, torch.device("cpu"))
    batch = collate_examples(data.train, torch.device("cpu"))
    torch.testing.assert_close(model(batch), restored(batch), rtol=0, atol=0)
    changed = copy.deepcopy(data)
    # 模拟编号被换掉的另一套目录，即使候选数相同也不能直接复用原权重。
    changed.catalog["A"], changed.catalog["B"] = 2, 1
    with pytest.raises(ValueError, match="fingerprints"):
        load_model(path, changed, torch.device("cpu"))


def test_prepared_json_round_trip_keeps_data_fingerprint(tmp_path):
    """预处理产物经过 JSON 写入/读取，数据结构及其身份指纹均应保持一致。"""
    data = PreparedData(train=[example([1], 2)], valid=[], test=[], catalog={"A": 1, "B": 2},
                        popularity=[2, 1], manifest={"train_item_counts": {"1": 1, "2": 1}})
    path = tmp_path / "data.json"
    write_json(path, prepared_payload(data))
    restored = load_prepared(path)
    assert restored == data
    assert data_fingerprint(restored) == data_fingerprint(data)
