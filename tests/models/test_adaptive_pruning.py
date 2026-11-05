import torch

from ese_agp.models.adaptive_pruning import AdaptiveGraphPruner, normalized_incoming


def test_directed_normalization_keeps_input_direction_and_adds_loops():
    edges = torch.tensor([[1, 2], [2, 2]])
    matrix = normalized_incoming(edges, torch.tensor([0.5, 0.5]), 4).to_dense()
    expected = torch.diag(torch.tensor([0., 1., 0.75, 1.]))
    expected[2, 1] = 0.5 / 2**0.5
    torch.testing.assert_close(matrix, expected)


def test_hard_pruning_and_structure_loss_update_both_branches():
    model = AdaptiveGraphPruner(1, threshold=0.75, rho=0.65, layers=1)
    for parameter in model.parameters():
        torch.nn.init.ones_(parameter)
    features = torch.tensor([[0.], [1.], [2.], [0.]])
    edges = torch.tensor([[1, 3], [2, 3]])
    weights = torch.tensor([0.2, 0.4])
    out = model(features, edges, weights)
    assert out.keep_mask.tolist() == [True, False]
    torch.testing.assert_close(out.edge_index, edges[:, :1])
    torch.testing.assert_close(out.edge_weight, weights[:1])
    torch.testing.assert_close(out.structure_loss, (out.scores.mean() - 0.65).abs())
    out.structure_loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    model.threshold = 0.5
    assert model(features, edges, weights).keep_mask.tolist() == [True, True]


def test_empty_graph_has_finite_zero_structure_loss():
    model = AdaptiveGraphPruner(2, threshold=0.6, rho=0.7)
    out = model(torch.zeros(3, 2), torch.empty(2, 0, dtype=torch.long), torch.empty(0))
    assert out.structure_loss.item() == 0 and out.edge_index.shape == (2, 0)
