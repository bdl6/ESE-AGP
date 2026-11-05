import torch

from ese_agp.models.recommender import LocalSessionRecommender


def test_candidates_choose_their_own_view_across_chunks():
    model = LocalSessionRecommender(3, dim=3, num_views=3, candidate_chunk_size=2)
    with torch.no_grad():
        model.embedding.weight[1:].copy_(torch.eye(3))
        for view, bias in zip([model.view, *model.extra_views], ([3, 0, 1], [1, 4, 2], [2, 1, 5])):
            view.weight.zero_()
            view.bias.copy_(torch.tensor(bias))
    logits = model.score_candidates(torch.zeros(1, 3))
    torch.testing.assert_close(logits, torch.tensor([[3., 4., 5.]]))
    logits.sum().backward()
    for i, view in enumerate([model.view, *model.extra_views]):
        torch.testing.assert_close(view.bias.grad, torch.eye(3)[i])
