# paper-element: alg-badge

import torch

from method.method import select_batch


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Linear(2, 2, bias=False)
        self.classifier = torch.nn.Linear(2, 2, bias=False)

    def forward_with_embedding(self, x):
        embedding = self.embedding(x)
        return self.classifier(embedding), embedding


def test_badge_returns_requested_distinct_pool_indices():
    model = TinyClassifier()
    x_unlabeled = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [-1.0, -1.0]],
        dtype=torch.float32,
    )
    selected = select_batch(model, x_unlabeled, batch_size=2, seed=7)

    assert selected.dtype == torch.int64
    assert selected.numel() == 2
    # BADGE's k-MEANS++ seeding uses the supplied seed for its first center.
    assert selected[0].item() == 3
    assert torch.all((selected >= 0) & (selected < x_unlabeled.shape[0]))
    assert torch.unique(selected).numel() == 2
