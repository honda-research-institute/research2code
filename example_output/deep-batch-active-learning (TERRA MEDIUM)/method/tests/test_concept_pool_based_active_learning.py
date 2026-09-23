from types import SimpleNamespace

import torch

from method.method import select_batch

# paper-element: concept-pool-based-active-learning


def _pool_classifier():
    model = SimpleNamespace(training=True)
    model.eval = lambda: setattr(model, "training", False)
    model.train = lambda mode: setattr(model, "training", mode)
    model.forward_with_embedding = lambda x: (torch.stack((x[:, 0], -x[:, 0]), dim=1), x)
    return model


def test_select_batch_returns_positions_from_the_supplied_unlabeled_pool():
    """Pool-based acquisition selects examples from U, not new examples."""
    unlabeled_pool = torch.tensor([[3.0], [7.0], [11.0]], dtype=torch.float32)

    selected_positions = select_batch(
        _pool_classifier(), unlabeled_pool, batch_size=2, seed=9
    )

    selected_examples = unlabeled_pool[selected_positions, 0].tolist()
    assert sorted(selected_examples) in ([3.0, 7.0], [3.0, 11.0], [7.0, 11.0])
