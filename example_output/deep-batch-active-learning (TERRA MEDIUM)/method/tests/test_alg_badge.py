# paper-element: alg-badge
"""Fixture-scale check for Algorithm 1 BADGE selection."""

from method.method import select_batch

torch = __import__("torch")


class _FixedGradientModel(torch.nn.Module):
    def forward_with_embedding(self, x):
        logits = torch.tensor(
            [[3.0, 0.0], [0.0, 3.0], [1.0, 1.0]], dtype=x.dtype
        )
        embeddings = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=x.dtype
        )
        return logits, embeddings


def test_badge_queries_every_pool_example_when_batch_equals_pool():
    """Algorithm 1 chooses B gradient embeddings from the unlabeled pool."""
    pool = torch.zeros((3, 1), dtype=torch.float32)

    selected = select_batch(_FixedGradientModel(), pool, batch_size=3, seed=17)

    assert set(selected) == {0, 1, 2}
