# paper-element: concept-diverse-uncertain-batches
"""Joint uncertainty-and-diversity selection in gradient-embedding space."""

torch = __import__("torch")

from method.method import select_batch


class _SeparatedGradientModel(torch.nn.Module):
    def forward_with_embedding(self, x):
        logits = torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 0.0]])
        embeddings = torch.tensor([[0.0], [0.0], [5.0]])
        return logits, embeddings


def test_select_batch_uses_gradient_magnitude_and_separation():
    """k-MEANS++ must select the sole nonzero separated gradient after center 1."""
    pool = torch.zeros((3, 1))

    selected = select_batch(_SeparatedGradientModel(), pool, batch_size=2, seed=1)

    assert selected == [1, 2]
