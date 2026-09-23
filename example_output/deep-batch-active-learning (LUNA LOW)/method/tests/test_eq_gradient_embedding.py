# paper-element: eq-gradient-embedding

from method.method import compute_gradient_embeddings


class _FixedClassifier:
    def __init__(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def train(self, mode: bool = True) -> None:
        self.training = mode

    def forward_with_embedding(self, x):
        import torch

        return torch.tensor([[0.0, 0.0]]), torch.tensor([[2.0, -1.0]])


def test_gradient_embedding_is_probability_error_times_penultimate_feature():
    import torch

    model = _FixedClassifier()
    result = compute_gradient_embeddings(model, torch.zeros(1, 1))

    """Equal probabilities imply predicted class 0 and the Eq. (1) values."""
    expected = torch.tensor([[-1.0, 0.5, 1.0, -0.5]])
    torch.testing.assert_close(result, expected)
    assert result[0, 0].item() == -1.0
