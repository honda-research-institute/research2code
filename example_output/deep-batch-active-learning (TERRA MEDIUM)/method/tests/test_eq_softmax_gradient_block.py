"""Tests for the per-class gradient embedding block."""

# paper-element: eq-softmax-gradient-block
import torch

from method.model import GradientEmbeddingMLP


def test_predicted_label_gradient_blocks_equal_residual_times_embedding() -> None:
    """Equation (1): (p_i - I(y_hat=i)) z forms each class block."""
    model = GradientEmbeddingMLP(input_dim=2, n_classes=3, hidden_dim=2)
    with torch.no_grad():
        model.encoder[0].weight.copy_(torch.eye(2))
        model.encoder[0].bias.zero_()
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()

    logits, embedding = model.forward_with_embedding(torch.tensor([[1.0, 2.0]]))
    predicted_label = logits.argmax(dim=1)
    torch.nn.functional.cross_entropy(logits, predicted_label).backward()

    expected_blocks = torch.tensor(
        [[-2.0 / 3.0, -4.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0]]
    )
    assert torch.allclose(embedding, torch.tensor([[1.0, 2.0]]))
    gradient_blocks = model.classifier.weight.grad
    assert gradient_blocks is not None
    assert torch.allclose(gradient_blocks, expected_blocks, atol=1e-6)
