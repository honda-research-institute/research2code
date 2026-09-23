# paper-element: alg-gradient-embedding
"""Checks the hallucinated final-layer gradient embedding."""

import torch

from method.model import GradientEmbeddingMLP


def test_predicted_label_loss_gradient_has_residual_scaled_penultimate_blocks():
    """The predicted-label cross-entropy has a final-layer gradient."""
    model = GradientEmbeddingMLP(input_dim=2, n_classes=3, hidden_dim=2)
    with torch.no_grad():
        model.encoder[0].weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        model.encoder[0].bias.zero_()
        model.classifier.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
        )
        model.classifier.bias.zero_()

    logits, embedding = model.forward_with_embedding(torch.tensor([[2.0, 3.0]]))
    predicted_label = logits.argmax(dim=1)
    gradient = torch.autograd.grad(
        torch.nn.functional.cross_entropy(logits, predicted_label), model.classifier.weight
    )[0]

    assert torch.allclose(embedding, torch.tensor([[2.0, 3.0]]))
    assert torch.allclose(logits, torch.tensor([[2.0, 3.0, -1.0]]))
    expected_gradient = torch.tensor(
        [[0.5307756, 0.7961634], [-0.5572020, -0.8358030], [0.0264264, 0.0396396]]
    )
    assert torch.allclose(gradient, expected_gradient, atol=1e-6, rtol=1e-6)
