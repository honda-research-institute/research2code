# paper-element: eq-classifier-prediction

"""Paper-level check for neural classifier prediction."""

import torch

from method.model import GradientEmbeddingMLP


def test_forward_prediction_is_argmax_of_logits():
    """The predicted class is the index of the largest class score."""
    model = GradientEmbeddingMLP(input_dim=2, n_classes=3, hidden_dim=2)
    with torch.no_grad():
        model.encoder[0].weight.copy_(torch.eye(2))
        model.encoder[0].bias.zero_()
        model.classifier.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]])
        )
        model.classifier.bias.copy_(torch.tensor([0.0, 0.0, 0.0]))

    logits = model.forward(torch.tensor([[1.0, 2.0]]))

    assert torch.argmax(logits, dim=1).tolist() == [1]
