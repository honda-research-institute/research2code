# paper-element: eq-classifier-cross-entropy

import torch

from method.model import BADGEClassifier


def test_classifier_forward_produces_class_probabilities():
    """Forward returns logits whose target CE equals log reciprocal probability."""
    model = BADGEClassifier(input_dim=2, n_classes=2, hidden_dim=2)
    with torch.no_grad():
        model.encoder[0].weight.copy_(torch.eye(2))
        model.encoder[0].bias.zero_()
        model.classifier.weight.copy_(torch.tensor([[2.0, 0.0], [-1.0, 0.0]]))
        model.classifier.bias.zero_()

    scores = model.forward(torch.tensor([[1.0, 0.0]]))
    target = torch.tensor([0])
    probability = torch.softmax(scores, dim=1)[0, target.item()]
    loss = torch.nn.functional.cross_entropy(scores, target)

    assert torch.argmax(scores, dim=1).item() == 0
    assert torch.allclose(loss, -torch.log(probability), atol=1e-6)
