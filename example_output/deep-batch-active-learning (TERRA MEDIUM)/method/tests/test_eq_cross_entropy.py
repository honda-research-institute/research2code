"""Paper-element test for cross-entropy training loss."""

# paper-element: eq-cross-entropy

import math
from typing import cast

import pytest
import torch
from torch import nn

import method.training as training
from method.model import GradientEmbeddingMLP
from method.training import train_from_scratch


class _FixedLogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([[1.0, 2.0]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(x.shape[0], -1)


def test_training_uses_predicted_probability_cross_entropy(monkeypatch) -> None:
    observed_losses: list[float] = []

    class RecordingCrossEntropy(nn.Module):
        def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            observed_losses.append(loss.item())
            return loss

    monkeypatch.setattr(training.nn, "CrossEntropyLoss", RecordingCrossEntropy)
    model = _FixedLogitModel()
    train_from_scratch(
        cast(GradientEmbeddingMLP, model),
        torch.zeros((1, 1)),
        torch.tensor([0]),
        max_epochs=1,
        train_until_accuracy=None,
        seed=0,
    )

    probability_of_label_zero = 1.0 / (1.0 + math.exp(1.0))
    assert observed_losses == [pytest.approx(-math.log(probability_of_label_zero))]
