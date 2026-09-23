"""GradientEmbeddingClassifier for *Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds*.

The paper evaluates BADGE with image and text classifiers, including a
ResNet-style SVHN benchmark.  This package ships a two-layer MLP over the
flattened input produced by ``data.load_data`` rather than reproducing a
benchmark backbone.  It preserves the contract BADGE needs: raw logits, a
linear final layer, and the real penultimate representation used to form
last-layer gradient embeddings.

Prerequisites for swapping the architecture
---------------------------------------------
A replacement must be a differentiable classifier whose ``forward`` accepts
the declared ``(batch, input_dim)`` tensor and returns raw ``(batch,
n_classes)`` logits.  It must also implement
``forward_with_embedding(x) -> (logits, penultimate_features)`` with features
from immediately before the final linear classifier.  Without that hook,
BADGE cannot compute final-layer gradient embeddings and silently degenerates
to a different acquisition rule; returning logits as the embedding removes
diversity from the selection.
"""

from __future__ import annotations

import torch
from torch import nn


class GradientEmbeddingClassifier(nn.Module):
    """Classifier exposing logits and the penultimate embedding."""

    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits for a batch of flattened examples."""
        return self.classifier(self.encoder(x))

    # essential: Final-layer gradient embedding access
    # paper-element: eq-gradient-embedding
    # paper-element: eq-gradient-block
    def forward_with_embedding(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and the penultimate features used by BADGE."""
        z = self.encoder(x)
        return self.classifier(z), z
