"""BADGE architecture for deep batch active learning.

This package ships a compact MLP instead of the paper's larger image
backbone, while preserving logits and penultimate representations required by
the acquisition method.

Prerequisites for swapping the architecture
--------------------------------------------
Replacement models must accept ``(batch, input_dim)`` float tensors, return
raw ``(batch, n_classes)`` logits, and expose ``forward_with_embedding`` with
the same representation contract. Without the embedding, gradient embeddings
cannot be computed and BADGE silently breaks; without a differentiable final
layer, uncertainty and class-gradient scores silently become invalid.
"""

from __future__ import annotations

import torch
from torch import nn


class BADGEClassifier(nn.Module):
    """Small classifier exposing logits and penultimate-layer features."""

    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        # essential: Penultimate-layer representation
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward_with_embedding(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw logits and the penultimate representation."""
        z = self.encoder(x)
        # paper-element: eq-gradient-embedding
        return self.classifier(z), z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits for cross-entropy and acquisition."""
        # essential: Final output layer with differentiable cross-entropy scores
        # paper-element: eq-classifier-cross-entropy
        return self.forward_with_embedding(x)[0]
