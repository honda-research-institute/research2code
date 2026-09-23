"""BADGE (Ash et al.) classifier: :class:`GradientEmbeddingMLP`.

The paper evaluates MLP, ResNet-18, and VGG-11 classifiers. This smoke-scale
package ships a two-affine-layer MLP for flattened MNIST/tabular features; it
preserves BADGE's real penultimate activation and raw-logit multiclass output,
but not the paper's image-model architecture results.

Prerequisites for swapping the architecture
-------------------------------------------
A replacement must accept the same flattened feature width, return raw
multiclass logits from ``forward(x)``, and provide
``forward_with_embedding(x) -> (logits, z)`` where ``z`` is the real activation
immediately before its final linear classifier. Without that hook (or if it
returns logits/input features instead), BADGE's final-layer gradient embedding
has the wrong diversity geometry and selection silently degenerates toward
pure uncertainty.
"""

from __future__ import annotations

import torch
from torch import nn


class GradientEmbeddingMLP(nn.Module):
    """Two-layer raw-logit classifier with BADGE's final-layer interface."""

    # essential: Penultimate-layer embedding hook
    # essential: Multiclass output layer
    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw multiclass logits (no trailing softmax)."""
        # paper-element: eq-classifier-prediction
        return self.classifier(self.encoder(x))

    def forward_with_embedding(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and the real penultimate activation used by BADGE."""
        # paper-element: alg-gradient-embedding
        # paper-element: eq-softmax-gradient-block
        z = self.encoder(x)
        return self.classifier(z), z
