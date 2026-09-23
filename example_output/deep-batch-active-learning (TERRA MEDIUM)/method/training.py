"""Fresh BADGE classifier training (Section 4).

Training uses cross-entropy with Adam (learning rate 0.001 for the bundled
flattened MNIST/image data) until 99% training accuracy or ``max_epochs``.
The notebook seeds once, then passes ``seed`` (or its round-specific offset)
before each fresh model's optimization.
"""

from __future__ import annotations

import torch
from torch import nn

from .model import GradientEmbeddingMLP


def build_model(input_dim: int, n_classes: int, **arch_kwargs: int) -> GradientEmbeddingMLP:
    """Build the BADGE-compatible smoke classifier (hidden_dim=256 by default)."""
    return GradientEmbeddingMLP(input_dim, n_classes, **arch_kwargs)


def train_from_scratch(
    model: GradientEmbeddingMLP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    learning_rate: float = 1e-3,
    max_epochs: int = 50,
    train_until_accuracy: float | None = 0.99,
    seed: int = 0,
) -> GradientEmbeddingMLP:
    """Train this freshly constructed classifier on the current labeled set."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Full-batch gradient descent: the entire labeled set is one batch per epoch.
    # Fine for smoke-sized labeled sets (≤ ~1k examples). At paper-scale labeled
    # sets with the paper's full architecture, wrap (x_train, y_train) in a
    # DataLoader and step per mini-batch instead.
    model.train()
    for _ in range(max_epochs):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        # paper-element: eq-cross-entropy
        loss.backward()
        optimizer.step()
        if train_until_accuracy is not None:
            model.eval()
            with torch.no_grad():
                accuracy = (model(x_train).argmax(dim=1) == y_train).float().mean().item()
            model.train()
            if accuracy >= train_until_accuracy:
                break
    return model
