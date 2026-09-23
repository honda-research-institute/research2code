"""Training protocol for the BADGE active-learning classifier.

The classifier uses Adam and cross-entropy with full-batch updates. Seeds are
set once at entry and threaded by callers for reproducible rounds (paper
training protocol, Section 4).
"""

from __future__ import annotations

import torch
from torch import nn

from .model import BADGEClassifier


def build_model(input_dim: int, n_classes: int, hidden_dim: int = 64) -> BADGEClassifier:
    """Construct the smoke-scale BADGE-compatible classifier."""
    return BADGEClassifier(input_dim=input_dim, n_classes=n_classes, hidden_dim=hidden_dim)


def train_from_scratch(
    model: BADGEClassifier,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    learning_rate: float = 0.001,
    max_epochs: int = 50,
    train_until_accuracy: float | None = 0.99,
    seed: int = 0,
) -> BADGEClassifier:
    """Retrain from scratch using full-batch Adam and cross-entropy."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(max_epochs):
        optimizer.zero_grad()
        loss_fn(model(x_train), y_train).backward()
        optimizer.step()
        if train_until_accuracy is not None:
            model.eval()
            with torch.no_grad():
                accuracy = (model(x_train).argmax(1) == y_train).float().mean().item()
            model.train()
            if accuracy >= train_until_accuracy:
                break
    return model
