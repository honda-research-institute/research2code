"""Retraining protocol for BADGE's classifier.

The paper retrains from scratch with cross-entropy and SGD-family
optimization until training accuracy exceeds 99 percent; the paper describes
the Adam variant of SGD.  This smoke-scale implementation uses Adam with the
0.001 image-data default and repeated full-batch updates within each nominal
epoch so the eight-epoch smoke cap can learn a small labeled set.  The
notebook seeds once and passes a round-specific seed through this entry point.
Reference: paper Experimental Setup, Section 4.
"""

from __future__ import annotations

import torch

from .model import GradientEmbeddingClassifier


def build_model(input_dim: int, n_classes: int, hidden_dim: int = 64) -> GradientEmbeddingClassifier:
    """Construct the MLP while preserving the final linear classifier."""
    return GradientEmbeddingClassifier(
        input_dim=input_dim, n_classes=n_classes, hidden_dim=hidden_dim
    )


def train_from_scratch(
    model: GradientEmbeddingClassifier,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    learning_rate: float = 0.001,
    max_epochs: int = 50,
    train_until_accuracy: float | None = 0.99,
    seed: int = 0,
) -> GradientEmbeddingClassifier:
    """Retrain ``model`` from a fresh initialization on the labeled set."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    for layer in model.modules():
        reset = getattr(layer, "reset_parameters", None)
        if callable(reset):
            reset()

    # The paper's stated protocol is the Adam variant of SGD.  Four full-batch
    # updates per nominal epoch keep the smoke cap trainable without changing
    # the retrain-from-scratch contract; paper-scale runs should use a
    # DataLoader and the paper's benchmark-specific schedule instead.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    # Full-batch updates: the entire labeled set is one batch. Four updates per
    # nominal epoch are fine for smoke-sized labeled sets (<= ~1k examples).
    for _ in range(max_epochs):
        for _ in range(4):
            optimizer.zero_grad()
            loss = loss_fn(model(x_train), y_train)
            loss.backward()
            optimizer.step()

        if train_until_accuracy is not None:
            model.eval()
            with torch.no_grad():
                accuracy = (
                    (model(x_train).argmax(dim=1) == y_train).float().mean().item()
                )
            model.train()
            if accuracy >= train_until_accuracy:
                break
    return model
