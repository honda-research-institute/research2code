"""
Training utilities for GBALD.

Protocol: Adam optimizer, retrain from scratch each round, train until
accuracy reaches target or max_epochs is exhausted. The paper follows
BatchBALD [5] which uses Adam with dataset-dependent learning rates.
Learning rate defaults to 1e-3 (standard for Adam on image data).

Seeding: torch.manual_seed(seed) is called at function entry before any
optimizer/model state changes to ensure reproducibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .model import MCDropoutMLP


def build_model(
    input_dim: int,
    n_classes: int,
    hidden_dim: int = 128,
    num_hidden_layers: int = 2,
    dropout_rate: float = 0.5,
) -> MCDropoutMLP:
    """Build the MCDropoutMLP architecture.

    Args:
        input_dim: Dimension of input features.
        n_classes: Number of output classes.
        hidden_dim: Dimension of hidden layers.
        num_hidden_layers: Number of hidden layers.
        dropout_rate: Dropout probability (0.5 matches the paper).

    Returns:
        A fresh MCDropoutMLP instance.
    """
    return MCDropoutMLP(
        input_dim=input_dim,
        n_classes=n_classes,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        dropout_rate=dropout_rate,
    )


def train_from_scratch(
    model: MCDropoutMLP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    learning_rate: float = 1e-3,
    max_epochs: int = 50,
    train_until_accuracy: float = 0.99,
    seed: int = 0,
) -> MCDropoutMLP:
    """Train the model from scratch on the given data.

    Full-batch gradient descent: the entire labeled set is one batch per epoch.
    Fine for smoke-sized labeled sets (≤ ~1k examples). At paper-scale labeled
    sets with the paper's full architecture, wrap (x_train, y_train) in a
    DataLoader and step per mini-batch instead.

    Protocol: Adam optimizer, CrossEntropyLoss, retrain from scratch.
    Stops early if train_until_accuracy is reached.

    Args:
        model: The MCDropoutMLP to train.
        x_train: Training features of shape (N, input_dim).
        y_train: Training labels of shape (N,) with values in [0, n_classes-1].
        learning_rate: Adam learning rate. Default 1e-3 (standard for image data).
        max_epochs: Maximum training epochs.
        train_until_accuracy: Target training accuracy to stop early.
        seed: Random seed for reproducibility.

    Returns:
        The trained model.
    """
    # Seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Prepare data loader (full-batch)
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=len(x_train), shuffle=False)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        acc = correct / total if total > 0 else 0.0

        # Early stopping
        if acc >= train_until_accuracy:
            break

    return model
