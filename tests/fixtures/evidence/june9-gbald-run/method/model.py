"""
Architecture for GBALD (Geometric Bayesian Active Learning by Disagreements).

Paper: "Bayesian Active Learning by Disagreements: A Geometric Perspective"
       by Xiaofeng Cao and Ivor W. Tsang (2022).

This module provides a simple MLP architecture with dropout layers for MC
dropout-based Bayesian inference. The paper's original architecture uses a
CNN-like structure (three blocks of conv/dropout/maxpool/relu), but for
demo-scale runs we use a standard MLP with dropout. The essential contract
is preserved: dropout layers active at inference for uncertainty estimation.

Prerequisites for swapping the architecture:
- Must have nn.Dropout layers (or nn.Dropout2d for image data).
- Must output raw logits (not softmax) for BALD scoring.
- Dropout rate should be 0.5 to match the paper's uncertainty calibration.
Without dropout layers active at inference, MC dropout produces identical
predictions, BALD scores collapse to zero, and the algorithm degenerates
to random sampling — the canonical silent-failure mode for Bayesian AL.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MCDropoutMLP(nn.Module):
    """MLP classifier with dropout layers for MC dropout Bayesian inference.

    # essential: Dropout layers active at inference
    # essential: Standard classifier head (logits output)
    # paper-element: concept-mc-dropout-inference
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        dropout_rate: float = 0.5,
    ) -> None:
        """
        Args:
            input_dim: Dimension of input features (e.g., 784 for flattened MNIST).
            n_classes: Number of output classes.
            hidden_dim: Dimension of hidden layers. Default 128 is reasonable for demo.
            num_hidden_layers: Number of hidden layers. Default 2.
            dropout_rate: Dropout probability. Default 0.5 matches the paper (Section 7.4).
        """
        super().__init__()

        self.dropout_rate = dropout_rate
        layers = []

        # Input to first hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))

        # Hidden to output (logits, no softmax)
        layers.append(nn.Linear(hidden_dim, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits.

        Args:
            x: Input tensor of shape (B, input_dim).

        Returns:
            Logits tensor of shape (B, n_classes).
        """
        return self.net(x)
