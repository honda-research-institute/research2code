"""BADGE: deep batch active learning by diverse, uncertain gradient lower bounds.

Public functions compute hallucinated last-layer gradient embeddings, seed
k-MEANS++ centers, and compose them into one acquisition batch. Based on
Ash et al., *Deep Batch Active Learning by Diverse, Uncertain Gradient Lower
Bounds*.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def compute_gradient_embeddings(model: Any, x_unlabeled: torch.Tensor) -> torch.Tensor:
    """Compute hallucinated-label last-layer gradients (Eq. 1, Section 3).

    Args:
        model: Trained classifier exposing ``forward_with_embedding``.
        x_unlabeled: Pool tensor with one row per candidate.
    Returns:
        Tensor whose rows are flattened classifier-weight gradients, with
        shape ``(pool_size, class_count * embedding_width)``.
    """
    # paper-element: eq-gradient-embedding
    # paper-element: eq-softmax-cross-entropy
    # essential: Final output layer with differentiable cross-entropy scores
    # essential: Penultimate-layer representation
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits, embedding = model.forward_with_embedding(x_unlabeled)
            probabilities = F.softmax(logits, dim=1)
            predicted = probabilities.argmax(dim=1)
            one_hot = torch.zeros_like(probabilities)
            one_hot.scatter_(1, predicted[:, None], 1.0)
            classifier_gradient = probabilities - one_hot
            return (classifier_gradient[:, :, None] * embedding[:, None, :]).reshape(
                embedding.shape[0], -1
            )
    finally:
        model.train(was_training)


def kmeans_plus_plus_seeding(
    embeddings: np.ndarray, batch_size: int, *, rng: np.random.Generator
) -> np.ndarray:
    """Select diverse centers with sequential k-MEANS++ seeding (Algorithm 1).

    Distances are maintained incrementally as the nearest-center squared
    distance, which is algebraically equivalent to the paper's ``D(x)^2``.
    """
    # paper-element: alg-kmeans-plus-plus
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    n_points = embeddings.shape[0]
    if n_points == 0:
        return np.empty(0, dtype=np.int64)
    count = min(max(int(batch_size), 0), n_points)
    if count == 0:
        return np.empty(0, dtype=np.int64)

    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(rng.integers(n_points))
    min_squared_distance = np.sum(
        (embeddings - embeddings[selected[0]]) ** 2, axis=1
    )
    for position in range(1, count):
        total = float(min_squared_distance.sum())
        if not np.isfinite(total) or total <= 0.0:
            # paper-fidelity: duplicate/equidistant smoke pools have no
            # meaningful k-MEANS++ distribution; choose an unselected point.
            available = np.setdiff1d(np.arange(n_points), selected[:position])
            selected[position] = int(rng.choice(available))
        else:
            probabilities = min_squared_distance / total
            selected[position] = int(rng.choice(n_points, p=probabilities))
        new_squared_distance = np.sum(
            (embeddings - embeddings[selected[position]]) ** 2, axis=1
        )
        min_squared_distance = np.minimum(min_squared_distance, new_squared_distance)
    return selected


def select_batch(model: Any, x_unlabeled: torch.Tensor, batch_size: int, *, seed: int) -> torch.Tensor:
    """Return pool indices selected by BADGE (Algorithm 1, Section 3).

    The requested count is clamped only to the available pool, an
    accommodation for smoke-sized pools that is a no-op at paper scale.
    """
    # paper-element: alg-badge
    rng = np.random.default_rng(seed)
    embeddings = compute_gradient_embeddings(model, x_unlabeled)
    selected = kmeans_plus_plus_seeding(
        embeddings.detach().cpu().numpy(), batch_size, rng=rng
    )
    return torch.as_tensor(selected, dtype=torch.int64, device=x_unlabeled.device)
