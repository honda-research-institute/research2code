"""BADGE batch active learning.

This module implements gradient embeddings and k-MEANS++ seeding for selecting
a diverse, uncertain batch from an unlabeled pool. Public functions expose the
gradient-embedding helper, the k-MEANS++ helper, and the paper's ``select_batch``
acquisition entry point.

Based on Ash et al., "Deep Batch Active Learning by Diverse, Uncertain Gradient
Lower Bounds" (ICLR 2020).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def compute_gradient_embeddings(model: Any, x_unlabeled: Any) -> np.ndarray:
    """Compute one hallucinated final-layer gradient per pool example.

    **Eq. 1, Section 3**: for probabilities ``p`` and penultimate features
    ``z``, the class block is ``(p_i - 1[y_hat=i]) * z``.  The result is a
    NumPy array with shape ``(pool_size, class_count * feature_width)``.
    ``model`` must expose ``forward_with_embedding`` returning logits and the
    actual penultimate features immediately before its final linear layer.
    """
    # paper-element: eq-gradient-embedding
    # paper-element: eq-gradient-block
    # paper-element: concept-hallucinated-label
    # essential: Final-layer gradient embedding access
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            logits, features = model.forward_with_embedding(x_unlabeled)
            # paper-element: eq-softmax-model
            # paper-element: eq-softmax-definition
            probabilities = F.softmax(logits, dim=-1)
            predicted_labels = probabilities.argmax(dim=-1)
            residual = probabilities.clone()
            residual.scatter_(1, predicted_labels.unsqueeze(1),
                              residual.gather(1, predicted_labels.unsqueeze(1)) - 1.0)
            embeddings = (residual.unsqueeze(-1) * features.unsqueeze(1)).reshape(
                features.shape[0], -1
            )
            # paper-element: concept-gradient-uncertainty
            # paper-element: concept-gradient-diversity
            # paper-element: concept-quality-diversity-tradeoff
            return embeddings.detach().cpu().numpy()
    finally:
        model.train(was_training)


def kmeans_plus_plus_seeding(
    embeddings: np.ndarray, batch_size: int, *, seed: int
) -> list[int]:
    """Select unique embedding centers with k-MEANS++ (Appendix A, Algorithm 2).

    The first center is uniform; each later center is sampled proportional to
    squared distance from its nearest selected center.  Distances are updated
    incrementally, avoiding an ``(N, batch_size, D)`` broadcast.
    """
    # paper-element: alg-kmeans-plus-plus
    points = np.asarray(embeddings, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"embeddings must be a 2-D array, got shape {points.shape}")
    n_points = points.shape[0]
    if n_points == 0:
        raise ValueError("cannot select from an empty unlabeled pool")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    # paper-fidelity: smoke-sized pools may be smaller than the requested paper batch.
    count = min(int(batch_size), n_points)
    rng = np.random.default_rng(seed)
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(rng.integers(n_points))
    nearest_sq = np.sum((points - points[selected[0]]) ** 2, axis=1)
    chosen = {int(selected[0])}
    for position in range(1, count):
        total = float(nearest_sq.sum())
        if not np.isfinite(total) or total <= 0.0:
            remaining = np.array(
                [index for index in range(n_points) if index not in chosen],
                dtype=np.int64,
            )
            selected[position] = int(rng.choice(remaining))
        else:
            probabilities = nearest_sq / total
            selected[position] = int(rng.choice(n_points, p=probabilities))
            while int(selected[position]) in chosen:
                selected[position] = int(rng.choice(n_points, p=probabilities))
        chosen.add(int(selected[position]))
        new_sq = np.sum((points - points[selected[position]]) ** 2, axis=1)
        nearest_sq = np.minimum(nearest_sq, new_sq)
    return selected.tolist()


def select_batch(model: Any, x_unlabeled: Any, batch_size: int, seed: int) -> list[int]:
    """Select a BADGE batch by composing gradient embeddings and k-MEANS++.

    **Algorithm 1, Section 3**: each unlabeled point is represented by its
    hallucinated-label last-layer gradient embedding, then diverse centers are
    selected with k-MEANS++ seeding.
    """
    # paper-element: alg-badge
    embeddings = compute_gradient_embeddings(model, x_unlabeled)
    return kmeans_plus_plus_seeding(embeddings, batch_size, seed=seed)
