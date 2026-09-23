"""BADGE acquisition for *Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds*.

Public functions: ``compute_gradient_embeddings`` constructs predicted-label
last-layer gradient embeddings; ``kmeans_plus_plus_seeding`` samples diverse
centers; ``select_batch`` composes both stages for one acquisition round.

Ash, Zhang, Krishnamurthy, Langford, and Agarwal (2020).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def compute_gradient_embeddings(model, x_unlabeled: Tensor) -> Tensor:
    """Construct BADGE hallucinated output-layer gradients (Eq. 1, Section 3).

    Args:
        model: Multiclass classifier exposing ``forward_with_embedding``.
        x_unlabeled: Candidate inputs in their original pool order.

    Returns:
        A ``(n_candidates, n_classes * embedding_dim)`` gradient tensor.
    """
    # paper-element: alg-gradient-embedding
    # paper-element: eq-gradient-embedding
    # essential: Penultimate-layer embedding hook
    # essential: Multiclass output layer
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits, embedding = model.forward_with_embedding(x_unlabeled)
            if logits.ndim != 2 or embedding.ndim != 2:
                raise ValueError(
                    "forward_with_embedding must return rank-2 logits and "
                    "penultimate embeddings"
                )
            if logits.shape[0] != embedding.shape[0]:
                raise ValueError("logits and embeddings must have the same batch size")

            # paper-element: eq-classifier-prediction
            predicted_labels = logits.argmax(dim=1)
            # paper-element: eq-softmax-cross-entropy
            probabilities = F.softmax(logits, dim=1)
            # paper-element: eq-softmax-gradient-block
            residuals = probabilities - F.one_hot(
                predicted_labels, num_classes=logits.shape[1]
            ).to(dtype=probabilities.dtype)
            # Each class block is (p_i - 1{y_hat=i}) z(x; V).
            return (residuals.unsqueeze(-1) * embedding.unsqueeze(1)).flatten(start_dim=1)
    finally:
        model.train(was_training)


def kmeans_plus_plus_seeding(
    embeddings: np.ndarray, batch_size: int, rng: np.random.Generator
) -> List[int]:
    """Sample BADGE batch positions by k-MEANS++ (Appendix A, Algorithm 2).

    Args:
        embeddings: One gradient embedding per candidate.
        batch_size: Requested number of unique positions.
        rng: Local seeded generator for the k-MEANS++ draws.

    Returns:
        Unique positions into ``embeddings``.
    """
    # paper-element: alg-kmeanspp-seeding
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (n_candidates, embedding_dim)")
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")

    n_candidates = embeddings.shape[0]
    n_select = min(batch_size, n_candidates)
    # paper-fidelity: Clamping only accommodates a smoke pool smaller than the
    # requested batch; at paper scale this is a no-op.
    if n_select == 0:
        return []

    selected = [int(rng.integers(n_candidates))]
    # paper-element: eq-kmeanspp-distance
    differences = embeddings - embeddings[selected[0]]
    nearest_squared_distance = np.einsum("ij,ij->i", differences, differences)

    while len(selected) < n_select:
        selected_mask = np.zeros(n_candidates, dtype=bool)
        selected_mask[selected] = True
        weights = nearest_squared_distance.copy()
        weights[selected_mask] = 0.0
        total_weight = float(weights.sum())

        # paper-element: eq-kmeanspp-probability
        if total_weight > 0.0 and np.isfinite(total_weight):
            next_index = int(rng.choice(n_candidates, p=weights / total_weight))
        else:
            # Equal embeddings make Algorithm 2's all-zero distribution
            # undefined; sample uniformly from the still-unselected indices.
            remaining = np.flatnonzero(~selected_mask)
            next_index = int(remaining[rng.integers(len(remaining))])

        selected.append(next_index)
        differences = embeddings - embeddings[next_index]
        new_squared_distance = np.einsum("ij,ij->i", differences, differences)
        nearest_squared_distance = np.minimum(
            nearest_squared_distance, new_squared_distance
        )

    return selected


def select_batch(model, x_unlabeled, batch_size, seed) -> List[int]:
    """Select a diverse uncertain BADGE batch (Algorithm 1, Section 3).

    Args:
        model: Trained multiclass classifier for this acquisition round.
        x_unlabeled: Candidate pool whose positions are returned.
        batch_size: Requested number of candidates.
        seed: Per-round seed for local k-MEANS++ randomness.

    Returns:
        Unique candidate positions in the input pool.
    """
    # paper-element: alg-badge
    # paper-element: concept-diverse-uncertain-batches
    # paper-element: concept-pool-based-active-learning
    if x_unlabeled.shape[0] == 0 or batch_size <= 0:
        return []

    gradient_embeddings = compute_gradient_embeddings(model, x_unlabeled)
    embedding_array = gradient_embeddings.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    return kmeans_plus_plus_seeding(embedding_array, batch_size, rng)
