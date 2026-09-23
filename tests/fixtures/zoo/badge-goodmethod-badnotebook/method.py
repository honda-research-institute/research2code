"""
BADGE (Batch Active learning by Diverse Gradient Embeddings) implementation.

This module implements the BADGE active learning algorithm from "Deep Batch Active
Learning by Diverse, Uncertain Gradient Lower Bounds" (Ash et al., 2020). BADGE
selects diverse, uncertain batches by computing gradient embeddings with respect
to the last layer (using hallucinated labels) and sampling via k-MEANS++ seeding.

**Public functions:**
- `select_batch`: Main BADGE acquisition function (pluggable component)
- `compute_gradient_embeddings`: Compute gradient embeddings for unlabeled examples
- `kmeans_plus_plus_seeding`: k-MEANS++ seeding for diverse batch selection

**Algorithm overview (Algorithm 1, Section 3):**
1. For each unlabeled example x, compute its hallucinated label ŷ = h(x)
2. Compute gradient embedding g_x = ∂/∂θ_out loss_CE(f(x;θ), ŷ) at θ=θ_current
3. Select a batch of size B from {g_x} using k-MEANS++ seeding

The gradient embedding captures both uncertainty (via gradient magnitude) and
representation (via penultimate layer output). k-MEANS++ automatically balances
uncertainty and diversity without hyperparameters.

Paper reference: Ash, J. T. et al. (2020). Deep Batch Active Learning by Diverse,
Uncertain Gradient Lower Bounds.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import List


def compute_gradient_embeddings(
    model: torch.nn.Module,
    x_unlabeled: torch.Tensor
) -> torch.Tensor:
    """
    Compute gradient embeddings for unlabeled examples (Eq. 1, Section 3).
    
    # paper-element: eq-gradient-embedding
    # paper-element: concept-hallucinated-label
    # paper-element: concept-gradient-uncertainty
    # essential: forward_with_embedding method
    
    For each example x, the gradient embedding g_x captures the gradient of
    cross-entropy loss with respect to the last layer weights, using the
    model's predicted (hallucinated) label ŷ = argmax(p). This provides a
    lower bound on the gradient norm for any true label (Proposition 1).
    
    The embedding is computed as:
        g_x_i = (p_i - I(ŷ = i)) * z(x; V)
    where:
        - p = softmax(W * z(x; V)) is the predicted probability vector
        - ŷ = argmax(p) is the hallucinated label
        - z(x; V) is the penultimate layer output
        - I(·) is the indicator function
    
    The gradient embedding has shape (N, n_classes * hidden_dim), where each
    example's embedding is the concatenation of (p_i - I(ŷ = i)) * z for all i.
    
    Args:
        model: Trained model with forward_with_embedding method
        x_unlabeled: Unlabeled examples of shape (N, n_features) or (N, C, H, W)
        
    Returns:
        Gradient embeddings of shape (N, n_classes * hidden_dim)
        
    Paper reference: Eq. 1, Section 3
    """
    model.eval()
    
    with torch.no_grad():
        # Get logits and penultimate embeddings
        # paper-element: eq-gradient-embedding - uses forward_with_embedding to get z(x; V)
        logits, embeddings = model.forward_with_embedding(x_unlabeled)
        
        # Compute predicted probabilities
        probs = F.softmax(logits, dim=1)  # Shape: (N, n_classes)
        
        # Get hallucinated labels (model's predictions)
        # paper-element: concept-hallucinated-label
        predicted_labels = logits.argmax(dim=1)  # Shape: (N,)
        
        # Compute gradient embedding: g_x_i = (p_i - I(ŷ = i)) * z(x; V)
        # For each class i, the gradient component is (p_i - I(ŷ = i)) times the embedding
        
        N = logits.size(0)
        n_classes = logits.size(1)
        hidden_dim = embeddings.size(1)
        
        # Create indicator matrix: I(ŷ = i) for all i
        indicator = F.one_hot(predicted_labels, num_classes=n_classes).float()  # Shape: (N, n_classes)
        
        # Compute (p_i - I(ŷ = i)) for all i
        # This is the gradient coefficient for each class
        grad_coeff = probs - indicator  # Shape: (N, n_classes)
        
        # Reshape for broadcasting: (N, n_classes, 1) * (N, 1, hidden_dim)
        grad_coeff = grad_coeff.unsqueeze(2)  # Shape: (N, n_classes, 1)
        embeddings_expanded = embeddings.unsqueeze(1)  # Shape: (N, 1, hidden_dim)
        
        # Compute gradient embedding for each class
        # g_x_i = (p_i - I(ŷ = i)) * z(x; V)
        gradient_components = grad_coeff * embeddings_expanded  # Shape: (N, n_classes, hidden_dim)
        
        # Concatenate all class components into a single embedding
        gradient_embeddings = gradient_components.view(N, -1)  # Shape: (N, n_classes * hidden_dim)
    
    return gradient_embeddings


def kmeans_plus_plus_seeding(
    embeddings: np.ndarray,
    batch_size: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    k-MEANS++ seeding algorithm for diverse batch selection (Algorithm 2, Appendix A).
    
    # paper-element: alg-kmeans-plus-plus
    # paper-element: concept-gradient-diversity
    
    Sequentially samples k centers from the ground set, where each new center is
    sampled with probability proportional to the squared distance from the nearest
    already-selected center. This ensures diversity while favoring high-magnitude
    points (uncertainty).
    
    Algorithm:
    1. Sample the first center uniformly at random
    2. For t = 2 to k:
       a. Compute D_t(x) = min_{c in C_{t-1}} ||x - c||_2 for all x
       b. Sample c_t with probability D_t(x)^2 / sum_{x} D_t(x)^2
       c. Add c_t to the center set
    3. Return the set of k centers
    
    This is preferred over k-DPP (which has similar statistical performance) due
    to computational efficiency: k-MEANS++ is O(k * N * d) vs. k-DPP's high-order
    polynomial complexity (Figure 1, Appendix G).
    
    Args:
        embeddings: Gradient embeddings of shape (N, d) where N is pool size
        batch_size: Number of examples to select (k in k-MEANS++)
        rng: NumPy random generator seeded from the function's seed parameter
        
    Returns:
        Indices of selected examples of shape (batch_size,)
        
    Paper reference: Algorithm 2, Appendix A
    
    # paper-fidelity: We use numpy's vectorized operations for efficiency rather
    # than the paper's pseudocode loop structure. The algorithm is identical.
    """
    N = embeddings.shape[0]
    
    # Clamp batch_size to available pool size
    # paper-fidelity: At paper scale (N_pool >> batch_size), this is a no-op.
    # At smoke scale (e.g., N_pool=5000, batch_size=50), this prevents errors.
    batch_size = min(batch_size, N)
    
    if batch_size <= 0:
        return np.array([], dtype=np.int64)
    
    if batch_size == 1:
        # Single selection: uniform random
        return np.array([rng.integers(0, N)])
    
    # Initialize: select first center uniformly at random
    selected_indices = [rng.integers(0, N)]
    
    # Iteratively select remaining centers
    for _ in range(1, batch_size):
        # Compute squared distances from each point to nearest selected center
        # embeddings: (N, d), selected: (t, d) -> distances: (N, t)
        selected_embeddings = embeddings[selected_indices]  # Shape: (t, d)
        
        # Compute pairwise squared Euclidean distances
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x·c
        embeddings_norm_sq = np.sum(embeddings ** 2, axis=1)  # Shape: (N,)
        selected_norm_sq = np.sum(selected_embeddings ** 2, axis=1)  # Shape: (t,)
        
        # Compute distances: (N, 1) + (1, t) - 2 * (N, t)
        distances_sq = embeddings_norm_sq[:, np.newaxis] + selected_norm_sq[np.newaxis, :] - 2 * np.dot(embeddings, selected_embeddings.T)
        
        # Ensure non-negative (numerical stability)
        distances_sq = np.maximum(distances_sq, 0.0)
        
        # D_t(x) = min squared distance to any selected center
        min_distances_sq = np.min(distances_sq, axis=1)  # Shape: (N,)
        
        # Sample next center with probability proportional to D_t(x)^2
        # Normalize to get probabilities
        total_distance_sq = np.sum(min_distances_sq)
        if total_distance_sq > 0:
            probabilities = min_distances_sq / total_distance_sq
        else:
            # Fallback to uniform if all distances are zero
            probabilities = np.ones(N) / N
        
        # Sample next center
        next_index = rng.choice(N, p=probabilities)
        selected_indices.append(next_index)
    
    return np.array(selected_indices, dtype=np.int64)


def select_batch(
    model: torch.nn.Module,
    x_unlabeled: torch.Tensor,
    batch_size: int,
    seed: int
) -> List[int]:
    """
    BADGE acquisition function: select diverse, uncertain batch (Algorithm 1, Section 3).
    
    # paper-element: alg-badge
    
    Implements the BADGE batch selection algorithm:
    1. Compute gradient embeddings for all unlabeled examples
    2. Select a diverse batch using k-MEANS++ seeding
    
    The gradient embedding captures uncertainty (via gradient magnitude) and
    representation (via penultimate layer output). k-MEANS++ automatically
    balances uncertainty and diversity without hyperparameters by favoring
    both high-magnitude points and diverse directions.
    
    Algorithm 1 (BADGE), Steps 4-7:
    - For each x in U \\ S (unlabeled pool):
      1. Compute hallucinated label ŷ(x) = h(x)
      2. Compute gradient embedding g_x
    - Sample batch S_t of size B from {g_x} using k-MEANS++ seeding
    
    Args:
        model: Trained model with forward_with_embedding method
        x_unlabeled: Unlabeled examples of shape (N, n_features) or (N, C, H, W)
        batch_size: Number of examples to select
        seed: Random seed for reproducibility
        
    Returns:
        List of indices (into x_unlabeled) of selected examples, length batch_size
        
    Paper reference: Algorithm 1, Section 3
    """
    # Create seeded random generator
    rng = np.random.default_rng(seed)
    
    # Compute gradient embeddings
    gradient_embeddings = compute_gradient_embeddings(model, x_unlabeled)
    
    # Convert to numpy for k-MEANS++
    gradient_embeddings_np = gradient_embeddings.detach().cpu().numpy()
    
    # Select diverse batch using k-MEANS++ seeding
    selected_indices = kmeans_plus_plus_seeding(
        gradient_embeddings_np,
        batch_size,
        rng
    )
    
    # Convert to list of ints
    return selected_indices.tolist()
