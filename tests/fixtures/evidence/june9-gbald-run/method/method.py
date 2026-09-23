"""
GBALD (Geometric Bayesian Active Learning by Disagreements) acquisition function.

Paper: "Bayesian Active Learning by Disagreements: A Geometric Perspective"
       by Xiaofeng Cao and Ivor W. Tsang (2022).

This module implements the two-stage GBALD algorithm:

Stage 1 (core-set construction): Constructs an initial core-set on an ellipsoid
geometry using a distance-based geometric prior (Eq. 5) and ellipsoid geodesic
rescaling (Eq. 11). This provides representative samples that cover the input
distribution, relieving the uninformative prior problem.

Stage 2 (model uncertainty estimation): Uses MC dropout to score unlabeled
candidates by BALD mutual information (Eq. 12), then ranks those candidates
by their geometric representativeness relative to the labeled set (Eq. 13-14)
to avoid redundant selections.

Public functions:
- select_batch: Main pluggable function implementing the full two-stage GBALD algorithm.
- compute_geometric_prior: Computes the geometric prior probability p(y|x,theta) per Eq. (5).
- compute_bald_scores: Computes BALD mutual information scores using MC dropout.
- geometric_ranking: Ranks BALD candidates by geometric representativeness per Eq. (13-14).
- construct_ellipsoid_core_set: Constructs the Stage 1 core-set using ellipsoid geodesic search.

# paper-element: concept-gbald-framework
# paper-element: concept-ellipsoid-core-set
# paper-element: concept-mc-dropout-inference
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans


def compute_geometric_prior(
    x: torch.Tensor,
    x_labeled: torch.Tensor,
    R_0: float,
) -> torch.Tensor:
    """Compute geometric prior probability p(y|x,theta) per Eq. (5).

    For each unlabeled sample x_i, computes:
      p(y_i|x_i,theta) = 1.0 if exists j: ||x_i - D_j|| <= R_0
                        = R_0 / min_j ||x_i - D_j|| otherwise

    where D_j are the labeled samples.

    # paper-element: eq-geometric-prior
    # essential: Dropout layers active at inference (used in Stage 2, not here)
    # paper-fidelity: R_0 assumes raw [0, 255] pixel scale (MNIST); see select_batch for validation

    Args:
        x: Unlabeled samples of shape (N_unlabeled, n_features).
        x_labeled: Labeled samples of shape (N_labeled, n_features).
        R_0: Distance threshold for the geometric prior. Assumes raw [0, 255] pixel
            values by default (paper's MNIST scale). Must be calibrated for the data scale.

    Returns:
        Geometric prior probabilities of shape (N_unlabeled,).
    """
    # Compute pairwise distances: ||x_i - D_j|| for all i, j
    # x: (N_unlabeled, n_features), x_labeled: (N_labeled, n_features)
    # Result: (N_unlabeled, N_labeled)
    x = x.float()
    x_labeled = x_labeled.float()

    # ||x_i - D_j||^2 = ||x_i||^2 + ||D_j||^2 - 2 * x_i · D_j
    x_norm_sq = torch.sum(x ** 2, dim=1, keepdim=True)  # (N_unlabeled, 1)
    labeled_norm_sq = torch.sum(x_labeled ** 2, dim=1)  # (N_labeled,)
    cross_term = torch.mm(x, x_labeled.t())  # (N_unlabeled, N_labeled)

    dist_sq = x_norm_sq + labeled_norm_sq - 2 * cross_term
    dist_sq = torch.clamp(dist_sq, min=0.0)  # Numerical stability
    distances = torch.sqrt(dist_sq)  # (N_unlabeled, N_labeled)

    # Find min distance to any labeled sample for each unlabeled sample
    min_distances = torch.min(distances, dim=1).values  # (N_unlabeled,)

    # Compute geometric prior: 1.0 if min_dist <= R_0, else R_0 / min_dist
    probs = torch.where(
        min_distances <= R_0,
        torch.ones_like(min_distances),
        R_0 / (min_distances + 1e-8),  # epsilon for numerical stability
    )

    return probs


def compute_bald_scores(
    model: nn.Module,
    x_unlabeled: torch.Tensor,
    mc_samples: int,
) -> torch.Tensor:
    """Compute BALD mutual information scores using MC dropout.

    BALD score for sample x: I[y; θ | x, D] = H[p_mean] - E_θ[H[p_θ]]

    where:
      - p_mean = (1/T) Σ_t p(y|x, θ_t) is the averaged predictive probability
      - p_θ = p(y|x, θ_t) is the per-sample probability from MC forward pass t
      - H[p] = -Σ_y p(y) log p(y) is the Shannon entropy

    # paper-element: eq-bald-score
    # paper-element: eq-batch-returns
    # paper-element: concept-mc-dropout-inference
    # essential: Dropout layers active at inference

    Args:
        model: Trained model with dropout layers.
        x_unlabeled: Unlabeled samples of shape (N_unlabeled, n_features).
        mc_samples: Number of MC dropout forward passes.

    Returns:
        BALD scores of shape (N_unlabeled,).
    """
    # Ensure model is in train mode for dropout to be active
    # essential: MC dropout active at inference
    was_train = model.training
    model.train()

    try:
        N = x_unlabeled.size(0)
        device = x_unlabeled.device

        # Collect probabilities from all MC samples
        # probs_all: (mc_samples, N, n_classes)
        probs_all = []

        with torch.no_grad():
            for _ in range(mc_samples):
                logits = model(x_unlabeled)  # (N, n_classes)
                probs = torch.softmax(logits, dim=1)  # (N, n_classes)
                probs_all.append(probs)

        probs_all = torch.stack(probs_all, dim=0)  # (mc_samples, N, n_classes)

        # Averaged predictive probability: p_mean = (1/T) Σ_t p_θ
        p_mean = torch.mean(probs_all, dim=0)  # (N, n_classes)

        # Prior entropy: H[p_mean] = -Σ_y p_mean(y) log p_mean(y)
        # Add epsilon for numerical stability
        eps = 1e-8
        prior_entropy = -torch.sum(p_mean * torch.log(p_mean + eps), dim=1)  # (N,)

        # Posterior entropy for each sample: H[p_θ] = -Σ_y p_θ(y) log p_θ(y)
        posterior_entropy = -torch.sum(probs_all * torch.log(probs_all + eps), dim=2)  # (mc_samples, N)

        # Expected posterior entropy: E_θ[H[p_θ]] = (1/T) Σ_t H[p_θ_t]
        expected_posterior_entropy = torch.mean(posterior_entropy, dim=0)  # (N,)

        # BALD score: I[y; θ | x, D] = H[p_mean] - E_θ[H[p_θ]]
        bald_scores = prior_entropy - expected_posterior_entropy  # (N,)

        return bald_scores

    finally:
        # Restore original model mode
        if was_train:
            model.train()
        else:
            model.eval()


def geometric_ranking(
    x_candidates: torch.Tensor,
    x_labeled: torch.Tensor,
    R_0: float,
    batch_outputs: int,
) -> List[int]:
    """Rank BALD candidates by geometric representativeness per Eq. (13-14).

    For each candidate x_i*, computes its representativeness as:
      max_j p(y_i| x_i*, theta) = max_j { R_0 / ||x_i* - D_j|| }

    Selects the top `batch_outputs` candidates with highest representativeness.

    # paper-element: eq-ranking-criterion
    # paper-element: eq-batch-output
    # paper-fidelity: R_0 assumes raw [0, 255] pixel scale (MNIST); see select_batch for validation

    Args:
        x_candidates: BALD-ranked candidates of shape (N_candidates, n_features).
        x_labeled: Labeled samples of shape (N_labeled, n_features).
        R_0: Distance threshold for the geometric prior. Assumes raw [0, 255] pixel
            values by default (paper's MNIST scale). Must be calibrated for the data scale.
        batch_outputs: Number of samples to select (b').

    Returns:
        Indices of the top `batch_outputs` most representative candidates.
    """
    # Compute geometric prior probabilities for candidates
    rep_probs = compute_geometric_prior(x_candidates, x_labeled, R_0)  # (N_candidates,)

    # Select top batch_outputs by representativeness
    # Clamp to feasible count
    n_select = min(batch_outputs, len(rep_probs))

    top_indices = torch.topk(rep_probs, n_select).indices.tolist()

    return top_indices


def construct_ellipsoid_core_set(
    x_pool: torch.Tensor,
    core_set_size: int,
    R_0: float,
    eta: float,
    seed: int,
    kmeans_centers: int = 10,
) -> List[int]:
    """Construct initial core-set using ellipsoid geodesic search (Stage 1).

    Implements the max-min optimization with ellipsoid geodesic rescaling:
    1. Initialize k-means centers U for unbiased likelihood baseline (Eq. 7)
    2. Iteratively acquire samples using the geometric prior (Eq. 5) and
        core-set acquisition criterion (Eq. 10)
    3. Apply ellipsoid geodesic rescaling (Eq. 11) to prevent boundary bias

    # paper-element: eq-coreset-acquisition
    # paper-element: eq-ellipsoid-geodesic
    # paper-element: alg-gbald (Stage 1)
    # paper-element: concept-ellipsoid-core-set
    # paper-fidelity: R_0 assumes raw [0, 255] pixel scale (MNIST); see select_batch for validation

    Args:
        x_pool: Full unlabeled pool of shape (N_pool, n_features).
        core_set_size: Number of core-set samples to construct (N_M).
        R_0: Distance threshold for the geometric prior. Assumes raw [0, 255] pixel
            values by default (paper's MNIST scale). Must be calibrated for the data scale.
        eta: Ellipsoid affine factor in (0, 1).
        seed: Random seed for reproducibility.
        kmeans_centers: Number of k-means centers for unbiased likelihood baseline (Eq. 7).

    Returns:
        Indices of the core-set samples in the original pool.
    """
    rng = np.random.default_rng(seed)
    N_pool = len(x_pool)

    # Clamp core_set_size to feasible count
    n_core = min(core_set_size, N_pool)

    if n_core == 0:
        return []

    # Initialize k-means centers for unbiased likelihood baseline (Eq. 7)
    n_kmeans = min(kmeans_centers, n_core)
    kmeans = KMeans(n_clusters=n_kmeans, random_state=seed, n_init=10)
    kmeans.fit(x_pool.cpu().numpy())
    centers = kmeans.cluster_centers_  # (n_kmeans, n_features)

    # Initialize core-set with k-means centers (nearest samples)
    core_set_indices: List[int] = []
    for center in centers:
        center_tensor = torch.from_numpy(center).float().to(x_pool.device)
        distances = torch.norm(x_pool - center_tensor, dim=1)
        nearest_idx = int(torch.argmin(distances).item())
        if nearest_idx not in core_set_indices:
            core_set_indices.append(nearest_idx)

    # Iteratively acquire remaining core-set samples
    x_labeled = x_pool[core_set_indices]  # Current labeled set

    # Convert k-means centers to tensor for L_0 computation
    centers_tensor = torch.from_numpy(centers).float().to(x_pool.device)

    for _ in range(n_core - len(core_set_indices)):
        # Compute geometric prior for remaining unlabeled samples
        remaining_indices = [i for i in range(N_pool) if i not in core_set_indices]
        if not remaining_indices:
            break

        x_remaining = x_pool[remaining_indices]

        # Compute ||L_0 - L||^2 (Eq. 7)
        # L_0: unbiased full likelihood over k-means centers U
        # L: full log-likelihood over current labeled set D_0
        # For geometric prior: E_y[log p(y|x,theta) + H[y|x,D]] ≈ log p(y|x,theta)
        # So L_0 ≈ sum_i log p(yi|xi,theta) over U, L ≈ sum_i log p(yi|xi,theta) over D_0

        # Compute log probabilities for remaining samples relative to k-means centers (L_0)
        rep_probs_U = compute_geometric_prior(x_remaining, centers_tensor, R_0)
        log_probs_U = torch.log(rep_probs_U + 1e-8)
        L_0 = torch.sum(log_probs_U)  # Scalar

        # Compute log probabilities for remaining samples relative to labeled set (L)
        rep_probs = compute_geometric_prior(x_remaining, x_labeled, R_0)
        log_probs = torch.log(rep_probs + 1e-8)
        L = torch.sum(log_probs)  # Scalar

        # ||L_0 - L||^2 term
        likelihood_reg = (L_0 - L) ** 2

        # Eq. (10): x* = argmax min { ||L_0 - L||^2 + log p(y|x,theta) }
        # The min over D_j is already in rep_probs (uses min distance)
        # So we compute: score = ||L_0 - L||^2 + log p(y|x,theta)
        # Note: ||L_0 - L||^2 is a scalar, same for all candidates
        # log p(y|x,theta) = log rep_probs for each candidate
        scores = likelihood_reg + log_probs

        # Select the sample with highest score
        best_local_idx = int(torch.argmax(scores).item())
        best_global_idx = remaining_indices[best_local_idx]

        # Apply ellipsoid geodesic rescaling (Eq. 11)
        # x*_e = x_i + eta * (x* - x_i), then snap to nearest neighbor
        if len(core_set_indices) > 0:
            x_i = x_pool[core_set_indices[-1]]  # Previous acquisition
            x_star = x_pool[best_global_idx]    # Selected acquisition
            x_e = x_i + eta * (x_star - x_i)    # Rescaled position

            # Snap to nearest neighbor in remaining pool
            distances = torch.norm(x_remaining - x_e, dim=1)
            snapped_local_idx = int(torch.argmin(distances).item())
            best_global_idx = remaining_indices[snapped_local_idx]

        # Add to core-set
        core_set_indices.append(best_global_idx)
        x_labeled = x_pool[core_set_indices]

    return core_set_indices


def select_batch(
    model: nn.Module,
    x_unlabeled: torch.Tensor,
    x_labeled: torch.Tensor,
    batch_size: int,
    seed: int,
    mc_samples: int = 100,
    core_set_size: int = 100,
    R_0: float = 2000.0,
    eta: float = 0.9,
    batch_returns: int = 300,
) -> List[int]:
    """GBALD acquisition function: two-stage ellipsoid core-set + geometric BALD ranking.

    Stage 1 (core-set construction): Constructs an initial core-set on an ellipsoid
    geometry using the geometric prior (Eq. 5) and ellipsoid geodesic rescaling
    (Eq. 11). This provides representative samples that cover the input distribution.

    Stage 2 (model uncertainty estimation): Uses MC dropout to score unlabeled
    candidates by BALD mutual information (Eq. 12), then ranks those candidates
    by their geometric representativeness (Eq. 13-14) to select the top `batch_size`
    most informative and representative samples.

    # paper-element: alg-gbald
    # paper-element: concept-gbald-framework
    # paper-fidelity: R_0=2000.0 assumes raw [0, 255] pixel scale (MNIST per Section 7.9).
    #   For normalized data [0,1], use R_0≈7.8. For normalized data [-1,1], use R_0≈15.6.
    #   The caller is responsible for calibrating R_0 to the data preprocessing mode.

    Args:
        model: Trained model with dropout layers for MC dropout.
        x_unlabeled: Unlabeled samples of shape (N_unlabeled, n_features).
        x_labeled: Labeled samples of shape (N_labeled, n_features).
        batch_size: Number of samples to select (b').
        seed: Random seed for reproducibility.
        mc_samples: Number of MC dropout forward passes for BALD scoring.
        core_set_size: Stage 1 core-set size N_M (used for initial acquisitions).
        R_0: Distance threshold for the geometric prior. Default 2000.0 assumes
            raw [0, 255] pixel values (MNIST scale per paper Section 7.9). For
            normalized data [0, 1], use R_0≈7.8. For normalized data [-1, 1], use R_0≈15.6.
        eta: Ellipsoid affine factor in (0, 1).
        batch_returns: Number of BALD-ranked candidates to consider (b).

    Returns:
        Indices of the selected samples in x_unlabeled.
    """
    rng = np.random.default_rng(seed)

    # Combine labeled and unlabeled for core-set construction (Stage 1)
    # The full pool is the union of labeled and unlabeled
    labeled_len = len(x_labeled)
    x_pool = torch.cat([x_labeled, x_unlabeled], dim=0)

    # Stage 1: Construct ellipsoid core-set
    # This augments the initial labeled set with representative samples
    core_set_indices = construct_ellipsoid_core_set(
        x_pool, core_set_size, R_0, eta, seed, kmeans_centers=10
    )

    # Update x_labeled with core-set (Stage 1 output)
    # The core-set indices are in the combined pool; map back to unlabeled
    # Offset indices from x_pool space to x_unlabeled space
    new_labeled_indices = [
        idx - labeled_len for idx in core_set_indices if idx >= labeled_len
    ]
    x_labeled = torch.cat(
        [x_labeled, x_unlabeled[new_labeled_indices]], dim=0
    )

    # Update x_unlabeled to exclude core-set acquisitions
    remaining_unlabeled_indices = [
        i for i in range(len(x_unlabeled)) if i not in new_labeled_indices
    ]
    x_unlabeled = x_unlabeled[remaining_unlabeled_indices]

    # Stage 2: Model uncertainty estimation with BALD scoring
    bald_scores = compute_bald_scores(model, x_unlabeled, mc_samples)

    # Select top batch_returns candidates by BALD score
    n_candidates = min(batch_returns, len(bald_scores))
    top_candidate_indices = torch.topk(bald_scores, n_candidates).indices

    x_candidates = x_unlabeled[top_candidate_indices]

    # Rank candidates by geometric representativeness and select top batch_size
    # Clamp batch_size to feasible count
    n_select = min(batch_size, len(x_candidates))

    selected_local_indices = geometric_ranking(
        x_candidates, x_labeled, R_0, n_select
    )

    # Map back to original x_unlabeled indices
    selected_unlabeled_indices = [
        remaining_unlabeled_indices[top_candidate_indices[i].item()]
        for i in selected_local_indices
    ]

    return selected_unlabeled_indices
