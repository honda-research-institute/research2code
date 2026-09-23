"""
GBALD (Geometric Bayesian Active Learning by Disagreements) implementation.

Paper: "Bayesian Active Learning by Disagreements: A Geometric Perspective"
       by Xiaofeng Cao and Ivor W. Tsang (2022)

This module implements GBALD's two-stage active learning algorithm:

Stage 1 - Core-set construction (construct_core_set):
    Builds an initial labeled set via ellipsoid geodesic search (Eq. 10-11),
    avoiding boundary sampling failures of spherical methods. Uses geometric
    probability model (Eq. 5) with R_0 threshold and eta rescaling factor.

Stage 2 - Model uncertainty estimation (select_batch):
    Scores unlabeled candidates via MC-dropout BALD (Eq. 12), then ranks
    top candidates by geometric representativeness (Eq. 13-14) to select
    the most diverse subset.

Public functions:
    - construct_core_set: Stage 1 ellipsoid-based core-set construction
    - compute_bald_scores: MC-dropout BALD mutual information scoring
    - geometric_ranking: Representativeness ranking via inverse distance
    - select_batch: Stage 2 pluggable acquisition function (BALD + ranking)

# paper-element: concept-gbald-framework
# paper-element: alg-gbald
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch import nn


def construct_core_set(
    x_pool: torch.Tensor,
    core_set_size: int,
    R_0: float = 2000.0,
    eta: float = 0.9,
    seed: int = 42,
) -> List[int]:
    """Construct initial core-set via ellipsoid geodesic search (Stage 1).

    This implements GBALD's Stage 1: ellipsoid-based geometric core-set
    construction that bootstraps the initial labeled set before any model
    training. The algorithm uses max-min optimization (Eq. 10) combined
    with ellipsoid geodesic rescaling (Eq. 11) to avoid boundary sampling.

    Args:
        x_pool: Unlabeled data pool of shape (N, n_features).
        core_set_size: Target number of core-set samples (N_M). Will be
            clamped to len(x_pool) if larger.
        R_0: Distance threshold for geometric probability model (Eq. 5).
            Paper value 2000.0 is calibrated for raw [0, 255] pixel data.
        eta: Ellipsoid geodesic scaling factor (0 < eta < 1). Paper uses 0.9.
            Controls how far updates move toward distribution boundaries.
        seed: Random seed for k-means initialization.

    Returns:
        List of indices into x_pool representing the constructed core-set.

    # paper-element: eq-geometric-prior
    # paper-element: eq-combined-acquisition
    # paper-element: eq-ellipsoid-geodesic
    # paper-element: hyp-geometric-params
    # essential: Ellipsoid geodesic core-set construction (must_replicate)
    """
    rng = np.random.default_rng(seed)
    n_samples = x_pool.shape[0]

    # Clamp core_set_size to feasible pool size
    # paper-fidelity: At paper scale (pool_size >> core_set_size), this is a no-op.
    core_set_size = min(core_set_size, n_samples)

    if core_set_size == 0:
        return []

    # Convert to numpy for distance computations
    x_pool_np = x_pool.cpu().numpy() if x_pool.is_cuda else x_pool.numpy()

    # Initialize core-set with k-means centers mapped to nearest real samples
    # This implements Algorithm 1 Line 5: "Perform k-means to initialize U"
    # paper-element: alg-gbald
    from sklearn.cluster import KMeans

    n_kmeans = min(core_set_size // max(1, core_set_size // 10), n_samples)
    kmeans = KMeans(n_clusters=n_kmeans, random_state=seed, n_init=10)
    kmeans.fit(x_pool_np)

    # Map k-means centers to nearest real samples
    # These form U, the k-means centers used for L_0 computation (Eq. 7)
    core_set_indices: List[int] = []
    for center in kmeans.cluster_centers_:
        distances = np.linalg.norm(x_pool_np - center, axis=1)
        nearest_idx = int(np.argmin(distances))
        if nearest_idx not in core_set_indices:
            core_set_indices.append(nearest_idx)

    # Compute L_0 once: the unbiased full likelihood over k-means centers U (Eq. 7)
    # L_0 = |sum_i E_{y_i}[log p(y_i|x_i,theta) + H[y_i|x_i,U]]|
    # Using geometric probability model (Eq. 5): p(y|x,theta) based on distance to centers
    # For simplicity, we approximate the entropy term and use log p as the main component
    # paper-element: eq-minimize-L
    def compute_likelihood(x_data: np.ndarray, centers: np.ndarray, R_0: float) -> float:
        """Compute likelihood L = sum_i log p(y_i|x_i,theta) using geometric probability."""
        if len(centers) == 0:
            return 0.0
        total = 0.0
        for x in x_data:
            dists = np.linalg.norm(centers - x, axis=1)
            min_dist = np.min(dists)
            if min_dist <= R_0:
                prob = 1.0
            else:
                prob = R_0 / min_dist
            total += np.log(prob + 1e-10)
        return abs(total)

    # Compute L_0 over the full pool using k-means centers as reference
    L_0 = compute_likelihood(x_pool_np, kmeans.cluster_centers_, R_0)

    # Iteratively acquire core-set samples via Eq. 10 + Eq. 11
    # paper-element: eq-combined-acquisition
    # paper-element: eq-ellipsoid-geodesic
    available_indices = set(range(n_samples)) - set(core_set_indices)

    for _ in range(core_set_size - len(core_set_indices)):
        if not available_indices:
            break

        x_available = x_pool_np[list(available_indices)]
        x_labeled = x_pool_np[core_set_indices]

        # Compute combined acquisition score (Eq. 10):
        # x* = argmax_{x_j in D_u} min_{D_j in D_0} { ||L_0 - L||^2 + log p(y_j|x_j,theta) }
        # paper-element: eq-geometric-prior
        # paper-element: eq-combined-acquisition

        # Compute L over current labeled set (grows with each acquisition)
        L = compute_likelihood(x_labeled, x_labeled, R_0)

        # Compute ||L_0 - L||^2 regularization term (Eq. 7)
        regularization = (L_0 - L) ** 2

        # Compute scores for each candidate
        scores = np.zeros(len(x_available))
        for i, x_cand in enumerate(x_available):
            # Compute min distance to any labeled center
            dists_to_labeled = np.linalg.norm(x_labeled - x_cand, axis=1)
            min_dist = np.min(dists_to_labeled)

            # Geometric probability (Eq. 5)
            if min_dist <= R_0:
                prob = 1.0
            else:
                prob = R_0 / min_dist

            # Combined score: ||L_0 - L||^2 + log p(y|x,theta) (Eq. 10)
            # paper-element: eq-combined-acquisition
            scores[i] = regularization + np.log(prob + 1e-10)

        # Select candidate with highest score
        best_local_idx = int(np.argmax(scores))
        best_global_idx = list(available_indices)[best_local_idx]
        x_star = x_pool_np[best_global_idx]

        # Apply ellipsoid geodesic rescaling (Eq. 11)
        # x_e* = x_i + eta * (x* - x_i), then snap to nearest neighbor
        if len(core_set_indices) > 0:
            x_prev = x_pool_np[core_set_indices[-1]]
            x_ellipsoid = x_prev + eta * (x_star - x_prev)

            # Snap to nearest neighbor in available pool
            dists_to_available = np.linalg.norm(x_available - x_ellipsoid, axis=1)
            snapped_local_idx = int(np.argmin(dists_to_available))
            snapped_global_idx = list(available_indices)[snapped_local_idx]
        else:
            snapped_global_idx = best_global_idx

        # Add to core-set
        core_set_indices.append(snapped_global_idx)
        available_indices.remove(snapped_global_idx)

    return core_set_indices


def compute_bald_scores(
    model: nn.Module,
    x_unlabeled: torch.Tensor,
    mc_samples: int = 2000,
) -> np.ndarray:
    """Compute BALD scores via MC dropout (Eq. 12).

    This implements the BALD mutual information scoring for GBALD's Stage 2.
    For each unlabeled candidate, computes:
        BALD(x) = H[p(y|x)] - E_{theta}[H[p(y|x,theta)]]
    where the expectation is approximated via T stochastic forward passes
    with dropout active.

    Args:
        model: Trained model with dropout layers. MUST be in train() mode
            for MC dropout to work. The function does NOT change model mode.
        x_unlabeled: Unlabeled data of shape (N, n_features).
        mc_samples: Number of MC dropout forward passes per sample (T).
            Paper uses 2000; demo runs may use fewer (20-50).

    Returns:
        BALD scores of shape (N,), one per unlabeled sample. Higher = more
        informative (higher mutual information between y and theta).

    # paper-element: eq-bald-batch
    # paper-element: concept-mc-dropout
    # essential: MC dropout layers active at inference
    Note: This function assumes model is in train() mode. If called with
    model.eval(), dropout is disabled and all T samples are identical,
    causing BALD scores to collapse to zero.
    """
    n_samples = x_unlabeled.shape[0]
    device = x_unlabeled.device

    # Collect predictions from all MC dropout samples
    # Shape: (mc_samples, n_samples, n_classes)
    all_logits = torch.zeros(mc_samples, n_samples, model.n_classes, device=device)

    for t in range(mc_samples):
        all_logits[t] = model(x_unlabeled)

    # Convert to probabilities: softmax over last axis
    # Shape: (mc_samples, n_samples, n_classes)
    all_probs = torch.softmax(all_logits, dim=-1)

    # Compute BALD = H[p(y|x)] - E_theta[H[p(y|x,theta)]]
    # paper-element: eq-bald-single

    # 1. Predictive distribution p(y|x) = mean over MC samples
    # Shape: (n_samples, n_classes)
    predictive_probs = all_probs.mean(dim=0)

    # 2. Predictive entropy H[p(y|x)]
    # H[p] = -sum_y p(y) log p(y)
    # Shape: (n_samples,)
    predictive_entropy = -(
        predictive_probs * torch.log(predictive_probs + 1e-10)
    ).sum(dim=-1)

    # 3. Expected entropy E_theta[H[p(y|x,theta)]]
    # Compute entropy for each MC sample, then average
    # Shape: (mc_samples, n_samples)
    sample_entropies = -(
        all_probs * torch.log(all_probs + 1e-10)
    ).sum(dim=-1)
    expected_entropy = sample_entropies.mean(dim=0)

    # 4. BALD = predictive_entropy - expected_entropy
    # Shape: (n_samples,)
    bald_scores = predictive_entropy - expected_entropy

    return bald_scores.detach().cpu().numpy()


def geometric_ranking(
    x_candidates: torch.Tensor,
    x_labeled: torch.Tensor,
    R_0: float = 2000.0,
    top_k: int = 100,
) -> List[int]:
    """Rank candidates by geometric representativeness (Eq. 13-14).

    From the BALD-scored candidates, select the most geometrically
    representative subset by maximizing the geometric probability
    R_0 / ||x_candidate - D_j|| to the nearest labeled center.
    This selects the most representative samples from high-density regions
    of the labeled set, avoiding boundary points.

    Args:
        x_candidates: BALD-ranked candidates of shape (b, n_features).
        x_labeled: Current labeled set of shape (N_labeled, n_features).
        R_0: Distance threshold from geometric probability model (Eq. 5).
            Same value used in core-set construction.
        top_k: Number of samples to return (b'). Clamped to len(x_candidates).

    Returns:
        Indices (into x_candidates) of top_k most representative samples.

    # paper-element: eq-representation-ranking
    # paper-element: eq-batch-acquisitions
    # paper-element: eq-geometric-prior
    # essential: Geometric representativeness ranking (must_replicate)
    """
    n_candidates = x_candidates.shape[0]

    # Clamp top_k to feasible candidate count
    # paper-fidelity: At paper scale (b >> b'), this is a no-op.
    top_k = min(top_k, n_candidates)

    if top_k == 0:
        return []

    x_candidates_np = x_candidates.cpu().numpy()
    x_labeled_np = x_labeled.cpu().numpy()

    # Compute representativeness score for each candidate
    # Eq. 13: rep_score = max_j (R_0 / ||x_candidate - D_j||)
    # paper-element: eq-representation-ranking
    rep_scores = np.zeros(n_candidates)

    for i, x_cand in enumerate(x_candidates_np):
        # Compute distances to all labeled centers
        dists = np.linalg.norm(x_labeled_np - x_cand, axis=1)
        min_dist = np.min(dists)

        # Geometric probability (Eq. 5)
        if min_dist <= R_0:
            rep_scores[i] = 1.0
        else:
            rep_scores[i] = R_0 / min_dist

    # Select top_k by representativeness score
    top_indices = np.argsort(rep_scores)[::-1][:top_k].tolist()

    return top_indices


def select_batch(
    model: nn.Module,
    x_unlabeled: torch.Tensor,
    x_labeled: torch.Tensor,
    batch_size: int,
    seed: int,
    R_0: float = 2000.0,
    eta: float = 0.9,
    mc_samples: int = 2000,
    batch_returns: int = 300,
    batch_outputs: int = 100,
) -> List[int]:
    """GBALD Stage 2: BALD scoring + geometric representativeness ranking.

    This is the pluggable acquisition function for GBALD's iterative
    active learning loop (Stage 2). The algorithm:
    1. Scores all unlabeled candidates via MC-dropout BALD (Eq. 12)
    2. Takes top `batch_returns` candidates by BALD score
    3. Ranks those candidates by geometric representativeness (Eq. 13-14)
    4. Returns top `batch_outputs` indices

    Args:
        model: Trained model with dropout layers. The function will set
            it to train() mode for MC dropout, then restore eval().
        x_unlabeled: Unlabeled data pool of shape (N_pool, n_features).
        x_labeled: Current labeled set of shape (N_labeled, n_features).
            Used for geometric representativeness ranking (Eq. 13).
        batch_size: Target number of samples to return. Will be clamped
            to len(x_unlabeled) if larger.
        seed: Random seed for any stochastic operations.
        R_0: Distance threshold for geometric probability (Eq. 5). Paper
            value 2000.0 for raw pixel data.
        eta: Ellipsoid geodesic factor (used in Stage 1, not Stage 2).
            Included for API consistency; not used in this function.
        mc_samples: Number of MC dropout forward passes. Paper uses 2000.
            Demo runs may use fewer (20-50).
        batch_returns: Number of BALD-ranked candidates to consider (b).
            Paper uses 300 for batch acquisition.
        batch_outputs: Number of samples to return (b'). Paper uses 100.

    Returns:
        List of indices into x_unlabeled representing the selected batch.

    # paper-element: alg-gbald
    # paper-element: eq-bald-batch
    # paper-element: eq-representation-ranking
    # paper-element: eq-batch-acquisitions
    # essential: MC dropout layers active at inference
    # essential: Geometric representativeness ranking (must_replicate)
    """
    rng = np.random.default_rng(seed)
    n_pool = x_unlabeled.shape[0]

    # Clamp batch_size and batch_returns to feasible pool size
    # paper-fidelity: At paper scale (pool_size >> batch parameters), clamping is no-op.
    batch_size = min(batch_size, n_pool)
    batch_returns = min(batch_returns, n_pool)
    batch_outputs = min(batch_outputs, batch_returns)

    if batch_size == 0:
        return []

    # Ensure model is in train() mode for MC dropout
    # paper-element: concept-mc-dropout
    # essential: MC dropout layers active at inference
    was_train = model.training
    model.train()

    try:
        # Step 1: Compute BALD scores for all unlabeled candidates
        # paper-element: eq-bald-batch
        bald_scores = compute_bald_scores(model, x_unlabeled, mc_samples)

        # Step 2: Select top batch_returns by BALD score
        top_b_indices = np.argsort(bald_scores)[::-1][:batch_returns]
        x_candidates = x_unlabeled[top_b_indices]

        # Step 3: Rank by geometric representativeness (Eq. 13-14)
        # paper-element: eq-representation-ranking
        # paper-element: eq-batch-acquisitions
        top_rep_indices = geometric_ranking(
            x_candidates, x_labeled, R_0, top_k=batch_outputs
        )

        # Step 4: Map back to original pool indices
        selected_pool_indices = [int(top_b_indices[i]) for i in top_rep_indices]

    finally:
        # Restore original model mode
        if not was_train:
            model.eval()

    return selected_pool_indices
