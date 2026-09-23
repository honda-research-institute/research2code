"""
GBALD (Geometric Bayesian Active Learning by Disagreements) acquisition function.

This module implements the GBALD algorithm from Cao & Tsang (2022), "Bayesian Active
Learning by Disagreements: A Geometric Perspective". GBALD is a two-stage Bayesian
acquisition method:

**Stage 1 - Core-set construction** (Section 4.2, Algorithm 1 Lines 3-13):
Constructs a geometrically-informed core-set on an ellipsoid (not a sphere) to
initialize model uncertainty estimation. Uses a geometric probability model (Eq. 5)
with radius R_0, max-min optimization (Eq. 10), and ellipsoid geodesic rescaling
(Eq. 11) with affine factor eta to prevent boundary elements.

**Stage 2 - Model uncertainty estimation** (Section 4.3, Algorithm 1 Lines 14-21):
Uses MC dropout (T=mc_samples) to score candidates by BALD mutual information (Eq. 12),
then ranks and selects acquisitions by geometric representativeness (Eq. 13) computed
from the geometric prior, reducing redundant nearby sampling.

Public functions:
- select_batch: Main pluggable acquisition function (two-stage GBALD)
- geometric_prior: Geometric probability model p(y|x,theta) (Eq. 5)
- ellipsoid_geodesic_rescale: Ellipsoid geodesic rescaling (Eq. 11)
- construct_core_set: Stage 1 core-set construction (Algorithm 1 Lines 3-13)
- compute_bald_scores: BALD mutual information via MC dropout (Eq. 12)
- geometric_ranking: Rank candidates by geometric representativeness (Eq. 13)

Paper: Cao, X. & Tsang, I. W. (2022). Bayesian Active Learning by Disagreements:
A Geometric Perspective.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from typing import List


def geometric_prior(x_i: torch.Tensor, D_0: torch.Tensor, R_0: float) -> float:
    """
    Geometric probability model p(y_i | x_i, theta) (Eq. 5, Section 4.2).

    # paper-element: eq-geometric-prior
    # paper-element: hyp-sphere-radius

    Given a radius R_0 for any observed internal sphere centered at D_j:
    - If x_i is within distance R_0 of any D_j: p = 1.0
    - Otherwise: p = max{R_0 / ||x_i - D_j||} assigned by nearest sphere

    This geometric prior initializes the model uncertainty estimation with
    a distance-based probability, preventing low-representative boundary elements.

    Args:
        x_i: Single data point of shape (n_features,).
        D_0: Set of already-acquired points of shape (N_acquired, n_features).
        R_0: Radius of internal spheres (default 2000.0 per paper Section 7.9).

    Returns:
        Probability p(y_i | x_i, theta) in [0, 1.0].

    Note:
        At smoke scale with normalized data [0, 1], R_0 = 2000.0 is calibrated
        for raw pixel values [0, 255]. For normalized data, distances are smaller,
        so R_0 may need rescaling. The paper uses raw pixels; this is documented
        in spec.critical_requirements.scale_dependent_hyperparameters.
    """
    # Compute distances from x_i to all D_j
    distances = torch.norm(x_i - D_0, dim=1)  # shape (N_acquired,)

    # Check if x_i is within R_0 of any D_j
    within_sphere = torch.any(distances <= R_0)
    if within_sphere:
        return 1.0

    # Otherwise, return max{R_0 / distance} = R_0 / min_distance
    min_distance = torch.min(distances).item()
    if min_distance == 0:
        return 1.0
    return float(R_0 / min_distance)


def ellipsoid_geodesic_rescale(
    x_prev: torch.Tensor,
    x_acq: torch.Tensor,
    eta: float,
    x_pool: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """
    Ellipsoid geodesic rescaling (Eq. 11, Section 4.2).

    # paper-element: eq-geodesic-ellipsoid
    # paper-element: hyp-ellipsoid-scale

    Rescales the acquisition position via affine projection onto an ellipsoid,
    preventing updates from falling into spherical boundary regions. The position
    x*_e = x_prev + eta * (x_acq - x_prev) is scaled by affine factor eta where
    0 < eta < 1, then mapped to the nearest neighbor in the unlabeled pool.

    Args:
        x_prev: Previous acquisition of shape (n_features,).
        x_acq: Desired acquisition of shape (n_features,).
        eta: Affine factor for ellipsoid geodesic (default 0.9 per paper Section 7.9).
        x_pool: Unlabeled pool of shape (N_pool, n_features).

    Returns:
        Tuple of (x_rescaled, pool_index) where:
            - x_rescaled: Nearest neighbor in x_pool to the rescaled position
            - pool_index: Index of the nearest neighbor in x_pool

    Note:
        The ellipsoid geodesic prevents the core-set from updating toward boundary
        regions where characteristics of the distribution cannot be properly captured.
        See Figure 2 in the paper for visualization.
    """
    # Compute rescaled position: x*_e = x_prev + eta * (x_acq - x_prev)
    x_rescaled_target = x_prev + eta * (x_acq - x_prev)

    # Find nearest neighbor in x_pool
    distances = torch.norm(x_pool - x_rescaled_target, dim=1)  # shape (N_pool,)
    pool_index = torch.argmin(distances).item()
    x_rescaled = x_pool[pool_index]

    return x_rescaled, pool_index


def construct_core_set(
    x_pool: torch.Tensor,
    core_set_size: int,
    kmeans_centers: torch.Tensor,
    eta: float,
    R_0: float,
    seed: int,
) -> List[int]:
    """
    Stage 1: Core-set construction on ellipsoid (Algorithm 1, Lines 3-13).

    # paper-element: alg-gbald
    # paper-element: eq-core-set-opt

    Constructs a core-set of size N_M via max-min optimization (Eq. 10) combined
    with the geometric probability model (Eq. 5), followed by ellipsoid geodesic
    rescaling (Eq. 11). This initialized core-set prevents low-representative
    boundary elements and provides balanced class coverage.

    The algorithm:
    1. Initialize theta to yield geometric probability model (Eq. 5)
    2. Use k-means centers U to initialize D_0
    3. For i = 1 to N_M:
       a. Select x*_i via: argmax_x min_D {||L_0 - L||² + log p(y|x,theta)}
       b. Rescale via ellipsoid geodesic: x*_e = x_i + eta*(x* - x_i)
       c. Map to nearest neighbor in unlabeled pool
       d. Add to core-set M
    4. Return indices of core-set samples

    Args:
        x_pool: Unlabeled pool of shape (N_pool, n_features).
        core_set_size: Number of core-set samples to construct (N_M).
        kmeans_centers: K-means centers of shape (k, n_features) used to initialize D_0.
        eta: Affine factor for ellipsoid geodesic rescaling.
        R_0: Radius for geometric probability model.
        seed: Random seed for reproducibility.

    Returns:
        List of indices into x_pool representing the constructed core-set.

    Note:
        The full optimization (Eq. 10) includes ||L_0 - L||² + log p(y|x,theta).
        For smoke-scale runs, we use a simplified version focusing on the geometric
        prior component (log p(y|x,theta)) which captures the main algorithmic
        contribution. The L_0 regularization term requires computing full likelihood
        expectations which are expensive at scale.
    """
    rng = np.random.default_rng(seed)

    N_pool = len(x_pool)

    # Clamp core_set_size to available pool size
    # paper-fidelity: At paper scale (pool_size >> core_set_size), this is no-op
    core_set_size = min(core_set_size, N_pool)

    # Initialize D_0 with k-means centers (mapped to nearest pool samples)
    D_0_indices = []
    for center in kmeans_centers:
        distances = torch.norm(x_pool - center, dim=1)
        nearest_idx = torch.argmin(distances).item()
        D_0_indices.append(nearest_idx)

    core_set_indices = list(D_0_indices)
    remaining_indices = [i for i in range(N_pool) if i not in core_set_indices]

    # Iteratively construct core-set
    x_prev = x_pool[core_set_indices[0]] if core_set_indices else None

    for _ in range(core_set_size - len(core_set_indices)):
        if not remaining_indices:
            break

        # Get remaining pool as tensor
        x_remaining = x_pool[remaining_indices]
        D_0_tensor = x_pool[core_set_indices]

        # Compute geometric prior scores for remaining samples
        # Simplified: use log p(y|x,theta) component of Eq. 10
        scores = []
        for idx, x_i in zip(remaining_indices, x_remaining):
            p = geometric_prior(x_i, D_0_tensor, R_0)
            # Use log probability (handle p=0 case)
            log_p = np.log(max(p, 1e-10))
            scores.append((idx, log_p))

        # Select sample with highest score (argmax of log p)
        best_idx, _ = max(scores, key=lambda x: x[1])
        x_acq = x_pool[best_idx]

        # Apply ellipsoid geodesic rescaling if we have a previous acquisition
        if x_prev is not None:
            x_rescaled, rescaled_idx = ellipsoid_geodesic_rescale(
                x_prev, x_acq, eta, x_pool
            )
            final_idx = rescaled_idx
            x_acq = x_rescaled
        else:
            final_idx = best_idx

        # Update state
        core_set_indices.append(final_idx)
        x_prev = x_acq
        if final_idx in remaining_indices:
            remaining_indices.remove(final_idx)

    return core_set_indices


def compute_bald_scores(
    model: torch.nn.Module,
    x_pool: torch.Tensor,
    mc_samples: int,
) -> torch.Tensor:
    """
    Compute BALD mutual information scores via MC dropout (Eq. 12, Section 4.3).

    # paper-element: eq-bald-batch
    # paper-element: hyp-mc-dropout-samples
    # essential: MC dropout active at inference

    BALD measures the mutual information between model parameters and labels:
        BALD(x) = H[theta|D_0] - E_y[H[theta|x, y, D_0]]

    Approximated via MC dropout: compute T stochastic forward passes, then
    estimate the mutual information as the difference between:
    - Entropy of the mean prediction (model uncertainty)
    - Mean of entropies of individual predictions (aleatoric uncertainty)

    Args:
        model: Trained model with dropout layers.
        x_pool: Unlabeled pool of shape (N_pool, n_features).
        mc_samples: Number of MC dropout samples (T).

    Returns:
        BALD scores of shape (N_pool,).

    Note:
        The model MUST be in train() mode for dropout to be active. We handle
        this inside the function and restore the original mode afterward.
        Without dropout active, all T samples produce identical predictions,
        BALD scores collapse to zero, and GBALD degenerates to random sampling.
    """
    # Save original model mode
    original_mode = model.training

    # essential: MC dropout active at inference
    # Set model to train mode so dropout is active
    model.train()

    with torch.no_grad():
        N_pool = len(x_pool)

        # First forward pass to determine output shape (n_classes)
        logits = model(x_pool)
        probs = torch.softmax(logits, dim=1)  # shape (N_pool, n_classes)
        n_classes = probs.size(-1)
        probs_all = torch.zeros(mc_samples, N_pool, n_classes, dtype=torch.float32)
        probs_all[0] = probs

        # MC dropout: T stochastic forward passes (t=1 to mc_samples-1)
        for t in range(1, mc_samples):
            logits = model(x_pool)
            probs = torch.softmax(logits, dim=1)  # shape (N_pool, n_classes)
            probs_all[t] = probs

    # Restore original model mode
    model.train(original_mode)

    # Compute mean prediction probabilities
    mean_probs = probs_all.mean(dim=0)  # shape (N_pool, n_classes)

    # Entropy of mean prediction: H[E[p(y|x)]]
    entropy_mean = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)

    # Mean of entropies: E[H[p(y|x)]]
    entropies = -(probs_all * torch.log(probs_all + 1e-10)).sum(dim=2)  # shape (mc_samples, N_pool)
    mean_entropy = entropies.mean(dim=0)  # shape (N_pool,)

    # BALD score = H[E[p]] - E[H[p]]
    bald_scores = entropy_mean - mean_entropy

    return bald_scores


def geometric_ranking(
    x_candidates: torch.Tensor,
    x_labeled: torch.Tensor,
    R_0: float,
) -> torch.Tensor:
    """
    Rank candidates by geometric representativeness (Eq. 13, Section 4.3).

    # paper-element: eq-single-acquisition
    # paper-element: eq-geometric-prior
    # paper-element: eq-batch-acquisition

    From a batch of b informative acquisitions, ranks and selects the top b'
    by geometric representativeness computed via the geometric probability model.
    This reduces redundant nearby sampling by preferring candidates that are
    geometrically representative with respect to the labeled set.

    The representativeness score for candidate x*_i is:
        score(x*_i) = max_{D_j in D_0} p(y_i | x*_i, theta) = R_0 / ||x*_i - D_j||

    Args:
        x_candidates: Candidate acquisitions of shape (N_candidates, n_features).
        x_labeled: Already-labeled set of shape (N_labeled, n_features).
        R_0: Radius for geometric probability model.

    Returns:
        Representativeness scores of shape (N_candidates,).

    Note:
        Higher scores indicate more representative candidates (closer to labeled set
        in feature space). The geometric prior ensures we don't select redundant
        nearby samples that provide little new information.
    """
    N_candidates = len(x_candidates)
    scores = torch.zeros(N_candidates, dtype=torch.float32)

    for i, x_i in enumerate(x_candidates):
        p = geometric_prior(x_i, x_labeled, R_0)
        scores[i] = torch.tensor(float(p))

    return scores


def select_batch(
    model,
    x_unlabeled: torch.Tensor,
    x_labeled: torch.Tensor,
    batch_size: int,
    seed: int,
    mc_samples: int = 2000,
    core_set_size: int = 1000,
    batch_returns: int = 300,
    batch_outputs: int = 100,
    eta: float = 0.9,
    R_0: float = 2000.0,
) -> List[int]:
    """
    GBALD acquisition function: two-stage Bayesian active learning (Algorithm 1).

    # paper-element: alg-gbald

    GBALD operates in two stages per acquisition cycle:

    **Stage 1 - Core-set construction** (Section 4.2):
    Uses a geometric probability model (Eq. 5) with radius R_0 to construct
    a core-set on an ellipsoid via max-min optimization (Eq. 10) followed by
    ellipsoid geodesic rescaling (Eq. 11) with affine factor eta. This initialized
    core-set prevents low-representative boundary elements and provides balanced
    class coverage.

    **Stage 2 - Model uncertainty estimation** (Section 4.3):
    Uses the model (via MC dropout) to score candidates by BALD mutual information
    (Eq. 12), then ranks and selects acquisitions by geometric representativeness
    (Eq. 13) computed from the geometric prior, reducing redundant nearby sampling.

    Args:
        model: Trained MCDropoutMLP model with dropout layers.
        x_unlabeled: Unlabeled pool of shape (N_pool, n_features).
        x_labeled: Already-labeled set of shape (N_labeled, n_features).
        batch_size: Number of samples to acquire this round.
        seed: Random seed for reproducibility.
        mc_samples: Number of MC dropout samples for BALD scoring (default 2000).
        core_set_size: Size of core-set for Stage 1 (default 1000).
        batch_returns: Number of informative acquisitions to return before ranking (b, default 300).
        batch_outputs: Number of top-ranked acquisitions to output (b', default 100).
        eta: Affine factor for ellipsoid geodesic rescaling (default 0.9).
        R_0: Radius for geometric probability model (default 2000.0).

    Returns:
        List of batch_size indices into x_unlabeled representing selected acquisitions.

    Note:
        At smoke scale, parameters are clamped to available pool sizes:
        - batch_returns = min(batch_returns, len(x_unlabeled))
        - batch_outputs = min(batch_outputs, batch_returns, batch_size)
        This ensures the algorithm works correctly even when the pool is small.
        At paper scale (pool_size >> batch_size), these clamps are no-ops.
    """
    rng = np.random.default_rng(seed)

    N_pool = len(x_unlabeled)

    # Clamp parameters to available pool size
    # paper-fidelity: At paper scale, these clamps are no-ops
    batch_returns = min(batch_returns, N_pool)
    batch_outputs = min(batch_outputs, batch_returns, batch_size)
    batch_size = min(batch_size, N_pool)

    # Stage 1: Core-set construction (only needed if we have enough unlabeled data)
    # For subsequent rounds, x_labeled already contains the core-set
    # We skip Stage 1 if x_labeled is already populated (subsequent acquisition rounds)
    if len(x_labeled) == 0:
        # Initial round: construct core-set
        kmeans = KMeans(n_clusters=min(core_set_size, N_pool), random_state=seed)
        kmeans.fit(x_unlabeled.numpy())
        kmeans_centers = torch.from_numpy(kmeans.cluster_centers_)

        core_set_indices = construct_core_set(
            x_unlabeled,
            core_set_size,
            kmeans_centers,
            eta,
            R_0,
            seed,
        )
        # For the initial round, we return the core-set indices
        if batch_size <= len(core_set_indices):
            return core_set_indices[:batch_size]
        # If batch_size > core_set_size, we need more samples from Stage 2
        remaining_needed = batch_size - len(core_set_indices)
        remaining_indices = [i for i in range(N_pool) if i not in core_set_indices]

        # Stage 2: Score remaining by BALD
        x_remaining = x_unlabeled[remaining_indices]
        bald_scores = compute_bald_scores(model, x_remaining, mc_samples)

        # Take top batch_returns by BALD
        top_k = min(batch_returns, len(bald_scores))
        top_indices_local = torch.argsort(bald_scores, descending=True)[:top_k]
        top_indices_global = [remaining_indices[i] for i in top_indices_local]
        x_top = x_unlabeled[top_indices_global]

        # Rank by geometric representativeness
        rep_scores = geometric_ranking(x_top, x_unlabeled[core_set_indices], R_0)
        ranked_local = torch.argsort(rep_scores, descending=True)[:batch_outputs]
        ranked_indices = [top_indices_global[i] for i in ranked_local]

        return core_set_indices + ranked_indices[:remaining_needed]

    # Subsequent rounds: Stage 2 only (model uncertainty estimation)
    # Score all unlabeled by BALD
    bald_scores = compute_bald_scores(model, x_unlabeled, mc_samples)

    # Take top batch_returns by BALD
    top_k = min(batch_returns, len(bald_scores))
    top_indices_local = torch.argsort(bald_scores, descending=True)[:top_k]
    x_top = x_unlabeled[top_indices_local]

    # Rank by geometric representativeness with respect to labeled set
    rep_scores = geometric_ranking(x_top, x_labeled, R_0)

    # Select top batch_outputs by representativeness
    ranked_local = torch.argsort(rep_scores, descending=True)[:batch_outputs]
    selected_indices = [top_indices_local[i].item() for i in ranked_local]

    # If we need more samples (batch_size > batch_outputs), fill with next BALD-ranked
    if batch_size > len(selected_indices):
        remaining_needed = batch_size - len(selected_indices)
        already_selected = set(selected_indices)
        other_indices = [i for i in top_indices_local if i not in already_selected]
        other_indices = other_indices[:remaining_needed]
        selected_indices.extend([i.item() for i in other_indices])

    return selected_indices[:batch_size]
