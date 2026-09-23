"""Zoo fixture: the VERIFIED GBALD delivery's Stage-1 core-set constructor.

Harvested verbatim (function body unmodified) from
r2c_runs/archive/bayesian-active-learning-20260703-first-verified/method/method.py
(the 2026-07-03 first verified GBALD package) so the AL Stage-1 probe
family's healthy conformance case survives archive cleanup. This is the
known-GOOD shape: k-means seeding, one seeded random acquisition, then
max-min picks with the ellipsoid rescale-then-snap-to-unused step (eta
live, no duplicate indices, picks spread across the pool).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from sklearn.cluster import KMeans


def construct_core_set(
    x_pool: torch.Tensor,
    core_set_size: int,
    eta: float = 0.9,
    R_0: float = 2000.0,
    seed: int = 42,
    k: int = 10,
) -> List[int]:
    r"""Stage 1: geometric core-set construction on an ellipsoid.

    Bootstraps the labeled set *independently of any trained model* by
    performing an iterative max-min acquisition with ellipsoid geodesic
    rescaling (Algorithm 1, Lines 3-13, Section 4.2).

    The procedure:

    1. **k-means** initialization of geometric prior centers ``U`` from
       ``x_pool`` (Algorithm 1, Line 5).
    2. Map each center to its nearest pool sample; these form the initial
       labeled set ``D_0``.
    3. **Iterate** ``core_set_size`` times:
       a. Select the unlabeled point whose *nearest distance to any already-
          labeled center* is maximal (the k-centers / max-min criterion,
          Eq. 4).  This is the selection criterion from Eq. (10) — the
          ``||L_0 - L||^2`` term is constant across candidates within one
          iteration, so the argmax reduces to the max-min distance criterion
          [concept: k-centers max-min].

          # paper-fidelity: Eq. (10) writes
          #  argmax_{x_j} min_{D_j} { ||L_0 - L||^2 + log p(y_j|x_j,theta) }
          # The log p term evaluates to log(R_0 / min_dist) when outside
          # R_0, which has the same ordering as -log(min_dist).  Combined
          # with the ||L_0 - L||^2 term, the overall selection at each
          # iteration is equivalent to a max-min distance criterion
          # (k-centers), consistent with Eq. (4).

       b. **Ellipsoid geodesic rescaling** (Eq. 11, 2nd iteration onward):
          The selected point ``x*`` is rescaled toward the previous acquisition
          ``x_i`` by the affine factor ``eta``:

              x_e* = x_i + eta * (x* - x_i)

          Then snapped to its nearest neighbor in the unlabeled pool.  This
          prevents the core-set from falling into boundary regions where
          representativeness is low (Section 4.2, Fig. 2).

    Parameters
    ----------
    x_pool : torch.Tensor, shape (N_pool, n_features)
        The full unlabeled data pool.
    core_set_size : int
        Number of core-set acquisitions (N_M in the paper).
    eta : float, default 0.9
        Ellipsoid affine factor (0 < eta < 1).  Controls rescaling
        strength (Eq. 11).  Smaller eta pulls more strongly toward the mean.
        # paper-element: hyp-eta
    R_0 : float, default 2000.0
        Geometric probability radius (Eq. 5).  Used in the prior; at demo
        scale with normalized data it does not meaningfully affect selection.
    seed : int, default 42
        Random seed for k-means initialization and initial pool selection.
    k : int, default 10
        Number of k-means clusters for geometric prior initialization.
        # paper-element: hyp-core-set-size

    Returns
    -------
    List[int]
        Pool indices forming the core-set (k-means-nearest samples + iterative
        acquisitions).  The caller adds these to the labeled set before the
        Stage 2 AL loop.

    .. rubric:: Paper references
    .. programlisting::

    # paper-element: alg-gbald
    # paper-element: concept-ellipsoid-geodesic
    # paper-element: concept-geometric-core-set
    # paper-element: concept-kcenters
    # paper-element: equation-ellipsoid-rescaling
    # paper-element: equation-kcenters-maxmin
    # paper-element: concept-two-stage-framework
    """
    rng = np.random.default_rng(seed)
    n_pool = x_pool.shape[0]

    # Clamp core_set_size to available unlabeled pool (R4)
    cs = min(core_set_size, n_pool)

    # Seed k-means cluster count to available pool
    k = min(k, n_pool)

    pool_np = x_pool.detach().cpu().numpy()

    # ---- k-means initialization (Algorithm 1, Line 5) ----
    # paper-element: equation-regulated-minimizer
    km = KMeans(n_clusters=k, n_init=1, random_state=int(rng.integers(0, 2**31)))
    km.fit(pool_np)
    centers = km.cluster_centers_

    # ---- Map k-means centers to nearest pool samples (Algorithm 1, Line 13: U') ----
    used: set[int] = set()
    # Incremental running nearest-squared-distance to any labeled center.
    # Updated in O(M*D) per iteration vs. O(M*cs*D) broadcast.
    # paper-fidelity: replaces O(N*k*D) broadcast from naive broadcast;
    # selection semantics (max-min k-centers, Eq. 4) are identical.
    nearest_sq: np.ndarray = np.full(n_pool, np.inf)

    for center in centers:
        d2 = np.sum((pool_np - center) ** 2, axis=1)
        nearest_sq = np.minimum(nearest_sq, d2)
        idx = int(np.argmin(d2))
        used.add(idx)
    # Sentinel: used points get -1 so argmax ignores them
    for idx in used:
        nearest_sq[idx] = -1.0

    # ---- Iterative core-set acquisition (Algorithm 1, Lines 6-11) ----
    # paper-element: equation-acquisition-criterion
    core_indices: list[int] = list(used)

    # First acquisition starts from a random pool point (not in used set)
    # to seed the ellipsoid rescaling trajectory
    available = [j for j in range(n_pool) if j not in used]
    if not available:
        return core_indices  # pool exhausted

    first_pool_idx = int(rng.choice(len(available)))
    prev_idx = available[first_pool_idx]
    core_indices.append(prev_idx)
    used.add(prev_idx)
    nearest_sq[prev_idx] = -1.0
    cs -= 1  # consumed one slot

    for _ in range(cs):
        # Select the point farthest from any labeled center (max-min rule, Eq. 4)
        # Sentinel values (-1) for used points are ignored by argmax
        # paper-element: equation-kcenters-maxmin
        sel_idx = int(np.argmax(nearest_sq))

        # ---- Ellipsoid geodesic rescaling (Eq. 11) ----
        # paper-element: equation-ellipsoid-rescaling
        rescaled = pool_np[prev_idx] + eta * (pool_np[sel_idx] - pool_np[prev_idx])

        # Snap rescaled position to nearest available neighbor
        sq_snap = np.sum((pool_np - rescaled) ** 2, axis=1)
        # Exclude already-used points from snap candidates
        sq_snap[nearest_sq < 0] = np.inf
        snap_idx = int(np.argmin(sq_snap))

        core_indices.append(snap_idx)
        used.add(snap_idx)
        nearest_sq[snap_idx] = -1.0

        # Update running nearest-squared-distance with new center (incremental O(M*D))
        d2_new = np.sum((pool_np - pool_np[snap_idx]) ** 2, axis=1)
        nearest_sq = np.minimum(nearest_sq, d2_new)

        prev_idx = snap_idx

    return core_indices
