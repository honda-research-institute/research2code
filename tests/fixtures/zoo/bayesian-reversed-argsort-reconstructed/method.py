"""RECONSTRUCTION — the reversed-argsort idiom, silent direction.

The 2026-07-29 bayesian-active-learning delivery shipped this shape in
`select_batch` Step 1 (delivery-review RCA 2026-08-03, finding 6): the coder
intended "sort candidates descending by BALD score" and wrote an argsort OVER
the reversed score array. The result executes cleanly and yields an ordering
that is meaningless against the original candidate array — contrast
`gbald-negstride-selector/`, where the same descending intent used the
CORRECT idiom (`np.argsort(x)[::-1]`) and instead crashed downstream on
torch's negative-stride rejection. Together the two fixtures pin both
directions of the descending-ranking idiom: the crash stays a crash, and the
silent scramble dies at lint (US-10, the pre-smoke static gate).

The buggy line is verbatim from the delivered `method/method.py:419`; the
surrounding code is trimmed to the minimal faithful selector shape. The
original run dir is gitignored, so this is a reconstruction per the zoo
convention.
"""

import numpy as np
import torch


def compute_bald_score(model, x_candidates, mc_samples=100):
    """Stub for the delivered BALD scorer: MC-dropout mutual information."""
    with torch.no_grad():
        probs = torch.stack([model(x_candidates) for _ in range(mc_samples)])
    mean = probs.mean(dim=0)
    entropy = -(mean * torch.log(mean + 1e-12)).sum(dim=-1)
    expected = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean(dim=0)
    return entropy - expected


def geometric_ranking(candidates, x_pool, x_labeled, R_0=1.0, batch_size=10):
    """Stub for the delivered geometric core-set ranker."""
    distances = np.linalg.norm(
        x_pool[candidates][:, None, :] - x_labeled[None, :, :], axis=-1,
    ).min(axis=1)
    order = np.argsort(distances)[::-1]  # descending — the correct idiom
    return [int(candidates[i]) for i in order[:batch_size]]


def select_batch(model, x_labeled, x_unlabeled, batch_returns=100,
                 batch_size=10, mc_samples=100, R_0=1.0, seed=0):
    rng = np.random.default_rng(seed)

    if isinstance(x_unlabeled, torch.Tensor):
        x_unlabeled_t = x_unlabeled
        x_pool_np = x_unlabeled.detach().cpu().numpy()
    else:
        x_unlabeled_t = torch.from_numpy(x_unlabeled).float()
        x_pool_np = np.array(x_unlabeled)
    if isinstance(x_labeled, torch.Tensor):
        x_lab_np = x_labeled.detach().cpu().numpy()
    else:
        x_lab_np = np.array(x_labeled)

    N_pool = len(x_pool_np)
    batch_returns = min(batch_returns, N_pool)
    batch_size = min(batch_size, batch_returns)

    # --- Step 1: Select candidates and score by BALD ---
    candidate_rng_indices = rng.choice(N_pool, size=batch_returns, replace=False)
    candidate_indices = np.array(candidate_rng_indices, dtype=np.intp)

    x_candidates_t = x_unlabeled_t[candidate_indices]
    bald_scores = compute_bald_score(model, x_candidates_t, mc_samples=mc_samples)

    # Sort candidates descending by BALD score (most informative first).
    sorted_idx = np.argsort(bald_scores.cpu().numpy()[::-1])
    sorted_candidates = candidate_indices[sorted_idx]

    # --- Step 2: Rank by geometric representativeness — select batch_size ---
    selected = geometric_ranking(
        sorted_candidates,
        x_pool_np,
        x_lab_np,
        R_0=R_0,
        batch_size=batch_size,
    )

    return selected
