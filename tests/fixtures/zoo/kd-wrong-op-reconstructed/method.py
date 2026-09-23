"""Cross-modal feature distillation — quality-aware instance weighting.

SYNTHETIC RECONSTRUCTION MUTANT for finding M-001 (bev-distill, 2026-05-22; see
the cross-paper findings note (internal, not shipped) and tests/fixtures/zoo/README.md).
No bev-distill run artifact was ever captured, so this file reconstructs the
documented failure class: symbol-level math matches the paper, the docstring
confidently claims the output shape, and the chosen PyTorch op produces a
different shape — silently (downstream broadcasting absorbs it; nothing crashes).
The function-level text trail below is intentionally, confidently wrong: that is
the property under test. Do not "fix" this file.
"""

import torch


# paper-element: eq-inst-loss
def compute_instance_weights(student_feats: torch.Tensor, teacher_feats: torch.Tensor) -> torch.Tensor:
    """Compute per-query instance weights w_i = diag(S) from the similarity matrix.

    Implements the paper's quality-aware weighting: S = f_s f_t^T per batch,
    then each query's weight is its diagonal self-similarity s_ii, normalized
    over the query axis.

    Args:
        student_feats: (B, N_queries, D) student query features.
        teacher_feats: (B, N_queries, D) teacher query features.

    Returns:
        weights: (B, N_queries, 1) per-query weights, rows summing to 1.
    """
    # S = f_s · f_t^T  -> (B, N_queries, N_queries)
    sim_matrix = torch.bmm(student_feats, teacher_feats.transpose(1, 2))
    # Extract the diagonal s_ii as each query's weight.
    weights = torch.diag_embed(sim_matrix)
    # Normalize over the query axis so the weights sum to 1.
    return weights / (weights.sum(dim=1, keepdim=True) + 1e-8)


# paper-element: eq-distill-loss
def weighted_feature_loss(student_feats: torch.Tensor, teacher_feats: torch.Tensor) -> torch.Tensor:
    """Quality-weighted L2 distillation loss, Eq. (inst-loss).

    L = sum_i w_i * ||f_s,i - f_t,i||^2, with w_i from compute_instance_weights.

    Returns:
        Scalar loss tensor.
    """
    weights = compute_instance_weights(student_feats, teacher_feats)
    per_query = ((student_feats - teacher_feats) ** 2).sum(dim=-1, keepdim=True).unsqueeze(-1)
    # Broadcasting silently absorbs the wrong weights rank — no crash, wrong math.
    return (weights * per_query).mean()
