# paper-element: eq-softmax-cross-entropy
"""Paper-level check for final-layer softmax cross-entropy gradients."""

from method.method import compute_gradient_embeddings
import torch


class _FixedSoftmaxModel(torch.nn.Module):
    def forward_with_embedding(self, x):
        """Use scores log(2), log(3), whose softmax probabilities are 2/5, 3/5."""
        logits = torch.log(torch.tensor([[2.0, 3.0]], dtype=x.dtype))
        return logits, torch.ones((1, 1), dtype=x.dtype)


def test_softmax_cross_entropy_gradient_uses_log_sum_exp_probabilities():
    """The stated log-sum-exp cross-entropy has gradient (2/5, -2/5) at z=1."""
    gradients = compute_gradient_embeddings(_FixedSoftmaxModel(), torch.zeros((1, 1)))

    assert torch.allclose(
        gradients[0], torch.tensor([2.0 / 5.0, -2.0 / 5.0]), atol=1e-6
    )
