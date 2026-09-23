# paper-element: eq-gradient-embedding

import torch

from method.method import compute_gradient_embeddings


class _FixedClassifier(torch.nn.Module):
    def forward_with_embedding(self, x):
        logits = torch.log(torch.tensor([[0.2, 0.8]], dtype=torch.float32))
        embedding = torch.tensor([[3.0, -1.0]], dtype=torch.float32)
        return logits, embedding


def test_predicted_label_cross_entropy_gradient_matches_output_layer_derivative():
    gradient = compute_gradient_embeddings(_FixedClassifier(), torch.tensor([[0.0]]))

    expected = torch.tensor([[0.6, -0.2, -0.6, 0.2]])
    assert torch.allclose(gradient, expected, rtol=1e-5, atol=1e-6)
