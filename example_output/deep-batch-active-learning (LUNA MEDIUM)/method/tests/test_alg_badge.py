# paper-element: alg-badge

from method.method import select_batch


def test_badge_selects_a_seeded_kmeans_batch():
    """Algorithm 1 returns a seeded random subset of gradient embeddings."""
    import torch

    class ToyModel:
        training = True

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

        def forward_with_embedding(self, x):
            logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
            features = torch.tensor([[1.0], [3.0]])
            return logits, features

    selected = select_batch(ToyModel(), torch.zeros(2, 1), batch_size=2, seed=0)

    assert selected == [1, 0]
