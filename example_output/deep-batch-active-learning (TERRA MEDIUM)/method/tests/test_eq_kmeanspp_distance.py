import numpy as np
from typing import cast

from method.method import kmeans_plus_plus_seeding

# paper-element: eq-kmeanspp-distance


class _RecordedRng:
    def __init__(self):
        self.probabilities = []
        self._integer_draws = iter([0])

    def integers(self, high):
        return next(self._integer_draws)

    def choice(self, n_candidates, p):
        self.probabilities.append(p.copy())
        return [1, 2][len(self.probabilities) - 1]


def test_kmeanspp_uses_the_nearest_selected_center_distance():
    embeddings = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    rng = _RecordedRng()

    kmeans_plus_plus_seeding(
        embeddings, batch_size=3, rng=cast(np.random.Generator, rng)
    )

    assert np.allclose(rng.probabilities[0], [0.0, 0.2, 0.8])
    assert np.allclose(rng.probabilities[1], [0.0, 0.0, 1.0])
