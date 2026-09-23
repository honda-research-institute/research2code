# paper-element: eq-kmeanspp-probability

import numpy as np
from typing import cast

from method.method import kmeans_plus_plus_seeding


class _RecordingRng:
    def __init__(self):
        self.probabilities = None

    def integers(self, high):
        assert high == 3
        return 0

    def choice(self, high, p):
        assert high == 3
        self.probabilities = p
        return 2


def test_second_center_uses_squared_distance_probabilities():
    embeddings = np.array([[0.0], [1.0], [3.0]])
    rng = _RecordingRng()

    selected = kmeans_plus_plus_seeding(
        embeddings, batch_size=2, rng=cast(np.random.Generator, rng)
    )

    assert rng.probabilities is not None
    np.testing.assert_allclose(rng.probabilities, np.array([0.0, 0.1, 0.9]))
    assert selected == [0, 2]
