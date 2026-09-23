# paper-element: alg-kmeanspp-seeding

import numpy as np
from typing import cast

from method.method import kmeans_plus_plus_seeding


class _DeterministicDraws:
    def integers(self, high):
        assert high == 3
        return 0

    def choice(self, high, p):
        assert high == 3
        np.testing.assert_allclose(p, np.array([0.0, 9.0 / 25.0, 16.0 / 25.0]))
        return 2


def test_kmeanspp_samples_by_squared_nearest_center_distance():
    embeddings = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])

    selected = kmeans_plus_plus_seeding(
        embeddings, batch_size=2, rng=cast(np.random.Generator, _DeterministicDraws())
    )

    assert selected == [0, 2]
