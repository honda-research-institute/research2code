# paper-element: alg-kmeans-plus-plus

import numpy as np

from method.method import kmeans_plus_plus_seeding


def test_kmeans_plus_plus_chooses_only_positive_distance_candidate():
    embeddings = np.array([[0.0], [10.0], [0.0]])
    selected = kmeans_plus_plus_seeding(
        embeddings, batch_size=2, rng=np.random.default_rng(0)
    )

    np.testing.assert_array_equal(selected, np.array([2, 1]))
    assert embeddings[selected[1], 0] == 10.0
