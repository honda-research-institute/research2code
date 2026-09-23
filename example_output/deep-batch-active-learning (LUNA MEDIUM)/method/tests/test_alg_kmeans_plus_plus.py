# paper-element: alg-kmeans-plus-plus

import numpy as np

from method.method import kmeans_plus_plus_seeding


def test_kmeans_plus_plus_samples_by_squared_nearest_distance():
    """A zero-distance point has zero chance at the next seeding step."""
    embeddings = np.array([[0.0], [1.0], [10.0]])

    selected = kmeans_plus_plus_seeding(embeddings, batch_size=3, seed=0)

    assert selected == [2, 0, 1]
