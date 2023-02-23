import numpy as np
import scipy.sparse as sparse
import sklearn.preprocessing


def spherical(x: sparse.csr_matrix, k: int, max_iter: int, tol: float) -> np.ndarray:
    """Perform spherical k-means clustering on x.

    Args:
        x (sparse.csr_matrix): Matrix with dimensions number of points * dimension of underlying space.
        k (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        tol (float): Stopping tolerance. Lower for more optimal clustering but longer run time.

    Returns:
        np.ndarray: An array of cluster indices.
    """
    x = sklearn.preprocessing.normalize(x, norm='l2', axis=1)
    init = np.random.choice(np.arange(x.shape[0]), size=k, replace=False)
    centroids = x[init]
    prev_sim = np.inf
    for _ in range(max_iter):
        similarity = x * centroids.transpose()
        cluster = similarity.argmax(axis=1)

        centroids = np.zeros(centroids.shape)
        for i in range(k):
            centroids[i] = x[cluster == i].sum(axis=0)

        avg_sim = similarity[:, cluster].sum() / x.shape[0]
        if prev_sim - avg_sim < tol:
            return cluster

        centroids = sklearn.preprocessing.normalize(
            centroids, norm='l2', axis=1)
        prev_sim = avg_sim

    return cluster
