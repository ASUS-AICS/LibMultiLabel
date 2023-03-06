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
        similarity = x * centroids.T
        cluster = similarity.argmax(axis=1).A1

        avg_sim = np.take_along_axis(
            similarity, cluster.reshape(-1, 1), axis=1).sum() / x.shape[0]
        if prev_sim - avg_sim < tol:
            return cluster

        centroids = np.zeros(centroids.shape)
        for i in range(k):
            centroids[i] = x[cluster == i].sum(axis=0).A1
        # should centroids be stored as sparse matrices?
        centroids = sparse.csr_matrix(centroids)
        centroids = sklearn.preprocessing.normalize(
            centroids, norm='l2', axis=1)
        prev_sim = avg_sim

    return cluster


def lloyd(x: sparse.csr_matrix, k: int, max_iter: int, tol: float) -> np.ndarray:
    """Perform lloyd's algorithm k-means clustering on x.

    Args:
        x (sparse.csr_matrix): Matrix with dimensions number of points * dimension of underlying space. Each row must be L2 normalized.
        k (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        tol (float): Stopping tolerance. Lower for more optimal clustering but longer run time.

    Returns:
        np.ndarray: An array of cluster indices.
    """
    init = np.random.choice(np.arange(x.shape[0]), size=k, replace=False)
    centroids = x[init]
    prev_dist = np.inf
    for _ in range(max_iter):
        similarity = x * centroids.T
        norm_square = centroids.power(2).sum(axis=1).T
        distance_square = norm_square - 2*similarity
        cluster = distance_square.argmin(axis=1).A1

        avg_dist = np.take_along_axis(
            distance_square, cluster.reshape(-1, 1), axis=1).sum() / x.shape[0]
        if prev_dist - avg_dist < tol:
            return cluster

        centroids = np.zeros(centroids.shape)
        for i in range(k):
            centroids[i] = x[cluster == i].mean(axis=0).A1
        # should centroids be stored as sparse matrices?
        centroids = sparse.csr_matrix(centroids)
        prev_dist = avg_dist

    return cluster
