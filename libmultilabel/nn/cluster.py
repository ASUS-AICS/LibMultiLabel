from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import normalize
from numpy import ndarray

__all__ = ["CLUSTER_NAME", "FILE_EXTENSION", "build_label_tree"]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CLUSTER_NAME = "label_clusters"
FILE_EXTENSION = ".npy"


def build_label_tree(sparse_x: csr_matrix, sparse_y: csr_matrix, cluster_size: int, output_dir: str | Path):
    """Build label tree described in AttentionXML

    Args:
        sparse_x: features extracted from texts in CSR sparse format
        sparse_y: labels in CSR sparse format
        cluster_size: the maximal number of labels inside a cluster
        output_dir: directory to save the clusters
    """
    # skip label clustering if the clustering file already exists
    output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)
    cluster_path = output_dir / f"{CLUSTER_NAME}{FILE_EXTENSION}"
    if cluster_path.exists():
        logger.info("Clustering has finished in a previous run")
        return

    # cluster meta info
    logger.info("Performing label clustering")
    logger.info(f"Cluster size: {cluster_size}")
    num_labels = sparse_y.shape[1]
    # the height of the tree satisfies the following inequation:
    # 2**(tree_height - 1) * cluster_size < num_labels <= 2**tree_height * cluster_size
    height = int(np.ceil(np.log2(num_labels / cluster_size)))
    logger.info(f"Labels will be grouped into {2**height} clusters")

    output_dir.mkdir(parents=True, exist_ok=True)

    # generate label representations
    label_repr = normalize(sparse_y.T @ csc_matrix(sparse_x))

    # clustering process
    rng = np.random.default_rng()
    clusters = [np.arange(num_labels)]
    for _ in range(height):
        assert sum(map(len, clusters)) == num_labels

        next_clusters = []
        for idx_in_cluster in clusters:
            next_clusters.extend(_cluster_split(idx_in_cluster, label_repr, rng))
        clusters = next_clusters
        logger.info(f"Having grouped {len(clusters)} clusters")

    np.save(cluster_path, np.asarray(clusters, dtype=object))
    logger.info(f"Finish clustering, saving cluster to '{cluster_path}'")


def _cluster_split(
    idx_in_cluster: ndarray, label_repr: csr_matrix, rng: np.random.Generator
) -> tuple[ndarray, ndarray]:
    """A variant of KMeans implemented in AttentionXML. Its main differences with sklearn.KMeans are:
    1. the distance metric is cosine similarity as all label representations are normalized.
    2. the end-of-loop criterion is the difference between the new and old average in-cluster distance to centroids.
    Possible drawbacks:
        Random initialization.
        cluster_size matters.
    """
    # tol is a possible hyperparameter
    tol = 1e-4
    if tol <= 0 or tol > 1:
        raise ValueError(f"tol should be a positive number that is less than 1, got {repr(tol)} instead.")

    # the corresponding label representations in the node
    tgt_repr = label_repr[idx_in_cluster]

    # the number of leaf labels in the node
    n = len(idx_in_cluster)

    # randomly choose two points as initial centroids
    centroids = tgt_repr[rng.choice(n, size=2, replace=False)].toarray()

    # initialize distances (cosine similarity)
    old_dist = -2.0
    new_dist = -1.0

    # "c" denotes clusters
    c0_idx = None
    c1_idx = None

    while new_dist - old_dist >= tol:
        # each points' distance (cosine similarity) to the two centroids
        dist = tgt_repr @ centroids.T  # shape: (n, 2)

        # generate clusters
        # let a = dist[:, 1] - dist[:, 0], the larger the element in a is, the closer the point is to the c1
        k = n // 2
        c_idx = np.argpartition(dist[:, 1] - dist[:, 0], kth=k)
        c0_idx = c_idx[:k]
        c1_idx = c_idx[k:]

        # update distances
        # the distance is the average in-cluster distance to the centroids
        old_dist = new_dist
        new_dist = (dist[c0_idx, 0].sum() + dist[c1_idx, 1].sum()) / n

        # update centroids
        # the new centroid is the average of the points in the cluster
        centroids = normalize(
            np.asarray(
                [
                    np.squeeze(np.asarray(tgt_repr[c0_idx].sum(axis=0))),
                    np.squeeze(np.asarray(tgt_repr[c1_idx].sum(axis=0))),
                ]
            )
        )
    return idx_in_cluster[c0_idx], idx_in_cluster[c1_idx]
