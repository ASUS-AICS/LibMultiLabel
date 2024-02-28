from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy import ndarray
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.preprocessing import normalize

__all__ = ["CLUSTER_NAME", "FILE_EXTENSION", "build_label_tree"]

logger = logging.getLogger(__name__)

CLUSTER_NAME = "label_clusters"
FILE_EXTENSION = ".npy"


def build_label_tree(sparse_x: csr_matrix, sparse_y: csr_matrix, cluster_size: int, output_dir: str | Path):
    """Build a binary tree to group labels into clusters, each of which contains up tp cluster_size labels. The tree has
    several layers; nodes in the last layer correspond to the output clusters.
    Given a set of labels (0, 1, 2, 3, 4, 5) and a cluster size of 2, the resulting clusters look something like:
    ((0, 2), (1, 3), (4, 5)).

    Args:
        sparse_x: features extracted from texts in CSR sparse format
        sparse_y: binarized labels in CSR sparse format
        cluster_size: the maximum number of labels within each cluster
        output_dir: directory to store the clustering file
    """
    # skip constructing label tree if the output file already exists
    output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)
    cluster_path = output_dir / f"{CLUSTER_NAME}{FILE_EXTENSION}"
    if cluster_path.exists():
        logger.info("Clustering has finished in a previous run")
        return

    # meta info
    logger.info("Label clustering started")
    logger.info(f"Cluster size: {cluster_size}")
    num_labels = sparse_y.shape[1]
    # The height of the tree satisfies the following inequality:
    # 2**(tree_height - 1) * cluster_size < num_labels <= 2**tree_height * cluster_size
    height = int(np.ceil(np.log2(num_labels / cluster_size)))
    logger.info(f"Labels will be grouped into {2**height} clusters")

    output_dir.mkdir(parents=True, exist_ok=True)

    # For each label, sum up instances relevant to the label and normalize to get the label representation
    label_repr = normalize(sparse_y.T @ csc_matrix(normalize(sparse_x)))

    # clustering by a binary tree:
    # at each layer split each cluster to two. Leave nodes correspond to the obtained clusters.
    clusters = [np.arange(num_labels)]
    for _ in range(height):
        next_clusters = []
        for cluster in clusters:
            next_clusters.extend(_split_cluster(cluster, label_repr))
        clusters = next_clusters
        logger.info(f"Having grouped {len(clusters)} clusters")

    np.save(cluster_path, np.asarray(clusters, dtype=object))
    logger.info(f"Label clustering finished. Saving results to {repr(cluster_path)}")


def _split_cluster(cluster: ndarray, label_repr: csr_matrix) -> tuple[ndarray, ndarray]:
    """A variant of KMeans implemented in AttentionXML. Here K = 2. The cluster is partitioned into two groups, each
    with approximately equal size. Its main differences with the KMeans algorithm in scikit-learn are:
    1. the distance metric is cosine similarity.
    2. the end-of-loop criterion is the difference between the new and old average in-cluster distances to centroids.

    Args:
        cluster: a subset of labels
        label_repr: the normalized representations of the relationship between labels and texts
    """
    tol = 1e-4

    # the normalized label representations corresponding to the cluster
    tgt_repr = label_repr[cluster]

    # the number of labels in the cluster
    n = len(cluster)

    # Randomly choose two points as initial centroids and obtain their label representations
    centroids = tgt_repr[np.random.choice(n, size=2, replace=False)].toarray()

    # Initialize distances (cosine similarity)
    # Cosine similarity always falls to the interval [-1, 1]
    old_dist = -2.0
    new_dist = -1.0

    # "c" denotes clusters
    c0_idx = None
    c1_idx = None

    while new_dist - old_dist >= tol:
        # each point's distances (cosine similarity) to the two centroids
        # tgt_repr and centroids.T have been normalized
        dist = tgt_repr @ centroids.T  # shape: (n, 2)

        # generate clusters
        # let a = dist[:, 1] - dist[:, 0], the larger the element in a is, the closer the point is to c1
        k = n // 2
        c_idx = np.argpartition(dist[:, 1] - dist[:, 0], kth=k)
        c0_idx = c_idx[:k]
        c1_idx = c_idx[k:]

        # update distances
        # the new distance is the average of in-cluster distances to the centroids
        old_dist = new_dist
        new_dist = (dist[c0_idx, 0].sum() + dist[c1_idx, 1].sum()) / n

        # update centroids
        # the new centroid is the normalized average of the points in the cluster
        centroids = normalize(
            np.asarray(
                [
                    np.squeeze(np.asarray(tgt_repr[c0_idx].sum(axis=0))),
                    np.squeeze(np.asarray(tgt_repr[c1_idx].sum(axis=0))),
                ]
            )
        )
    return cluster[c0_idx], cluster[c1_idx]
