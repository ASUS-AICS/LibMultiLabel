from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import torch
from numpy import ndarray
from scipy.sparse import csr_matrix, issparse
from torch import Tensor, is_tensor
from torch.utils.data import Dataset
from tqdm import tqdm


class PlainDataset(Dataset):
    """Basic class for multi-label dataset."""

    def __init__(self, x: list | ndarray | Tensor, y: Optional[csr_matrix | ndarray | Tensor] = None):
        """General dataset class for multi-label dataset.

        Args:
            x: texts
            y: labels
        """
        if y is not None:
            assert len(x) == y.shape[0], "Sizes mismatch between x and y"
        self.x = x
        self.y = y

    def __getitem__(self, idx: int) -> tuple[Sequence, ndarray] | tuple[Sequence]:
        item = {"text": self.x[idx]}

        # train/valid/test
        if self.y is not None:
            if issparse(self.y):
                y = self.y[idx].toarray().squeeze(0)
            elif is_tensor(self.y) or isinstance(self.y, (ndarray, torch.Tensor)):
                y = self.y[idx]
            else:
                raise TypeError(
                    "The type of y should be one of scipy.csr_matrix, torch.Tensor, and numpy.ndarry."
                    f"Instead, got {type(self.y)}."
                )
            item["label"] = y
        return item

    def __len__(self):
        return len(self.x)


class PLTDataset(PlainDataset):
    """Dataset class for AttentionXML."""

    def __init__(
        self,
        x,
        y: Optional[csr_matrix | ndarray] = None,
        *,
        num_classes: int,
        mapping: ndarray,
        clusters_selected: ndarray | Tensor,
        cluster_scores: Optional[ndarray | Tensor] = None,
    ):
        """Dataset for AttentionXML.

        Args:
            x: texts
            y: labels
            num_classes: number of clusters at the current level.
            mapping: [[0,..., 7], [8,..., 15], ...]. shape: (len(clusters), cluster_size). Map from clusters to labels.
            clusters_selected: [[7, 1, 128, 6], [21, 85, 64, 103], ...]. shape: (len(x), top_k). numbers are predicted clusters
                from last level.
            cluster_scores: corresponding scores. shape: (len(x), top_k)
        """
        super().__init__(x, y)
        self.num_classes = num_classes
        self.mapping = mapping
        self.clusters_selected = clusters_selected
        self.cluster_scores = cluster_scores
        self.label_scores = None

        # labels_selected are positive clusters at the current level. shape: (len(x), cluster_size * top_k)
        # look like [[0, 1, 2, 4, 5, 18, 19,...], ...]
        self.labels_selected = [
            np.concatenate(self.mapping[labels])
            for labels in tqdm(self.clusters_selected, leave=False, desc="Retrieving labels from selected clusters")
        ]
        if self.cluster_scores is not None:
            # label_scores are corresponding scores for selected labels and
            # look like [[0.1, 0.1, 0.1, 0.4, 0.4, 0.5, 0.5,...], ...]. shape: (len(x), cluster_size * top_k)
            # notice how scores repeat for each cluster.
            self.label_scores = [
                np.repeat(scores, [len(i) for i in self.mapping[labels]])
                for labels, scores in zip(self.clusters_selected, self.cluster_scores)
            ]

        # top_k * n (n <= cluster_size). number of maximum possible number selected labels at the current level.
        self.num_labels_selected = self.clusters_selected.shape[1] * max(len(clusters) for clusters in self.mapping)

    def __getitem__(self, idx: int):
        item = {"text": self.x[idx], "labels_selected": np.asarray(self.labels_selected[idx], dtype=np.int64)}

        # train/valid/test
        if self.y is not None:
            item["label"] = self.y[idx].toarray().squeeze(0)

        # train
        if self.label_scores is None:
            # randomly select clusters as selected labels when less than required
            if len(item["labels_selected"]) < self.num_labels_selected:
                sample = np.random.randint(
                    self.num_classes, size=self.num_labels_selected - len(item["labels_selected"])
                )
                item["labels_selected"] = np.concatenate([item["labels_selected"], sample])
        # valid/test
        else:
            item["label_scores"] = self.label_scores[idx]

            # add dummy elements when less than required
            if len(item["labels_selected"]) < self.num_labels_selected:
                item["label_scores"] = np.concatenate(
                    [item["label_scores"], [-np.inf] * (self.num_labels_selected - len(item["labels_selected"]))]
                )
                item["labels_selected"] = np.concatenate(
                    [
                        item["labels_selected"],
                        [self.num_classes] * (self.num_labels_selected - len(item["labels_selected"])),
                    ]
                )

            item["label_scores"] = np.asarray(item["label_scores"], dtype=np.float32)
        return item
