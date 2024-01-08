from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from numpy import ndarray
from scipy.sparse import csr_matrix, issparse
from torch import Tensor, is_tensor
from torch.utils.data import Dataset
from tqdm import tqdm


class MultiLabelDataset(Dataset):
    """Basic class for multi-label dataset."""

    def __init__(self, x: list | ndarray | Tensor, y: Optional[csr_matrix | ndarray | Tensor] = None):
        """General dataset class for multi-label dataset.

        Args:
            x: text.
            y: labels.
        """
        if y is not None:
            assert len(x) == y.shape[0], "Sizes mismatch between x and y"
        self.x = x
        self.y = y

    def __getitem__(self, idx: int) -> tuple[Sequence, ndarray] | tuple[Sequence]:
        x = self.x[idx]

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
            return x, y
        # predict
        return x

    def __len__(self):
        return len(self.x)


class PLTDataset(MultiLabelDataset):
    """Dataset class for AttentionXML."""

    def __init__(
        self,
        x,
        y: Optional[csr_matrix | ndarray] = None,
        *,
        num_labels: int,
        mapping: ndarray,
        node_label: ndarray | Tensor,
        node_score: Optional[ndarray | Tensor] = None,
    ):
        """Dataset for FastAttentionXML.
        ~ means variable length.

        Args:
            x: text
            y: labels
            num_labels: number of nodes at the current level.
            mapping: [[0,..., 7], [8,..., 15], ...]. shape: (len(nodes), ~cluster_size). parent nodes to child nodes.
                Cluster size will only vary at the last level.
            node_label: [[7, 1, 128, 6], [21, 85, 64, 103], ...]. shape: (len(x), top_k). numbers are predicted nodes
                from last level.
            node_score: corresponding scores. shape: (len(x), top_k)
        """
        super().__init__(x, y)
        self.num_labels = num_labels
        self.mapping = mapping
        self.node_label = node_label
        self.node_score = node_score
        self.candidate_scores = None

        # candidate are positive nodes at the current level. shape: (len(x), ~cluster_size * top_k)
        # look like [[0, 1, 2, 4, 5, 18, 19,...], ...]
        prog = rank_zero_only(tqdm)(self.node_label, leave=False, desc="Generating candidates")
        if prog is None:
            prog = self.node_label
        self.candidates = [np.concatenate(self.mapping[labels]) for labels in prog]
        if self.node_score is not None:
            # candidate_scores are corresponding scores for candidates and
            # look like [[0.1, 0.1, 0.1, 0.4, 0.4, 0.5, 0.5,...], ...]. shape: (len(x), ~cluster_size * top_k)
            # notice how scores repeat for each cluster.
            self.candidate_scores = [
                np.repeat(scores, [len(i) for i in self.mapping[labels]])
                for labels, scores in zip(self.node_label, self.node_score)
            ]

        # top_k * n (n <= cluster_size). number of maximum possible number candidates at the current level.
        self.num_candidates = self.node_label.shape[1] * max(len(node) for node in self.mapping)

    def __getitem__(self, idx: int):
        x = self.x[idx]
        candidates = np.asarray(self.candidates[idx], dtype=np.int64)

        # train/valid/test
        if self.y is not None:
            # squeezing is necessary here because csr_matrix.toarray() always returns a 2d array
            # e.g., np.ndarray([[0, 1, 2]])
            y = self.y[idx].toarray().squeeze(0)

            # train
            if self.candidate_scores is None:
                # randomly select nodes as candidates when less than required
                if len(candidates) < self.num_candidates:
                    sample = np.random.randint(self.num_labels, size=self.num_candidates - len(candidates))
                    candidates = np.concatenate([candidates, sample])
                return x, y, candidates

            # valid/test
            else:
                candidate_scores = self.candidate_scores[idx]

                # add dummy elements when less than required
                if len(candidates) < self.num_candidates:
                    candidate_scores = np.concatenate(
                        [candidate_scores, [-np.inf] * (self.num_candidates - len(candidates))]
                    )
                    candidates = np.concatenate(
                        [candidates, [self.num_labels] * (self.num_candidates - len(candidates))]
                    )

                candidate_scores = np.asarray(candidate_scores, dtype=np.float32)
                return x, y, candidates, candidate_scores

        # predict
        else:
            candidate_scores = self.candidate_scores[idx]

            # add dummy elements when less than required
            if len(candidates) < self.num_candidates:
                candidate_scores = np.concatenate(
                    [candidate_scores, [-np.inf] * (self.num_candidates - len(candidates))]
                )
                candidates = np.concatenate([candidates, [self.num_labels] * (self.num_candidates - len(candidates))])

            candidate_scores = np.asarray(candidate_scores, dtype=np.float32)
            return x, candidates, candidate_scores
