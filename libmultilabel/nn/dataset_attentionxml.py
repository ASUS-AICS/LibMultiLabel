from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix, issparse
from torch import Tensor, is_tensor
from torch.utils.data import Dataset


class PlainDataset(Dataset):
    """Plain (compared to nn.data_utils.TextDataset) dataset class for multi-label dataset.
    WHY EXISTS: The reason why this class is necessary is that it can process labels in sparse format, while TextDataset
    does not.
    Moreover, TextDataset implements multilabel binarization in a mandatory way. Nevertheless, AttentionXML already does
    this while generating clusters. There is no need to do multilabel binarization again.

    Args:
        x (list | ndarray | Tensor): texts.
        y (Optional: csr_matrix | ndarray | Tensor): labels.
    """

    def __init__(self, x, y=None):
        if y is not None:
            assert len(x) == y.shape[0], "Sizes mismatch between texts and labels"
        self.x = x
        self.y = y

    def __getitem__(self, idx: int) -> tuple[Sequence, ndarray] | tuple[Sequence]:
        item = {"text": self.x[idx]}

        # train/val/test
        if self.y is not None:
            if issparse(self.y):
                y = self.y[idx].toarray().squeeze(0).astype(np.int32)
            elif isinstance(self.y, ndarray):
                y = self.y[idx].astype(np.int32)
            elif is_tensor(self.y):
                y = self.y[idx].int()
            else:
                raise TypeError(
                    "The type of y should be one of scipy.csr_matrix, numpy.ndarry, and torch.Tensor."
                    f"But got {type(self.y)} instead."
                )
            item["label"] = y
        return item

    def __len__(self):
        return len(self.x)


class PLTDataset(PlainDataset):
    """Dataset for model_1 of AttentionXML.

    Args:
        x: texts.
        y: labels.
        num_classes: number of classes.
        num_labels_selected: the number of selected labels. Pad any labels that fail to reach this number.
        labels_selected: sampled predicted labels from model_0. Shape: (len(x), predict_top_k).
        label_scores: scores for each label. Shape: (len(x), predict_top_k).
    """

    def __init__(
        self,
        x,
        y: Optional[csr_matrix | ndarray] = None,
        *,
        num_classes: int,
        num_labels_selected: int,
        labels_selected: ndarray | Tensor,
        label_scores: Optional[ndarray | Tensor] = None,
    ):
        super().__init__(x, y)
        self.num_classes = num_classes
        self.num_labels_selected = num_labels_selected
        self.labels_selected = labels_selected
        self.label_scores = label_scores

    def __getitem__(self, idx: int):
        item = {"text": self.x[idx], "labels_selected": np.asarray(self.labels_selected[idx])}

        if self.y is not None:
            item["label"] = self.y[idx].toarray().squeeze(0).astype(np.int32)

        # PyTorch requires inputs to be of the same shape. Pad any instances whose length is below num_labels_selected
        # train
        if self.label_scores is None:
            # add real labels when the number is below num_labels_selected
            if len(item["labels_selected"]) < self.num_labels_selected:
                samples = np.random.randint(
                    self.num_classes,
                    size=self.num_labels_selected - len(item["labels_selected"]),
                )
                item["labels_selected"] = np.concatenate([item["labels_selected"], samples])

        # val/test/pred
        else:
            item["label_scores"] = self.label_scores[idx]
            # add dummy labels when the number is below num_labels_selected
            if len(item["labels_selected"]) < self.num_labels_selected:
                item["label_scores"] = np.concatenate(
                    [
                        item["label_scores"],
                        [-np.inf] * (self.num_labels_selected - len(item["labels_selected"])),
                    ]
                )
                item["labels_selected"] = np.concatenate(
                    [
                        item["labels_selected"],
                        [self.num_classes] * (self.num_labels_selected - len(item["labels_selected"])),
                    ]
                )
        return item
