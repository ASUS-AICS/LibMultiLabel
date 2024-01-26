from typing import Optional

import torch
from torch import Tensor

from .model import MultiLabelModel


class PLTModel(MultiLabelModel):
    def __init__(
        self,
        classes,
        word_dict,
        embed_vecs,
        network,
        loss_function="binary_cross_entropy_with_logits",
        log_path=None,
        **kwargs,
    ):
        super().__init__(
            classes=classes,
            word_dict=word_dict,
            embed_vecs=embed_vecs,
            network=network,
            loss_function=loss_function,
            log_path=log_path,
            **kwargs,
        )

    def multilabel_binarize(
        self,
        logits: Tensor,
        samples: Tensor,
        label_scores: Tensor,
    ) -> Tensor:
        """self-implemented MultiLabelBinarizer for AttentionXML"""
        src = torch.sigmoid(logits.detach()) * label_scores
        # make sure preds and src use the same precision, e.g., either float16 or float32
        preds = torch.zeros(samples.size(0), self.num_labels + 1, device=samples.device, dtype=src.dtype)
        preds.scatter_(dim=1, index=samples, src=src)
        # remove dummy samples
        preds = preds[:, :-1]
        return preds

    def training_step(self, batch, batch_idx):
        x = batch["text"]
        y = batch["label"]
        samples = batch["samples"]
        logits = self.network(x, samples=samples)
        loss = self.loss_func(logits, torch.take_along_dim(y.float(), samples, dim=1))
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["text"]
        y = batch["label"]
        samples = batch["samples"]
        label_scores = batch["label_scores"]
        logits = self.network(x, samples=samples)
        y_pred = self.multilabel_binarize(logits, samples, label_scores)
        self.val_metric.update(y_pred, y.long())

    def test_step(self, batch, batch_idx):
        x = batch["text"]
        y = batch["label"]
        samples = batch["samples"]
        label_scores = batch["label_scores"]
        logits = self.network(x, samples=samples)
        y_pred = self.multilabel_binarize(logits, samples, label_scores)
        self.test_metrics.update(y_pred, y.long())

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch["text"]
        samples = batch["samples"]
        label_scores = batch["label_scores"]
        logits = self.network(x, samples=samples)
        scores, labels = torch.topk(torch.sigmoid(logits) * label_scores, self.top_k)
        return scores, torch.take_along_dim(samples, labels, dim=1)
