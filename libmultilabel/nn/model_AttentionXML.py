import torch
from torch import Tensor

from .model import Model


class PLTModel(Model):
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

    def scatter_preds(
        self,
        logits: Tensor,
        labels_selected: Tensor,
        label_scores: Tensor,
    ) -> Tensor:
        """map predictions from sample space to label space. The scores of unsampled labels are set to 0."""
        src = torch.sigmoid(logits.detach()) * label_scores
        preds = torch.zeros(
            labels_selected.size(0), len(self.classes) + 1, device=labels_selected.device, dtype=src.dtype
        )
        preds.scatter_(dim=1, index=labels_selected, src=src)
        # remove dummy labels
        preds = preds[:, :-1]
        return preds

    def shared_step(self, batch):
        """Return loss and predicted logits of the network.

        Args:
            batch (dict): A batch of text and label.

        Returns:
            loss (torch.Tensor): Loss between target and predict logits.
            pred_logits (torch.Tensor): The predict logits (batch_size, num_classes).
        """
        x = batch["text"]
        y = batch["label"]
        labels_selected = batch["labels_selected"]
        logits = self.network(x, labels_selected=labels_selected)["logits"]
        loss = self.loss_function(logits, torch.take_along_dim(y.float(), labels_selected, dim=1))
        return loss, logits

    def _shared_eval_step(self, batch, batch_idx):
        x = batch["text"]
        y = batch["label"]
        labels_selected = batch["labels_selected"]
        label_scores = batch["label_scores"]
        logits = self.network(x, labels_selected=labels_selected)["logits"]
        y_pred = self.scatter_preds(logits, labels_selected, label_scores)
        self.eval_metric.update(y_pred, y.long())

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch["text"]
        labels_selected = batch["labels_selected"]
        label_scores = batch["label_scores"]
        logits = self.network(x, labels_selected=labels_selected)["logits"]
        scores, labels = torch.topk(torch.sigmoid(logits) * label_scores, self.top_k)
        return scores.numpy(force=True), torch.take_along_dim(labels_selected, labels, dim=1).numpy(force=True)
