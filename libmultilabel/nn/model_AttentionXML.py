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

    def scatter_logits(
        self,
        logits: Tensor,
        labels_selected: Tensor,
        label_scores: Tensor,
    ) -> Tensor:
        """For each instance, we only have predictions on selected labels. This subroutine maps these predictions to
        the whole label space. The scores of unsampled labels are set to 0."""
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
        y = torch.take_along_dim(batch["label"], batch["labels_selected"], dim=1)
        pred_logits = self(batch)
        loss = self.loss_function(pred_logits, y)
        return loss, pred_logits

    def _shared_eval_step(self, batch, batch_idx):
        logits = self(batch)
        logits = self.scatter_logits(logits, batch["labels_selected"], batch["label_scores"])
        self.eval_metric.update(logits, batch["label"].long())

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch)
        scores, labels = torch.topk(torch.sigmoid(logits) * batch["label_scores"], self.save_k_predictions)
        # This calculation is to align with LibMultiLabel class where logits rather than probabilities are returned
        logits = torch.logit(scores)
        return {
            "top_k_pred": torch.take_along_dim(batch["labels_selected"], labels, dim=1).numpy(force=True),
            "top_k_pred_scores": logits.numpy(force=True),
        }
