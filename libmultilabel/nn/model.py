from abc import abstractmethod

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..common_utils import dump_log, argsort_top_k
from ..nn.metrics import get_metrics, tabulate_metrics


class MultiLabelModel(pl.LightningModule):
    """Abstract class handling Pytorch Lightning training flow

    Args:
        num_classes (int): Total number of classes.
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 0.0001.
        optimizer (str, optional): Optimizer name (i.e., sgd, adam, or adamw). Defaults to 'adam'.
        momentum (float, optional): Momentum factor for SGD only. Defaults to 0.9.
        weight_decay (int, optional): Weight decay factor. Defaults to 0.
        metric_threshold (float, optional): The decision value threshold over which a label is predicted as positive. Defaults to 0.5.
        monitor_metrics (list, optional): Metrics to monitor while validating. Defaults to None.
        log_path (str): Path to a directory holding the log files and models.
        multiclass (bool, optional): Enable multiclass mode. Defaults to False.
        silent (bool, optional): Enable silent mode. Defaults to False.
        save_k_predictions (int, optional): Save top k predictions on test set. Defaults to 0.
    """

    def __init__(
        self,
        num_classes,
        learning_rate=0.0001,
        optimizer="adam",
        momentum=0.9,
        weight_decay=0,
        metric_threshold=0.5,
        monitor_metrics=None,
        log_path=None,
        multiclass=False,
        silent=False,
        save_k_predictions=0,
        **kwargs
    ):
        super().__init__()

        # optimizer
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay

        # dump log
        self.log_path = log_path
        self.silent = silent
        self.save_k_predictions = save_k_predictions

        # metrics for evaluation
        self.multiclass = multiclass
        top_k = 1 if self.multiclass else None
        self.eval_metric = get_metrics(metric_threshold, monitor_metrics, num_classes, top_k=top_k)

    @abstractmethod
    def shared_step(self, batch):
        """Return loss and predicted logits"""
        return NotImplemented

    def configure_optimizers(self):
        """Initialize an optimizer for the free parameters of the network."""
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer_name = self.optimizer
        if optimizer_name == "sgd":
            optimizer = optim.SGD(
                parameters, self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay
            )
        elif optimizer_name == "adam":
            optimizer = optim.Adam(parameters, weight_decay=self.weight_decay, lr=self.learning_rate)
        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(parameters, weight_decay=self.weight_decay, lr=self.learning_rate)
        elif optimizer_name == "adamax":
            optimizer = optim.Adamax(parameters, weight_decay=self.weight_decay, lr=self.learning_rate)
        else:
            raise RuntimeError("Unsupported optimizer: {self.optimizer}")

        torch.nn.utils.clip_grad_value_(parameters, 0.5)

        return optimizer

    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)

    def validation_step_end(self, batch_parts):
        return self._shared_eval_step_end(batch_parts)

    def validation_epoch_end(self, step_outputs):
        return self._shared_eval_epoch_end(step_outputs, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)

    def test_step_end(self, batch_parts):
        return self._shared_eval_step_end(batch_parts)

    def test_epoch_end(self, step_outputs):
        return self._shared_eval_epoch_end(step_outputs, "test")

    def _shared_eval_step(self, batch, batch_idx):
        loss, pred_logits = self.shared_step(batch)
        return {
            "batch_idx": batch_idx,
            "loss": loss,
            "pred_scores": torch.sigmoid(pred_logits),
            "target": batch["label"],
        }

    def _shared_eval_step_end(self, batch_parts):
        return self.eval_metric.update(
            preds=batch_parts["pred_scores"], target=batch_parts["target"], loss=batch_parts["loss"]
        )

    def _shared_eval_epoch_end(self, step_outputs, split):
        """Get scores such as `Micro-F1`, `Macro-F1`, and monitor metrics defined
        in the configuration file in the end of an epoch.

        Args:
            step_outputs (list): List of the return values from the val or test step end.
            split (str): One of the `val` or `test`.

        Returns:
            metric_dict (dict): Scores for all metrics in the dictionary format.
        """
        metric_dict = self.eval_metric.compute()
        self.log_dict(metric_dict)
        for k, v in metric_dict.items():
            metric_dict[k] = v.item()
        if self.log_path:
            dump_log(metrics=metric_dict, split=split, log_path=self.log_path)
        self.print(tabulate_metrics(metric_dict, split))
        self.eval_metric.reset()
        return metric_dict

    def predict_step(self, batch, batch_idx):
        """`predict_step` is triggered when calling `trainer.predict()`.
        This function is used to get the top-k labels and their prediction scores.

        Args:
            batch (dict): A batch of text and label.
            batch_idx (int): Index of current batch.

        Returns:
            dict: Top k label indexes and the prediction scores.
        """
        _, pred_logits = self.shared_step(batch)
        pred_scores = pred_logits.detach().cpu().numpy()
        k = self.save_k_predictions
        top_k_idx = argsort_top_k(pred_scores, k, axis=1)
        top_k_scores = np.take_along_axis(pred_scores, top_k_idx, axis=1)

        return {"top_k_pred": top_k_idx, "top_k_pred_scores": top_k_scores}

    def print(self, *args, **kwargs):
        """Prints only from process 0 and not in silent mode. Use this in any
        distributed mode to log only once."""

        if not self.silent:
            # print() in LightningModule to print only from process 0
            super().print(*args, **kwargs)


class Model(MultiLabelModel):
    """A class that implements `MultiLabelModel` for initializing and training a neural network.

    Args:
        classes (list): List of class names.
        word_dict (torchtext.vocab.Vocab): A vocab object which maps tokens to indices.
        embed_vecs (torch.Tensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        network (nn.Module): Network (i.e., CAML, KimCNN, or XMLCNN).
        loss_function (str, optional): Loss function name (i.e., binary_cross_entropy_with_logits,
            cross_entropy). Defaults to 'binary_cross_entropy_with_logits'.
        log_path (str): Path to a directory holding the log files and models.
    """

    def __init__(
        self,
        classes,
        word_dict,
        embed_vecs,
        network,
        loss_function="binary_cross_entropy_with_logits",
        log_path=None,
        **kwargs
    ):
        super().__init__(num_classes=len(classes), log_path=log_path, **kwargs)
        self.save_hyperparameters()
        self.word_dict = word_dict
        self.embed_vecs = embed_vecs
        self.classes = classes
        self.network = network
        self.configure_loss_function(loss_function)

    def configure_loss_function(self, loss_function):
        assert hasattr(
            F, loss_function
        ), """
            Invalid `loss_function`. Make sure the loss function is defined here:
            https://pytorch.org/docs/stable/nn.functional.html#loss-functions"""
        self.loss_function = getattr(F, loss_function)

    def shared_step(self, batch):
        """Return loss and predicted logits of the network.

        Args:
            batch (dict): A batch of text and label.

        Returns:
            loss (torch.Tensor): Loss between target and predict logits.
            pred_logits (torch.Tensor): The predict logits (batch_size, num_classes).
        """
        target_labels = batch["label"]
        outputs = self.network(batch)
        pred_logits = outputs["logits"]
        loss = self.loss_function(pred_logits, target_labels.float())

        return loss, pred_logits
