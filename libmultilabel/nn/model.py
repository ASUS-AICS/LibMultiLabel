from abc import abstractmethod

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..common_utils import argsort_top_k, dump_log
from ..nn.metrics import get_metrics, tabulate_metrics


class MultiLabelModel(L.LightningModule):
    """Abstract class handling Pytorch Lightning training flow

    Args:
        num_classes (int): Total number of classes.
        optimizer (str, optional): Optimizer name (i.e., sgd, adam, or adamw). Defaults to 'adam'.
        optimizer_config (dict, optional): Optimizer parameters. The keys in the dictionary should match the parameter names defined by PyTorch for the optimizer.
        lr_scheduler: (str, optional): Learning rate scheduler. Defaults to None, i.e., no learning rate scheduler. Currently, the only supported lr_scheduler is 'ReduceLROnPlateau'.
        scheduler_config (dict, optional): Learning rate scheduler parameters. The keys in the dictionary should match the parameter names defined by PyTorch for the learning rate scheduler.
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
        optimizer="adam",
        optimizer_config=None,
        lr_scheduler=None,
        scheduler_config=None,
        val_metric=None,
        metric_threshold=0.5,
        monitor_metrics=None,
        log_path=None,
        multiclass=False,
        silent=False,
        save_k_predictions=0,
        **kwargs,
    ):
        super().__init__()

        # optimizer
        self.optimizer = optimizer
        self.optimizer_config = optimizer_config if optimizer_config is not None else {}

        # lr_scheduler
        self.lr_scheduler = lr_scheduler
        self.scheduler_config = scheduler_config
        self.val_metric = val_metric

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
            optimizer = optim.SGD(parameters, **self.optimizer_config)
        elif optimizer_name == "adam":
            optimizer = optim.Adam(parameters, **self.optimizer_config)
        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(parameters, **self.optimizer_config)
        elif optimizer_name == "adamax":
            optimizer = optim.Adamax(parameters, **self.optimizer_config)
        else:
            raise RuntimeError("Unsupported optimizer: {self.optimizer}")

        if self.lr_scheduler:
            if self.lr_scheduler == "ReduceLROnPlateau":
                lr_scheduler_config = {
                    "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="min" if self.val_metric == "Loss" else "max", **dict(self.scheduler_config)
                    ),
                    "monitor": self.val_metric,
                }
            else:
                raise RuntimeError("Unsupported learning rate scheduler: {self.lr_scheduler}")
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config} if self.lr_scheduler else optimizer

    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        return self._shared_eval_epoch_end(split="val")

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx)

    def on_test_epoch_end(self):
        return self._shared_eval_epoch_end(split="test")

    def _shared_eval_step(self, batch, batch_idx):
        loss, pred_logits = self.shared_step(batch)
        pred_scores = torch.sigmoid(pred_logits)
        self.eval_metric.update(preds=pred_scores, target=batch["label"], loss=loss)

    def _shared_eval_epoch_end(self, split):
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
        pred_logits = self(batch)
        pred_scores = pred_logits.detach().cpu().numpy()
        k = self.save_k_predictions
        top_k_idx = argsort_top_k(pred_scores, k, axis=1)
        top_k_scores = np.take_along_axis(pred_scores, top_k_idx, axis=1)

        return {"top_k_pred": top_k_idx, "top_k_pred_scores": top_k_scores}

    def forward(self, batch):
        """compute predicted logits"""
        return self.network(batch)["logits"]

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
        network (nn.Module): Network (i.e., CAML, KimCNN, or XMLCNN).
        loss_function (str, optional): Loss function name (i.e., binary_cross_entropy_with_logits,
            cross_entropy). Defaults to 'binary_cross_entropy_with_logits'.
        log_path (str): Path to a directory holding the log files and models.
    """

    def __init__(
        self,
        classes,
        word_dict,
        network,
        loss_function="binary_cross_entropy_with_logits",
        log_path=None,
        **kwargs
    ):
        super().__init__(num_classes=len(classes), log_path=log_path, **kwargs)
        self.save_hyperparameters(
            ignore=["log_path"]
        )  # If log_path is saved, loading the checkpoint will cause an error since each experiment has unique log_path (result_dir).
        self.word_dict = word_dict
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
        pred_logits = self(batch)
        loss = self.loss_function(pred_logits, target_labels.float())

        return loss, pred_logits
