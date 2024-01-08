from abc import abstractmethod
from typing import Optional

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from lightning import LightningModule
from torch import nn, Tensor
from torch.nn import Module
from torch.optim import Optimizer

from ..common_utils import argsort_top_k, dump_log
from ..nn.metrics import get_metrics, tabulate_metrics
from libmultilabel.nn import networks


class MultiLabelModel(L.LightningModule):
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
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay

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
        **kwargs,
    ):
        super().__init__(num_classes=len(classes), log_path=log_path, **kwargs)
        self.save_hyperparameters(
            ignore=["log_path"]
        )  # If log_path is saved, loading the checkpoint will cause an error since each experiment has unique log_path (result_dir).
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
        pred_logits = self(batch)
        loss = self.loss_function(pred_logits, target_labels.float())

        return loss, pred_logits


class BaseModel(LightningModule):
    def __init__(
        self,
        network: str,
        network_config: dict,
        embed_vecs: Tensor,
        num_labels: int,
        optimizer: str,
        metrics: list[str],
        val_metric: str,
        top_k: int,
        is_multiclass: bool,
        init_weight: Optional[str] = None,
        loss_func: str = "binary_cross_entropy_with_logits",
        optimizer_params: Optional[dict] = None,
        lr_scheduler: Optional[str] = None,
        metric_threshold: int = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="embed_vecs")

        self.network = getattr(networks, network)(embed_vecs=embed_vecs, num_classes=num_labels, **network_config)
        self.init_weight = init_weight
        if init_weight is not None:
            init_weight = networks.get_init_weight_func(init_weight=init_weight)
            self.network.apply(init_weight)

        self.loss_func = self.configure_loss_func(loss_func)

        # optimizer config
        self.optimizer_name = optimizer.lower()
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}

        self.lr_scheduler_name = lr_scheduler

        self.top_k = top_k

        self.num_labels = num_labels

        self.metric_list = metrics
        self.val_metric_name = val_metric
        self.test_metric_names = metrics
        self.is_multiclass = is_multiclass
        self.metric_threshold = metric_threshold

    @staticmethod
    def configure_loss_func(loss_func: str) -> Module:
        try:
            loss_func = getattr(F, loss_func)
        except AttributeError:
            raise AttributeError(f"Invalid loss function name: {loss_func}")
        return loss_func

    def configure_optimizers(self) -> Optimizer:
        parameters = [p for p in self.parameters() if p.requires_grad]

        if self.optimizer_name == "sgd":
            optimizer = optim.SGD
        elif self.optimizer_name == "adam":
            optimizer = optim.Adam
        elif self.optimizer_name == "adamw":
            optimizer = optim.AdamW
        elif self.optimizer_name == "adamax":
            optimizer = optim.Adamax
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        optimizer = optimizer(parameters, **self.optimizer_params)
        if self.lr_scheduler_name is None:
            return optimizer

        if self.lr_scheduler_name is not None and self.lr_scheduler_name.lower() == "reducelronplateau":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min" if self.val_metric_name == "loss" else "max",
            )
        else:
            raise ValueError(f"Unsupported learning rate scheduler: {self.lr_scheduler}")

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "monitor": self.val_metric_name,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def on_fit_start(self):
        self.val_metric = get_metrics(
            metric_threshold=self.metric_threshold,
            monitor_metrics=[self.val_metric_name],
            num_classes=self.num_labels,
            top_k=1 if self.is_multiclass else None,
        ).to(self.device)

    def training_step(self, batch: Tensor, batch_idx: int):
        x, y = batch
        logits = self.network(x)
        loss = self.loss_func(logits, y.float())
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int):
        x, y = batch
        logits = self.network(x)
        self.val_metric.update(torch.sigmoid(logits), y.long())

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metric.compute(), prog_bar=True)
        self.val_metric.reset()

    def on_test_start(self):
        self.test_metrics = get_metrics(
            metric_threshold=self.metric_threshold,
            monitor_metrics=self.test_metric_names,
            num_classes=self.num_labels,
            top_k=1 if self.is_multiclass else None,
        ).to(self.device)

    def test_step(self, batch: Tensor, batch_idx: int):
        x, y = batch
        logits = self.network(x)
        self.test_metrics.update(torch.sigmoid(logits), y.long())

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0):
        # lightning will put tensors on cpu
        x = batch
        logits = self.network(x)
        scores, labels = torch.topk(torch.sigmoid(logits), self.top_k)
        return scores, labels

    def forward(self, x):
        return self.network(x)


class PLTModel(BaseModel):
    def __init__(
        self,
        network: str,
        network_config: dict,
        embed_vecs: Tensor,
        num_labels: int,
        optimizer: str,
        metrics: list[str],
        val_metric: str,
        top_k: int,
        is_multiclass: bool,
        loss_func: str = "binary_cross_entropy_with_logits",
        optimizer_params: Optional[dict] = None,
        lr_scheduler: Optional[str] = None,
    ):
        super().__init__(
            network=network,
            network_config=network_config,
            embed_vecs=embed_vecs,
            num_labels=num_labels,
            optimizer=optimizer,
            metrics=metrics,
            val_metric=val_metric,
            top_k=top_k,
            is_multiclass=is_multiclass,
            loss_func=loss_func,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
        )

    def multilabel_binarize(
        self,
        logits: Tensor,
        candidates: Tensor,
        candidate_scores: Tensor,
    ) -> Tensor:
        """self-implemented MultiLabelBinarizer for AttentionXML"""
        src = torch.sigmoid(logits.detach()) * candidate_scores
        # make sure preds and src use the same precision, e.g., either float16 or float32
        preds = torch.zeros(candidates.size(0), self.num_labels + 1, device=candidates.device, dtype=src.dtype)
        preds.scatter_(dim=1, index=candidates, src=src)
        # remove dummy samples
        preds = preds[:, :-1]
        return preds

    def training_step(self, batch, batch_idx):
        x, y, candidates = batch
        logits = self.network(x, candidates=candidates)
        loss = self.loss_func(logits, torch.take_along_dim(y.float(), candidates, dim=1))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, candidates, candidate_scores = batch
        logits = self.network(x, candidates=candidates)
        # FIXME: Cannot calculate loss, candidates might contain element whose value is self.num_labels (see dataset.py)
        # loss = self.loss_func(logits, torch.from_numpy(np.concatenate([y[:, candidates], offset])))
        y_pred = self.multilabel_binarize(logits, candidates, candidate_scores)
        self.val_metric.update(y_pred, y.long())

    def test_step(self, batch, batch_idx):
        x, y, candidates, candidate_scores = batch
        logits = self.network(x, candidates=candidates)
        y_pred = self.multilabel_binarize(logits, candidates, candidate_scores)
        self.test_metrics.update(y_pred, y.long())

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, candidates, candidate_scores = batch
        logits = self.network(x, candidates=candidates)
        scores, labels = torch.topk(torch.sigmoid(logits) * candidate_scores, self.top_k)
        return scores, torch.take_along_dim(candidates, labels, dim=1)
