from __future__ import annotations

import logging
from abc import abstractmethod
from collections import deque
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn, Tensor
from torch.nn import Module
from torch.optim import Optimizer

from ..common_utils import dump_log, argsort_top_k
from ..nn.metrics import get_metrics, tabulate_metrics, list2metrics
from libmultilabel.nn import networks


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
            raise RuntimeError(f"Unsupported optimizer: {self.optimizer}")

        torch.nn.utils.clip_grad_value_(parameters, 0.5)
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
        network: Module,
        loss_function: str = "binary_cross_entropy_with_logits",
        log_path: str | None = None,
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
        outputs = self.network(batch)
        pred_logits = outputs["logits"]
        loss = self.loss_function(pred_logits, target_labels.float())
        return loss, pred_logits


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        network: str,
        network_config: dict,
        embed_vecs,
        optimizer: str,
        num_labels: int,
        metrics: list[str],
        top_k: int,
        loss_fn: str = "binary_cross_entropy_with_logits",
        optimizer_params: Optional[dict] = None,
        swa_epoch_start: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["embed_vecs"])

        try:
            self.network = getattr(networks, network)(embed_vecs=embed_vecs, num_classes=num_labels, **network_config)
        except AttributeError as e:
            logging.warning(e)
            raise AttributeError(f"Invalid network name: {network}")

        self.loss_fn = self.configure_loss_fn(loss_fn)

        # optimizer config
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}

        self.top_k = top_k if top_k is not None else 100

        self.swa_epoch_start = swa_epoch_start

        self.num_labels = num_labels
        self.state = {}

        self.metric_list = metrics
        self.valid_metrics = list2metrics(["ndcg@5"], self.num_labels)
        self.test_metrics = list2metrics(metrics, self.num_labels)

        self._clip_grad = 5.0
        self._grad_norm_queue = deque([torch.tensor(float("inf"))], maxlen=5)

        self.state = {}

    @staticmethod
    def configure_loss_fn(loss_fn: str) -> Module:
        try:
            loss_fn = getattr(F, loss_fn)
        except AttributeError:
            raise AttributeError(f"Invalid loss function name: {loss_fn}")
        return loss_fn

    def configure_optimizers(self) -> Optimizer:
        try:
            optimizer = getattr(torch.optim, self.optimizer)(self.parameters(), **self.optimizer_params)
        except AttributeError:
            raise AttributeError(f"Invalid optimizer name: {self.optimizer}")
        except TypeError:
            raise TypeError(f"Invalid optimizer params in {self.optimizer_params}")
        return optimizer

    def on_train_epoch_start(self):
        if self.current_epoch == self.swa_epoch_start:
            self.swa_init()

    def training_step(self, batch: Tensor, batch_idx: int):
        """log metrics on step"""
        x, y = batch
        logits = self.network(x)
        loss = self.loss_fn(logits, y)
        return loss

    def swa_init(self):
        if "swa" not in self.state:
            swa_state = self.state["swa"] = {"models_num": 1}
            for n, p in self.network.named_parameters():
                if self.trainer.is_global_zero:
                    logging.info("SWA Initializing")
                swa_state[n] = p.data.clone().detach()

    def swa_step(self):
        if "swa" in self.state:
            swa_state = self.state["swa"]
            swa_state["models_num"] += 1
            beta = 1.0 / swa_state["models_num"]
            with torch.no_grad():
                for n, p in self.network.named_parameters():
                    swa_state[n].mul_(1.0 - beta).add_(p.data, alpha=beta)

    def swap_swa_params(self):
        if "swa" in self.state:
            swa_state = self.state["swa"]
            for n, p in self.network.named_parameters():
                p.data, swa_state[n] = swa_state[n], p.data

    def on_validation_start(self):
        self.swa_step()
        self.swap_swa_params()

    def validation_step(self, batch: Tensor, batch_idx: int):
        x, y = batch
        logits = self.network(x)
        self.valid_metrics.update(torch.sigmoid(logits), y.long())

    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute(), prog_bar=True)
        self.valid_metrics.reset()

    def on_validation_end(self):
        self.swap_swa_params()

    def test_step(self, batch: Tensor, batch_idx: int):
        x, y = batch
        logits = self.network(x)
        self.test_metrics.update(torch.sigmoid(logits).detach(), y.long())

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0):
        # unequal distributed inference is not supported by lightning (v2.0) yet.
        # make sure prediction is on a single GPU
        x = batch
        logits = self.network(x)
        scores, labels = torch.topk(torch.sigmoid(logits), self.top_k)
        return scores.detach().cpu(), labels.detach().cpu()

    def forward(self, x):
        return self.network(x)

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint):
            if k not in self.CHECKPOINT_KEYS:
                checkpoint.pop(k)

    def on_after_backward(self):
        if self._clip_grad is not None:
            max_norm = max(self._grad_norm_queue)
            total_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm * self._clip_grad)
            self._grad_norm_queue += [min(total_norm, max_norm * 2.0, torch.tensor(1.0))]
            if total_norm > max_norm * self._clip_grad:
                if self.trainer.is_global_zero:
                    logging.warning(f"Clipping gradients with total norm {total_norm:.4f} and max norm {max_norm:.4f}")


class PLTModel(BaseModel):
    def __init__(
        self,
        network: str,
        network_config: dict,
        embed_vecs,
        optimizer: str,
        num_nodes: int,
        metrics: list[str],
        top_k: int,
        eval_metric: str,
        loss_fn: str = "binary_cross_entropy_with_logits",
        optimizer_params: Optional[dict] = None,
        swa_epoch_start: Optional[int] = None,
    ):
        super().__init__(
            network=network,
            network_config=network_config,
            embed_vecs=embed_vecs,
            optimizer=optimizer,
            num_labels=num_nodes,
            metrics=metrics,
            top_k=top_k,
            loss_fn=loss_fn,
            optimizer_params=optimizer_params,
            swa_epoch_start=swa_epoch_start,
        )
        self.state["best"] = {}
        self.eval_metric = "_".join(["valid", eval_metric.lower()])
        self.best_monitor = -1

    def multilabel_binarize(
        self,
        candidates: Tensor,
        logits: Tensor,
        candidate_scores: Tensor,
    ) -> Tensor:
        """self-implemented MultiLabelBinarizer for AttentionXML using Tensor"""
        src = torch.sigmoid(logits) * candidate_scores
        # make sure preds and src use the same precision, e.g., either float16 or float32
        preds = torch.zeros(candidates.size(0), self.num_labels + 1, device=candidates.device, dtype=src.dtype)
        preds.scatter_(dim=1, index=candidates, src=src)
        # remove dummy samples
        preds = preds[:, :-1]
        return preds

    def training_step(self, batch, batch_idx):
        x, y, candidates = batch
        logits = self.network(x, candidates=candidates)
        loss = self.loss_fn(logits, torch.take_along_dim(y, candidates, dim=1))
        # self.log("loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def on_train_end(self):
        model_dict = self.network.state_dict()
        for key in model_dict:
            model_dict[key][:] = self.state["best"][key]

    def validation_step(self, batch, batch_idx):
        x, y, candidates, candidate_scores = batch
        logits = self.network(x, candidates=candidates)
        # FIXME: Cannot calculate loss, candidates might contain element whose value is self.num_labels (see dataset)
        # loss = self.loss_fn(logits, torch.from_numpy(np.concatenate([y[:, candidates], offset])))
        y_pred = self.multilabel_binarize(candidates, logits.detach(), candidate_scores)
        self.valid_metrics.update(y_pred, y.long())

    def on_validation_epoch_end(self):
        metric = self.valid_metrics.compute()
        self.valid_metrics.reset()
        self.log_dict(metric, prog_bar=True, sync_dist=True)
        # TODO: Remove hard code
        if metric[self.eval_metric].item() > self.best_monitor:
            self.best_monitor = metric[self.eval_metric].item()
            model_dict = self.network.state_dict()
            for key in model_dict:
                self.state["best"][key] = model_dict[key].cpu().detach()

    def test_step(self, batch, batch_idx):
        x, y, candidates, candidate_scores = batch
        logits = self.network(x, candidates=candidates)
        y_pred = self.multilabel_binarize(candidates, logits.detach(), candidate_scores)
        self.test_metrics.update(y_pred, y.long())

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # unequal distributed inference is not supported by lightning (v2.0) yet.
        # make sure prediction is on a single GPU
        x, candidates, candidate_scores = batch
        logits = self.network(x, candidates=candidates)
        scores, labels = torch.topk(torch.sigmoid(logits) * candidate_scores, self.top_k)
        return scores.cpu(), torch.take_along_dim(candidates, labels, dim=1).cpu()

    def load_from_pretrained(self, state_dict: dict):
        self.network.embedding.load_state_dict(
            {n.split(".", 2)[-1]: p for n, p in state_dict.items() if n.startswith("network.embedding")}
        )
        self.network.encoder.load_state_dict(
            {n.split(".", 2)[-1]: p for n, p in state_dict.items() if n.startswith("network.encoder")}
        )
        self.network.output.load_state_dict(
            {n.split(".", 2)[-1]: p for n, p in state_dict.items() if n.startswith("network.output")}
        )
