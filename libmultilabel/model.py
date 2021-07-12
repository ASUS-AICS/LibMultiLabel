from abc import abstractmethod
from argparse import Namespace

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim

from . import networks
from .metrics import MultiLabelMetrics
from .utils import dump_log, argsort_top_k


class MultiLabelModel(pl.LightningModule):
    """Abstract class handling Pytorch Lightning training flow
    """

    def __init__(
        self,
        learning_rate=0.0001,
        optimizer='adam',
        momentum=0.9,
        weight_decay=0,
        metric_threshold=0.5,
        monitor_metrics=None,
        log_path=None,
        silent=False,
        save_k_predictions=0,
        **kwargs
    ):
        super().__init__()
        # optimizers
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        # evaluator
        self.eval_metric = MultiLabelMetrics(metric_threshold, monitor_metrics)
        # dump log
        self.log_path = log_path
        self.silent = silent
        self.save_k_predictions = save_k_predictions

    def configure_optimizers(self):
        """Initialize an optimizer for the free parameters of the network.
        """
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer_name = self.optimizer
        if optimizer_name == 'sgd':
            optimizer = optim.SGD(parameters, self.learning_rate,
                                  momentum=self.momentum,
                                  weight_decay=self.weight_decay)
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(parameters,
                                   weight_decay=self.weight_decay,
                                   lr=self.learning_rate)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(parameters,
                                    weight_decay=self.weight_decay,
                                    lr=self.learning_rate)
        else:
            raise RuntimeError(
                'Unsupported optimizer: {self.optimizer}')

        torch.nn.utils.clip_grad_value_(parameters, 0.5)

        return optimizer

    @abstractmethod
    def shared_step(self, batch):
        """Return loss and predicted logits"""
        return NotImplemented

    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)

    def validation_step_end(self, batch_parts):
        return self._shared_eval_step_end(batch_parts)

    def validation_epoch_end(self, step_outputs):
        return self._shared_eval_epoch_end(step_outputs, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)

    def test_step_end(self, batch_parts):
        return self._shared_eval_step_end(batch_parts)

    def test_epoch_end(self, step_outputs):
        return self._shared_eval_epoch_end(step_outputs, 'test')

    def _shared_eval_step(self, batch, batch_idx):
        loss, pred_logits = self.shared_step(batch)
        return {'loss': loss.item(),
                'pred_scores': torch.sigmoid(pred_logits).detach().cpu().numpy(),
                'target': batch['label'].detach().cpu().numpy()}

    def _shared_eval_step_end(self, batch_parts):
        pred_scores = np.vstack(batch_parts['pred_scores'])
        target = np.vstack(batch_parts['target'])
        return self.eval_metric.update(target, pred_scores)

    def _shared_eval_epoch_end(self, step_outputs, split):
        metric_dict = self.eval_metric.get_metric_dict()
        self.log_dict(metric_dict)
        dump_log(metrics=metric_dict, split=split, log_path=self.log_path)

        if not self.silent and (not self.trainer or self.trainer.is_global_zero):
            print(f'====== {split} dataset evaluation result =======')
            print(self.eval_metric)
            print()
        self.eval_metric.reset()
        return metric_dict

    def predict_step(self, batch, batch_idx, dataloader_idx):
        outputs = self.network(batch['text'])
        pred_scores= torch.sigmoid(outputs['logits']).detach().cpu().numpy()
        k = self.save_k_predictions
        top_k_idx = argsort_top_k(pred_scores, k, axis=1)
        top_k_scores = np.take_along_axis(pred_scores, top_k_idx, axis=1)

        return {'top_k_pred': top_k_idx,
                'top_k_pred_scores': top_k_scores}

    def print(self, string):
        if not self.silent:
            if not self.trainer or self.trainer.is_global_zero:
                print(string)


class Model(MultiLabelModel):
    def __init__(
        self,
        device,
        model_name,
        classes,
        word_dict,
        init_weight=None,
        log_path=None,
        **kwargs
    ):
        super().__init__(log_path=log_path, **kwargs)
        self.save_hyperparameters()

        self.word_dict = word_dict
        self.classes = classes
        self.num_classes = len(self.classes)

        embed_vecs = self.word_dict.vectors
        self.network = getattr(networks, model_name)(
            embed_vecs=embed_vecs,
            num_classes=self.num_classes,
            **kwargs
        ).to(device)

        if init_weight is not None:
            init_weight = networks.get_init_weight_func(
                init_weight=init_weight)
            self.apply(init_weight)

    def shared_step(self, batch):
        target_labels = batch['label']
        outputs = self.network(batch['text'])
        pred_logits = outputs['logits']
        loss = F.binary_cross_entropy_with_logits(pred_logits, target_labels)
        return loss, pred_logits
