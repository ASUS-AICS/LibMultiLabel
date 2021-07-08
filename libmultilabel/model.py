from abc import abstractmethod
from argparse import Namespace

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.utilities.parsing import AttributeDict

from . import networks
from .metrics import MultiLabelMetrics
from .utils import dump_log


class MultiLabelModel(pl.LightningModule):
    """Abstract class handling Pytorch Lightning training flow"""

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(config, Namespace):
            config = vars(config)
        if isinstance(config, dict):
            config = AttributeDict(config)
        self.config = config
        self.eval_metric = MultiLabelMetrics(self.config)

    def configure_optimizers(self):
        """Initialize an optimizer for the free parameters of the network.
        """
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer_name = self.config.optimizer
        if optimizer_name == 'sgd':
            optimizer = optim.SGD(parameters, self.config.learning_rate,
                                  momentum=self.config.momentum,
                                  weight_decay=self.config.weight_decay)
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(parameters,
                                   weight_decay=self.config.weight_decay,
                                   lr=self.config.learning_rate)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(parameters,
                                    weight_decay=self.config.weight_decay,
                                    lr=self.config.learning_rate)
        else:
            raise RuntimeError(
                'Unsupported optimizer: {self.config.optimizer}')

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
        loss, pred_logits = self.shared_step(batch)

        return {'loss': loss.item(),
                'pred_scores': torch.sigmoid(pred_logits).detach().cpu().numpy(),
                'target': batch['label'].detach().cpu().numpy()}

    def validation_step_end(self, batch_parts):
        pred_scores = np.vstack(batch_parts['pred_scores'])
        target = np.vstack(batch_parts['target'])
        self.eval_metric.add_values(target, pred_scores)

    def validation_epoch_end(self, step_outputs):
        return self.evaluate(step_outputs, 'val')

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_step_end(self, batch_parts):
        self.validation_step_end(batch_parts)

    def test_epoch_end(self, step_outputs):
        return self.evaluate(step_outputs, 'test')

    def predict_step(self, batch, batch_idx, dataloader_idx):
        outputs = self.network(batch['text'])
        pred_scores= torch.sigmoid(outputs['logits']).detach().cpu().numpy()
        k = self.config.save_k_predictions
        unsorted_top_k_idx = np.argpartition(pred_scores, -k, axis=1)[:,-k:]
        unsorted_top_k_scores = np.take_along_axis(pred_scores, unsorted_top_k_idx, axis=1)
        sorted_order = np.argsort(-unsorted_top_k_scores, axis=1)
        sorted_top_k_idx = np.take_along_axis(unsorted_top_k_idx, sorted_order, axis=1)
        sorted_top_k_scores = np.take_along_axis(unsorted_top_k_scores, sorted_order, axis=1)

        return {'top_k_pred': sorted_top_k_idx,
                'top_k_pred_scores': sorted_top_k_scores}

    def evaluate(self, step_outputs, split):
        metric_dict = self.eval_metric.get_metric_dict()
        self.log_dict(metric_dict)
        dump_log(config=self.config, metrics=metric_dict, split=split)

        self.print(f'====== {split} dataset evaluation result =======')
        self.print(self.eval_metric)
        self.print("")
        self.eval_metric.reset()
        return metric_dict

    def print(self, string):
        if not self.config.get('silent', False):
            if not self.trainer or self.trainer.is_global_zero:
                print(string)


class Model(MultiLabelModel):
    def __init__(self, config, word_dict=None, classes=None):
        super().__init__(config)
        self.save_hyperparameters()

        self.word_dict = word_dict
        self.classes = classes
        self.config.num_classes = len(self.classes)

        embed_vecs = self.word_dict.vectors
        self.network = getattr(networks, self.config.model_name)(
            self.config, embed_vecs)

        if config.init_weight is not None:
            init_weight = networks.get_init_weight_func(self.config)
            self.apply(init_weight)

    def shared_step(self, batch):
        target_labels = batch['label']
        outputs = self.network(batch['text'])
        pred_logits = outputs['logits']
        loss = F.binary_cross_entropy_with_logits(pred_logits, target_labels)
        return loss, pred_logits
