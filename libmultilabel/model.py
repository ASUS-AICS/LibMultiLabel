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

    def validation_epoch_end(self, step_outputs):
        eval_metric = MultiLabelMetrics(self.config)
        for step_output in step_outputs:
            eval_metric.add_values(y_pred=step_output['pred_scores'],
                                   y_true=step_output['target'])
        metric_dict = eval_metric.get_metric_dict()
        self.log_dict(metric_dict)
        self.print(eval_metric)
        dump_log(config=self.config, metrics=metric_dict, split='val')
        return eval_metric

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, step_outputs):
        self.print('====== Test dataset evaluation result =======')
        eval_metric = self.validation_epoch_end(step_outputs)
        self.test_results = eval_metric
        return eval_metric

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
            self.config, embed_vecs).to(self.config.device)

        if config.init_weight is not None:
            init_weight = networks.get_init_weight_func(self.config)
            self.apply(init_weight)

    def shared_step(self, batch):
        target_labels = batch['label']
        outputs = self.network(batch['text'])
        pred_logits = outputs['logits']
        loss = F.binary_cross_entropy_with_logits(pred_logits, target_labels)
        return loss, pred_logits
