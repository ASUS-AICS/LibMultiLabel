from abc import abstractmethod
from argparse import Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.utilities.parsing import AttributeDict

from . import networks
from .metrics import MultiLabelMetrics
from .utils import dump_log


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
        silent=False, # TODO discuss if we need silent
        **kwargs
    ):
        super().__init__()
        # optimizers
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        # metrics
        self.metric_threshold = metric_threshold
        self.monitor_metrics = monitor_metrics
        # dump log
        self.log_path = log_path
        self.silent = silent

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
        loss, pred_logits = self.shared_step(batch)
        return {'loss': loss.item(),
                'pred_scores': torch.sigmoid(pred_logits).detach().cpu().numpy(),
                'target': batch['label'].detach().cpu().numpy()}

    def validation_epoch_end(self, step_outputs):
        eval_metric = self.evaluate(step_outputs, 'val')
        return eval_metric

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, step_outputs):
        eval_metric = self.evaluate(step_outputs, 'test')
        self.test_results = eval_metric
        return eval_metric

    def evaluate(self, step_outputs, split):
        eval_metric = MultiLabelMetrics(self.metric_threshold, self.monitor_metrics)
        for step_output in step_outputs:
            eval_metric.add_values(y_pred=step_output['pred_scores'],
                                   y_true=step_output['target'])
        metric_dict = eval_metric.get_metric_dict()
        self.log_dict(metric_dict)
        dump_log(metrics=metric_dict, split=split, log_path=self.log_path)

        self.print(f'\n====== {split.upper()} dataset evaluation result =======')
        self.print(eval_metric)
        return eval_metric

    def print(self, string):
        if not self.silent:
            if not self.trainer or self.trainer.is_global_zero:
                print(string)


class Model(MultiLabelModel):
    def __init__(
        self,
        device,
        model_name,
        classes=None,
        word_dict=None,
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
