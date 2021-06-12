import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim

from . import networks
from .metrics import MultiLabelMetrics


class Model(pl.LightningModule):
    """High level model that handles initializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, config, word_dict=None, classes=None, ckpt=None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.word_dict = word_dict
        self.classes = classes
        self.config.num_classes = len(self.classes)

        embed_vecs = self.word_dict.vectors
        self.network = getattr(networks, config.model_name)(
            config, embed_vecs).to(config.device)

        init_weight = networks.get_init_weight_func(config)
        self.apply(init_weight)

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
            optimizer = optim.Adam(
                parameters, weight_decay=self.config.weight_decay, lr=self.config.learning_rate)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.config.optimizer)

        torch.nn.utils.clip_grad_value_(parameters, 0.5)

        return optimizer

    def shared_step(self, batch):
        target_labels = batch['labels'] = batch['label']
        outputs = self.network(batch['text'])
        pred_logits = outputs['logits']
        loss = F.binary_cross_entropy_with_logits(pred_logits, target_labels)
        return loss, pred_logits

    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred_logits = self.shared_step(batch)
        return {'loss': loss.item(),
                'pred_scores': torch.sigmoid(pred_logits).detach().cpu().numpy(),
                'target': batch['labels'].detach().cpu().numpy()}

    def validation_epoch_end(self, step_outputs):
        eval_metric = MultiLabelMetrics(
            monitor_metrics=self.config.monitor_metrics)
        outputs_dict = {k: [outputs[k] for outputs in step_outputs]
                        for k in step_outputs[0]}  # collect batch outputs to lists
        pred_scores = np.vstack(outputs_dict['pred_scores'])
        target = np.vstack(outputs_dict['target'])
        eval_metric.add_values(y_pred=pred_scores, y_true=target)
        eval_metric.eval()
        self.log_dict(eval_metric.get_metric_dict())
        self.print(eval_metric)
        return eval_metric

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, step_outputs):
        self.print('====== Test dataset evaluation result =======')
        eval_metric = self.validation_epoch_end(step_outputs)
        self.test_results = eval_metric
        return eval_metric

    def print(self, string):
        if not self.config.silent:
            print(string)
