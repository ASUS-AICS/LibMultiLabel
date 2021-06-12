import logging
import os
import shutil

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from . import data_utils, networks
from .evaluate import MultiLabelMetrics, evaluate
from .utils import AverageMeter, Timer, dump_log


class Model(pl.LightningModule):
    """High level model that handles initializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, config, word_dict=None, classes=None, ckpt=None):
        super().__init__()
        self.config = config
        self.patience = config.patience

        if ckpt:
            self.config.run_name = ckpt['run_name']
            self.word_dict = ckpt['word_dict']
            self.classes = ckpt['classes']
            self.best_metric = ckpt['best_metric']
            self.epoch = ckpt['epoch']
        else:
            self.word_dict = word_dict
            self.classes = classes
            self.epoch = 0
            self.best_metric = 0

        self.config.num_classes = len(self.classes)

        embed_vecs = self.word_dict.vectors
        self.network = getattr(networks, config.model_name)(
            config, embed_vecs).to(config.device)

        if ckpt:
            self.network.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
        elif config.init_weight is not None:
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

    def run_batch(self, batch):
        target_labels = batch['labels'] = batch['label']
        outputs = self.network(batch['text'])
        pred_logits = outputs['logits']
        loss = F.binary_cross_entropy_with_logits(pred_logits, target_labels)
        return loss, pred_logits

    def training_step(self, batch, batch_idx):
        # Run forward
        loss, _ = self.run_batch(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred_logits = self.run_batch(batch)
        return {'loss': loss.item(),
                'pred_scores': torch.sigmoid(pred_logits).detach().cpu().numpy(),
                'target': batch['labels'].detach().cpu().numpy()}

    def validation_epoch_end(self, validation_step_outputs):
        eval_metric = MultiLabelMetrics(
            monitor_metrics=self.config.monitor_metrics)
        validation_step_outputs_dict = {k: [outputs[k] for outputs in validation_step_outputs]
                                        for k in validation_step_outputs[0]}  # collect batch outputs to lists
        pred_scores = np.vstack(validation_step_outputs_dict['pred_scores'])
        target = np.vstack(validation_step_outputs_dict['target'])
        eval_metric.add_values(y_pred=pred_scores, y_true=target)
        eval_metric.eval()
        self.log_dict(eval_metric.get_metric_dict())
        print(eval_metric)

    def save(self, epoch, is_best=False):
        self.network.eval()
        ckpt = {
            'epoch': epoch,
            'run_name': self.config.run_name,
            'state_dict': self.network.state_dict(),
            'word_dict': self.word_dict,
            'classes': self.classes,
            'optimizer': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
        }
        ckpt_path = os.path.join(self.config.result_dir,
                                 self.config.run_name, 'model_last.pt')
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        logging.info(f"Save current  model: {ckpt_path}")
        torch.save(ckpt, ckpt_path)
        if is_best:
            best_ckpt_path = ckpt_path.replace('last', 'best')
            logging.info(
                f"Save best model ({self.config.val_metric}: {self.best_metric}): {best_ckpt_path}")
            shutil.copyfile(ckpt_path, best_ckpt_path)
        self.network.train()

    @staticmethod
    def load(config, ckpt_path):
        ckpt = torch.load(ckpt_path)
        return Model(config, ckpt=ckpt)

    def load_best(self):
        best_ckpt_path = os.path.join(self.config.result_dir,
                                      self.config.run_name, 'model_best.pt')
        best_model = self.load(self.config, best_ckpt_path)
        self.network = best_model.network
