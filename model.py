import logging
import os
import pickle
import shutil

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import data_utils
import networks
from evaluate import evaluate
from utils import AverageMeter, Timer


class Model(object):
    """High level model that handles initializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, config, word_dict=None, classes=None, ckpt=None):
        self.config = config
        self.device = config.device
        self.start_epoch = 0

        if ckpt:
            self.config.run_name = ckpt['run_name']
            self.word_dict = ckpt['word_dict']
            self.classes = ckpt['classes']
            self.best_metric = ckpt['best_metric']
            self.start_epoch = ckpt['epoch']
        else:
            self.word_dict = word_dict
            self.classes = classes
            self.start_epoch = 0
            self.best_metric = 0

            # load embedding
            if os.path.exists(config.embed_file):
                logging.info(f'Load pretrained embedding from file: {config.embed_file}.')
                embedding_weights = data_utils.get_embedding_weights_from_file(self.word_dict, config.embed_file)
                self.word_dict.set_vectors(self.word_dict.stoi, embedding_weights,dim=embedding_weights.shape[1], unk_init=False)
            elif not config.embed_file.isdigit():
                logging.info(f'Load pretrained embedding from torchtext.')
                self.word_dict.load_vectors(config.embed_file)
            else:
                raise NotImplementedError
        self.config.num_classes = len(self.classes)

        embed_vecs = self.word_dict.vectors
        self.network = getattr(networks, config.model_name)(config, embed_vecs).to(self.device)
        self.init_optimizer()

        if ckpt:
            self.network.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
        elif config.init_weight is not None:
            init_weight = networks.get_init_weight_func(config)
            self.network.apply(init_weight)

    def init_optimizer(self, optimizer=None):
        """Initialize an optimizer for the free parameters of the network.
        Args:
            state_dict: network parameters
        """
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        optimizer_name = optimizer or self.config.optimizer
        if optimizer_name == 'sgd':
            self.optimizer = optim.SGD(parameters, self.config.learning_rate,
                                       momentum=self.config.momentum,
                                       weight_decay=self.config.weight_decay)
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(parameters, weight_decay=self.config.weight_decay, lr=self.config.learning_rate)

        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.config.optimizer)

        torch.nn.utils.clip_grad_value_(parameters, 0.5)

    def train(self, train_data, val_data):
        train_loader = data_utils.get_dataset_loader(
            self.config, train_data, self.word_dict, self.classes, train=True)
        val_loader = data_utils.get_dataset_loader(
            self.config, val_data, self.word_dict, self.classes, train=False)

        logging.info('Start training')
        try:
            epoch = self.start_epoch + 1
            patience = self.config.patience
            while epoch <= self.config.epochs:
                if patience == 0:
                    logging.info('Reach training patience. Stopping...')
                    break

                logging.info(f'============= Starting epoch {epoch} =============')

                self.train_epoch(train_loader)

                logging.info('Start predicting a validation set')
                val_metrics = evaluate(self.config, self, val_loader)

                if val_metrics[self.config.val_metric] > self.best_metric:
                    self.best_metric = val_metrics[self.config.val_metric]
                    self.save(epoch, is_best=True)
                    patience = self.config.patience
                else:
                    logging.info(f'Performance does not increase, training will stop in {patience} epochs')
                    self.save(epoch)
                    patience -= 1

                epoch += 1
        except KeyboardInterrupt:
            logging.info('Training process terminated')

    def train_epoch(self, data_loader):
        """Run through one epoch of model training with the provided data loader."""

        train_loss = AverageMeter()
        epoch_time = Timer()
        progress_bar = tqdm(data_loader)

        for idx, batch in enumerate(progress_bar):
            loss, batch_label_scores = self.train_step(batch)
            train_loss.update(loss)
            progress_bar.set_postfix(loss=train_loss.avg)

        logging.info(f'Epoch done. Time for epoch = {epoch_time.time():.2f} (s)')
        logging.info(f'Epoch loss: {train_loss.avg}')

    def train_step(self, inputs):
        """Forward a batch of examples; stop the optimizer to update weights.
        """
        # Train mode
        self.network.train()

        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device, non_blocking=True)

        # Run forward
        target_labels = inputs['label']
        outputs = self.network(inputs['text'])
        pred_logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        loss = F.binary_cross_entropy_with_logits(pred_logits, target_labels)
        batch_label_scores = torch.sigmoid(pred_logits)

        # Update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), batch_label_scores

    def predict(self, inputs):
        """Forward a batch of examples only to get predictions.

        Args:
            inputs: the batch of inputs
            top_n: Number of predictions to return per batch element (default: all predictions).
        Output:
            {
                'scores': predicted score tensor,
                'logits': predicted logit tensor,
                'outputs': full predict output,
                'top_results': top results from extract_top_n_predictions.
            }
        """
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device, non_blocking=True)

        # Run forward
        with torch.no_grad():
            outputs = self.network(inputs['text'])
            logits = outputs['logits']
            batch_label_scores = torch.sigmoid(logits)

        return {
            'scores': batch_label_scores,
            'logits': logits,
            'outputs': outputs,
        }

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
            logging.info(f"Save best model ({self.config.val_metric}: {self.best_metric}): {best_ckpt_path}")
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
