import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from libmultilabel.nn import data_utils
from libmultilabel.nn import networks
from libmultilabel.nn.model import Model
from libmultilabel.nn.nn_utils import init_model, init_trainer
from libmultilabel.utils import dump_log, init_device, set_seed


class TorchTrainer:
    """A wrapper for training neural network models with pytorch lightning trainer.

    Args:
        config (AttributeDict): Config of the experiment.
    """
    def __init__(
        self,
        config: dict
    ):
        # Run name
        self.run_name = '{}_{}_{}'.format(
            config.data_name,
            Path(config.config).stem if config.config else config.model_name,
            datetime.now().strftime('%Y%m%d%H%M%S'),
        )
        logging.info(f'Run name: {self.run_name}')
        self.checkpoint_dir = os.path.join(config.result_dir, self.run_name)
        self.log_path = os.path.join(self.checkpoint_dir, 'logs.json')

        # Set up seed & device
        set_seed(seed=config.seed)
        self.device = init_device(use_cpu=config.cpu)
        self.config = config

        # Load dataset
        self.datasets = data_utils.load_datasets(data_dir=config.data_dir,
                                                 train_path=config.train_path,
                                                 test_path=config.test_path,
                                                 val_path=config.val_path,
                                                 val_size=config.val_size,
                                                 is_eval=config.eval)
        self._setup_model(log_path=self.log_path, checkpoint_path=config.checkpoint_path)
        self.trainer = init_trainer(checkpoint_dir=config.checkpoint_path,
                                    epochs=config.epochs,
                                    patience=config.patience,
                                    val_metric=config.val_metric)

        # Dump config to log
        dump_log(self.log_path, config=config)

    def _setup_model(self, log_path=None, checkpoint_path=None):
        """Setup model from checkpoint if a checkpoint path is passed in or specified in the config.
        Otherwise, initialize model from scratch.

        Args:
            log_path (str, optional): Path to the log file. The log file contains the validation
                results for each epoch and the test results. If the `log_path` is None, no performance
                results will be logged. Defaults to None.
            checkpoint_path (str, optional): The checkpoint to warm-up with. Defaults to None.
        """
        if 'checkpoint_path' in self.config and self.config.checkpoint_path is not None:
            checkpoint_path = self.config.checkpoint_path

        if checkpoint_path is not None:
            logging.info(f'Loading model from `{checkpoint_path}`...')
            self.model = Model.load_from_checkpoint(checkpoint_path)
        else:
            logging.info('Initialize model from scratch.')
            word_dict = data_utils.load_or_build_text_dict(
                dataset=self.datasets['train'],
                vocab_file=self.config.vocab_file,
                min_vocab_freq=self.config.min_vocab_freq,
                embed_file=self.config.embed_file,
                silent=self.config.silent,
                normalize_embed=self.config.normalize_embed
            )
            classes = data_utils.load_or_build_label(
                self.datasets, self.config.label_file, self.config.silent)

            self.model = init_model(model_name=self.config.model_name,
                                    network_config=dict(self.config.network_config),
                                    classes=classes,
                                    word_dict=word_dict,
                                    init_weight=self.config.init_weight,
                                    log_path=log_path,
                                    learning_rate=self.config.learning_rate,
                                    optimizer=self.config.optimizer,
                                    momentum=self.config.momentum,
                                    weight_decay=self.config.weight_decay,
                                    metric_threshold=self.config.metric_threshold,
                                    monitor_metrics=self.config.monitor_metrics,
                                    silent=self.config.silent,
                                    save_k_predictions=self.config.save_k_predictions
                                   )

    def _get_dataset_loader(self, split, shuffle=False):
        """Get dataset loader.

        Args:
            split (str): One of 'train', 'test', or 'val'.
            shuffle (bool): Whether to shuffle training data before each epoch. Defaults to False.

        Returns:
            torch.utils.data.DataLoader: Dataloader for the train, test, or valid dataset.
        """
        return data_utils.get_dataset_loader(
            data=self.datasets[split],
            word_dict=self.model.word_dict,
            classes=self.model.classes,
            device=self.device,
            max_seq_length=self.config.max_seq_length,
            batch_size=self.config.batch_size if split == 'train' else self.config.eval_batch_size,
            shuffle=shuffle,
            data_workers=self.config.data_workers
        )

    def train(self):
        """Train model with pytorch lightning trainer. Set model to the best model after the training
        process is finished.
        """
        assert self.trainer is not None, "Please make sure the trainer is successfully initialized by `self._setup_trainer()`."
        train_loader = self._get_dataset_loader(split='train', shuffle=self.config.shuffle)
        val_loader = self._get_dataset_loader(split='val')
        self.trainer.fit(self.model, train_loader, val_loader)

        # Set model to current best model
        logging.info(f'Finished training. Load best model from {self.checkpoint_callback.best_model_path}')
        self._setup_model(checkpoint_path=self.checkpoint_callback.best_model_path)

    def test(self):
        """Test model with pytorch lightning trainer. Top-k predictions are saved
        if `save_k_predictions` > 0.
        """
        assert 'test' in self.datasets and self.trainer is not None
        test_loader = self._get_dataset_loader(split='test')
        self.trainer.test(self.model, test_dataloaders=test_loader)

        if self.config.save_k_predictions > 0:
            if not self.config.predict_out_path:
                predict_out_path = os.path.join(self.checkpoint_dir, 'predictions.txt')
            else:
                predict_out_path = self.config.predict_out_path
            self._save_predictions(test_loader, predict_out_path)

    def _save_predictions(self, dataloader, predict_out_path):
        """Save top k label results.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader for the test or valid dataset.
            predict_out_path (str): Path to the an output file holding top k label results.
                Defaults to None.
        """
        batch_predictions = self.trainer.predict(self.model, dataloaders=dataloader)
        pred_labels = np.vstack([batch['top_k_pred']
                                for batch in batch_predictions])
        pred_scores = np.vstack([batch['top_k_pred_scores']
                                for batch in batch_predictions])
        with open(predict_out_path, 'w') as fp:
            for pred_label, pred_score in zip(pred_labels, pred_scores):
                out_str = ' '.join([f'{self.model.classes[label]}:{score:.4}' for label, score in zip(
                    pred_label, pred_score)])
                fp.write(out_str+'\n')
        logging.info(f'Saved predictions to: {predict_out_path}')
