import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from libmultilabel import data_utils
from libmultilabel import networks
from libmultilabel.model import Model
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
        self._setup_trainer()

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

            network = getattr(networks, self.config.model_name)(
                embed_vecs=word_dict.vectors,
                num_classes=len(classes),
                **dict(self.config.network_config)
            )
            if self.config.init_weight is not None:
                init_weight = networks.get_init_weight_func(
                    init_weight=self.config.init_weight)
                network.apply(init_weight)

            self.model = Model(
                classes=classes,
                word_dict=word_dict,
                network=network,
                log_path=log_path,
                **dict(self.config)
            )

    def _setup_trainer(self):
        """Setup torch trainer and callbacks."""
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir, filename='best_model', save_last=True,
            save_top_k=1, monitor=self.config.val_metric, mode='max')
        self.earlystopping_callback = EarlyStopping(
            patience=self.config.patience, monitor=self.config.val_metric, mode='max')
        self.trainer = pl.Trainer(logger=False, num_sanity_val_steps=0,
                                  gpus=0 if self.config.cpu else 1,
                                  progress_bar_refresh_rate=0 if self.config.silent else 1,
                                  max_epochs=self.config.epochs,
                                  callbacks=[self.checkpoint_callback, self.earlystopping_callback])

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

    def train(self, shuffle=False):
        """Train model with pytorch lightning trainer. Set model to the best model after the training
        process is finished.

        Args:
            shuffle (bool): Whether to shuffle training data before each epoch. Defaults to False.
        """
        assert self.trainer is not None, "Please make sure the trainer is successfully initialized by `self._setup_trainer()`."
        train_loader = self._get_dataset_loader(split='train', shuffle=shuffle)
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
