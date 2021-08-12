import logging
import os
from typing import Callable, Dict
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from libmultilabel import data_utils
from libmultilabel.model import Model
from libmultilabel.utils import dump_log, init_device, set_seed


class TorchTrainer:
    def __init__(
        self,
        config: Dict,
        dataset_fn: Callable = None,
        model_fn: Callable = None,
        trainer_fn: Callable = None,
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

        # Setup functions
        if dataset_fn is None:
            dataset_fn = data_utils.load_datasets

        if model_fn is None:
            model_fn = self._setup_model

        if trainer_fn is None:
            trainer_fn = self._setup_trainer

        # Load dataset
        self.datasets = dataset_fn(data_dir=config.data_dir,
                                    train_path=config.train_path,
                                    test_path=config.test_path,
                                    val_path=config.val_path,
                                    val_size=config.val_size,
                                    is_eval=config.eval)

        model_fn(config, log_path=self.log_path,
                 checkpoint_path=config.checkpoint_path)
        trainer_fn(config)

        # Dump config to log
        dump_log(self.log_path, config=config)

    def _setup_model(self, config=None, log_path=None, checkpoint_path=None):
        assert config is not None or checkpoint_path is not None, \
            "Please specify either config or checkpoint_path to initialize the model."

        if checkpoint_path is not None:
            logging.info(f'Loading model from `{checkpoint_path}`...')
            self.model = Model.load_from_checkpoint(checkpoint_path)
        else:
            logging.info('Initialize model from scratch.')
            word_dict = data_utils.load_or_build_text_dict(
                dataset=self.datasets['train'],
                vocab_file=config.vocab_file,
                min_vocab_freq=config.min_vocab_freq,
                embed_file=config.embed_file,
                silent=config.silent,
                normalize_embed=config.normalize_embed
            )
            classes = data_utils.load_or_build_label(
                self.datasets, config.label_file, config.silent)

            self.model = Model(
                device=self.device,
                classes=classes,
                word_dict=word_dict,
                log_path=log_path,
                **dict(config)
            )

    def _setup_trainer(self, config):
        self.checkpoint_callback = ModelCheckpoint(dirpath=self.checkpoint_dir,
                                              filename='best_model',
                                              save_last=True, save_top_k=1,
                                              monitor=config.val_metric, mode='max')
        self.earlystopping_callback = EarlyStopping(patience=config.patience,
                                               monitor=config.val_metric, mode='max')
        self.trainer = pl.Trainer(logger=False, num_sanity_val_steps=0,
                                  gpus=0 if config.cpu else 1,
                                  progress_bar_refresh_rate=0 if config.silent else 1,
                                  max_epochs=config.epochs,
                                  callbacks=[self.checkpoint_callback, self.earlystopping_callback])

    def _get_dataset_loader(self, split, dataset_config=None, shuffle=False):
        if dataset_config is None:
            # If dataset config is None, use global config instead.
            dataset_config = self.config

        return data_utils.get_dataset_loader(
            data=self.datasets[split],
            word_dict=self.model.word_dict,
            classes=self.model.classes,
            device=self.device,
            max_seq_length=dataset_config.max_seq_length,
            batch_size=dataset_config.batch_size if split=='train' else dataset_config.eval_batch_size,
            shuffle=shuffle,
            data_workers=dataset_config.data_workers
        )

    def train(self, dataset_config=None, shuffle=False):
        assert self.trainer is not None
        train_loader = self._get_dataset_loader(split='train', dataset_config=dataset_config, shuffle=shuffle)
        val_loader = self._get_dataset_loader(split='val', dataset_config=dataset_config)
        self.trainer.fit(self.model, train_loader, val_loader)

        # Set model to current best model
        logging.info(f'Finished training. Load best model from {self.checkpoint_callback.best_model_path}')
        self._setup_model(checkpoint_path=self.checkpoint_callback.best_model_path)

    def test(self, dataset_config=None):
        assert 'test' in self.datasets and self.trainer is not None
        test_loader = self._get_dataset_loader(
            split='test', dataset_config=dataset_config)
        self.trainer.test(self.model, test_dataloaders=test_loader)
