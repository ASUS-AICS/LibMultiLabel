import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch
import transformers

from libmultilabel.nn import data_utils
from libmultilabel.nn import networks
from libmultilabel.nn.model import Model
from libmultilabel.nn.networks import bert_dataset, token_dataset
from libmultilabel.nn.nn_utils import init_device, init_model, init_trainer, set_seed
from libmultilabel.common_utils import AttributeDict, dump_log


class TorchTrainer:
    """A wrapper for training neural network models with pytorch lightning trainer.

    Args:
        config (AttributeDict): Config of the experiment.
        datasets (dict, optional): Datasets for training, validation, and test. Defaults to None.
        embed_vecs (torch.Tensor, optional): The pre-trained word vectors of shape (vocab_size, embed_dim).
        search_params (bool, optional): Enable pytorch-lightning trainer to report the results to ray tune
            on validation end during hyperparameter search. Defaults to False.
        save_checkpoints (bool, optional): Whether to save the last and the best checkpoint or not.
            Defaults to True.
    """

    def __init__(
        self,
        config: AttributeDict,
        datasets: 'Optional[dict[str, pd.DataFrame]]' = None,
        search_params: bool = False,
    ):
        self.run_name = config.run_name
        self.checkpoint_dir = config.checkpoint_dir
        self.log_path = config.log_path
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Set up seed & device
        set_seed(seed=config.seed)
        self.device = init_device(use_cpu=config.cpu)
        self.config = config

        if datasets is None:
            self.datasets = data_utils.load_datasets(
                training_file=config.training_file,
                test_file=config.test_file,
                val_file=config.val_file,
                val_size=config.val_size,
                merge_train_val=config.merge_train_val,
                remove_no_label_data=config.remove_no_label_data
            )
        else:
            self.datasets = datasets

        self._setup_model(
            log_path=self.log_path,
            checkpoint_path=config.checkpoint_path
        )
        self.trainer = init_trainer(
            checkpoint_dir=self.checkpoint_dir,
            epochs=config.epochs,
            patience=config.patience,
            val_metric=config.val_metric,
            silent=config.silent,
            use_cpu=config.cpu,
            limit_train_batches=config.limit_train_batches,
            limit_val_batches=config.limit_val_batches,
            limit_test_batches=config.limit_test_batches,
            search_params=search_params,
            save_checkpoints=True
        )
        callbacks = [callback for callback in self.trainer.callbacks if isinstance(
            callback, ModelCheckpoint)]
        self.checkpoint_callback = callbacks[0] if callbacks else None

        # Dump config to log
        dump_log(self.log_path, config=config)

    def _setup_model(
        self,
        log_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ):
        """Setup model from checkpoint if a checkpoint path is passed in or specified in the config.
        Otherwise, initialize model from scratch.

        Args:
            embed_vecs (torch.Tensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
            log_path (str): Path to the log file. The log file contains the validation
                results for each epoch and the test results. If the `log_path` is None, no performance
                results will be logged.
            checkpoint_path (str): The checkpoint to warm-up with.
        """
        if 'checkpoint_path' in self.config and self.config.checkpoint_path is not None:
            checkpoint_path = self.config.checkpoint_path

        if checkpoint_path is not None:
            logging.info(f'Loading model from `{checkpoint_path}`...')
            self.model = Model.load_from_checkpoint(checkpoint_path)
        else:
            logging.info('Initialize model from scratch.')
            classes = data_utils.load_or_build_label(
                self.datasets, self.config.label_file, self.config.include_test_labels)
            self.classes = classes

            if self.config.model_name in {'BERT', 'BERTAttention'}:
                self._initialize_transformer()
            else: # TODO: write explicit list
                self._initialize_rnn()

            if self.config.val_metric not in self.config.monitor_metrics:
                logging.warn(
                    f'{self.config.val_metric} is not in `monitor_metrics`. Add {self.config.val_metric} to `monitor_metrics`.')
                self.config.monitor_metrics += [self.config.val_metric]

            self.model = init_model(
                model_name=self.config.model_name,
                network_config=dict(self.config.network_config),
                classes=classes,
                word_dict=vocab,
                embed_vecs=embed_vecs,
                init_weight=self.config.init_weight,
                log_path=log_path,
                learning_rate=self.config.learning_rate,
                optimizer=self.config.optimizer,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                metric_threshold=self.config.metric_threshold,
                monitor_metrics=self.config.monitor_metrics,
                multiclass=self.config.multiclass,
                loss_function=self.config.loss_function,
                silent=self.config.silent,
                save_k_predictions=self.config.save_k_predictions
            )

    def _initialize_transformer(self):
        """Initializes self.network and self.dataloaders for transformers.
        """
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.network_config['lm_weight'], use_fase=True)
        def make_dataloader(split):
            dataset = bert_dataset.BertDataset(
                self.datasets[split],
                self.classes,
                tokenizer,
                self.config.max_seq_length,
                self.config.add_special_tokens,
            )

            shuffle = split == 'train' and self.config.shuffle
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=self.config.data_workers,
                collate_fn=bert_dataset.collate_fn,
                pin_memory='cuda' in self.device.type,
            )

        self.dataloaders = {
            'train': make_dataloader('train'),
            'val': make_dataloader('val'),
            'test': make_dataloader('testin'),
        }

        # TODO: write explicit list
        self.network = getattr(networks, self.network.model_name)(
            num_classes=len(self.classes),
            **self.config.network_config,
        )

    def _initialize_rnn(self):
        """Initializes self.network and self.dataloaders for RNNs. Modifies self.datasets.
        """
        self.datasets['train']['text'] = self.datasets['train']['text'].map(token_dataset.tokenize)
        self.datasets['val']['text'] = self.datasets['val']['text'].map(token_dataset.tokenize)
        self.datasets['test']['text'] = self.datasets['test']['text'].map(token_dataset.tokenize)

        if self.config.vocab_file:
            vocab = token_dataset.load_vocabulary(self.config.vocab_file)
        else:
            vocab = token_dataset.build_vocabulary(
                self.datasets['train'], self.config.min_vocab_freq)
        embed_vecs = token_dataset.load_embedding_weights(
            vocab, self.config.embed_file, self.config.embed_cache_dir, self.config.normalize)

        def make_dataloader(split):
            dataset = token_dataset.TokenDataset(
                self.datasets[split],
                vocab,
                self.classes,
                self.config.max_seq_length,
            )

            shuffle = split == 'train' and self.config.shuffle
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=self.config.data_workers,
                collate_fn=token_dataset.collate_fn,
                pin_memory='cuda' in self.device.type,
            )

        self.dataloaders = {
            'train': make_dataloader('train'),
            'val': make_dataloader('val'),
            'test': make_dataloader('testin'),
        }

        # TODO: write explicit list
        self.network = getattr(networks, self.config.model_name)(
            embed_vecs=embed_vecs,
            num_classes=len(self.classes),
            **self.config.network_config,
        )

    def train(self):
        """Train model with pytorch lightning trainer. Set model to the best model after the training
        process is finished.
        """
        assert self.trainer is not None, "Please make sure the trainer is successfully initialized by `self._setup_trainer()`."

        if 'val' not in self.datasets:
            logging.info(
                'No validation dataset is provided. Train without vaildation.')
            self.trainer.fit(self.model, self.dataloaders['train'])
        else:
            self.trainer.fit(
                self.model, self.dataloaders['train'], self.dataloaders['val'])

        # Set model to the best model. If the validation process is skipped during
        # training (i.e., val_size=0), the model is set to the last model.
        model_path = self.checkpoint_callback.best_model_path or self.checkpoint_callback.last_model_path
        if model_path:
            logging.info(
                f'Finished training. Load best model from {model_path}.')
            self._setup_model(checkpoint_path=model_path)
        else:
            logging.info('No model is saved during training. \
                If you want to save the best and the last model, please set `save_checkpoints` to True.')

    def test(self, split='test'):
        """Test model with pytorch lightning trainer. Top-k predictions are saved
        if `save_k_predictions` > 0.

        Args:
            split (str, optional): One of 'train', 'test', or 'val'. Defaults to 'test'.

        Returns:
            dict: Scores for all metrics in the dictionary format.
        """
        assert 'test' in self.datasets and self.trainer is not None

        logging.info(f'Testing on {split} set.')
        metric_dict = self.trainer.test(
            self.model, dataloaders=self.dataloaders[split])[0]

        if self.config.save_k_predictions > 0:
            self._save_predictions(
                self.dataloaders[split], self.config.predict_out_path)

        return metric_dict

    def _save_predictions(self, dataloader, predict_out_path):
        """Save top k label results.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader for the test or valid dataset.
            predict_out_path (str): Path to the an output file holding top k label results.
        """
        batch_predictions = self.trainer.predict(
            self.model, dataloaders=dataloader)
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
