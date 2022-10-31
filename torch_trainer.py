import logging
import os

import numpy as np
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from transformers import AutoTokenizer

from libmultilabel.nn import data_utils
from libmultilabel.nn.model import Model
from libmultilabel.nn.metrics import get_metrics
from libmultilabel.nn.nn_utils import init_device, init_model, init_trainer, set_seed
from libmultilabel.common_utils import dump_log


class TorchTrainer:
    """A wrapper for training neural network models with pytorch lightning trainer.

    Args:
        config (AttributeDict): Config of the experiment.
        datasets (dict, optional): Datasets for training, validation, and test. Defaults to None.
        classes(list, optional): List of class names.
        word_dict(torchtext.vocab.Vocab, optional): A vocab object which maps tokens to indices.
        search_params (bool): Enable pytorch-lightning trainer to report the results to ray tune
            on validation end during hyperparameter search. Defaults to False.
        save_checkpoints (bool): Whether to save the last and the best checkpoint or not. Defaults to True.
    """
    def __init__(
        self,
        config: dict,
        datasets: dict = None,
        classes: list = None,
        word_dict: dict = None,
        search_params: bool = False,
        save_checkpoints: bool = True
    ):
        self.run_name = config.run_name
        self.checkpoint_dir = config.checkpoint_dir
        self.log_path = config.log_path
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Set up seed & device
        set_seed(seed=config.seed)
        self.device = init_device(use_cpu=config.cpu)
        self.config = config

        # Load pretrained tokenizer for dataset loader
        self.tokenizer = None
        tokenize_text = 'lm_weight' not in config.network_config
        if not tokenize_text:
            self.tokenizer = AutoTokenizer.from_pretrained(config.network_config['lm_weight'], use_fast=True)
        # Load dataset
        if datasets is None:
            self.datasets = data_utils.load_datasets(
                train_path=config.train_path,
                test_path=config.test_path,
                val_path=config.val_path,
                val_size=config.val_size,
                merge_train_val=config.merge_train_val,
                tokenize_text=tokenize_text
            )
        else:
            self.datasets = datasets

        self._setup_model(classes=classes,
                          word_dict=word_dict,
                          log_path=self.log_path,
                          checkpoint_path=config.checkpoint_path)
        self.trainer = init_trainer(checkpoint_dir=self.checkpoint_dir,
                                    epochs=config.epochs,
                                    patience=config.patience,
                                    val_metric=config.val_metric,
                                    silent=config.silent,
                                    use_cpu=config.cpu,
                                    limit_train_batches=config.limit_train_batches,
                                    limit_val_batches=config.limit_val_batches,
                                    limit_test_batches=config.limit_test_batches,
                                    search_params=search_params,
                                    save_checkpoints=save_checkpoints)
        callbacks = [callback for callback in self.trainer.callbacks if isinstance(callback, ModelCheckpoint)]
        self.checkpoint_callback = callbacks[0] if callbacks else None

        # Dump config to log
        dump_log(self.log_path, config=config)

    def _setup_model(
        self,
        classes: list = None,
        word_dict: dict = None,
        log_path: str = None,
        checkpoint_path: str = None
    ):
        """Setup model from checkpoint if a checkpoint path is passed in or specified in the config.
        Otherwise, initialize model from scratch.

        Args:
            classes(list): List of class names.
            word_dict(torchtext.vocab.Vocab): A vocab object which maps tokens to indices.
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
            self.label_desc_idx = None
            if self.config.label_file:
                classes, _, _ = data_utils.load_or_build_label(
                    self.datasets, self.config.label_file, self.config.include_test_labels, self.config.fewshot_threshold)
                # TBD: for zero-shot model
                self.model.classes = classes
                self.label_desc_idx = data_utils.load_label_description(self.config.label_file, self.model.classes, self.model.word_dict, embedding=False)
        else:
            logging.info('Initialize model from scratch.')
            if self.config.embed_file is not None:
                logging.info('Load word dictionary ')
                word_dict = data_utils.load_or_build_text_dict(
                    dataset=self.datasets['train'],
                    vocab_file=self.config.vocab_file,
                    min_vocab_freq=self.config.min_vocab_freq,
                    embed_file=self.config.embed_file,
                    silent=self.config.silent,
                    normalize_embed=self.config.normalize_embed,
                    embed_cache_dir=self.config.embed_cache_dir
                )
            if not classes:
                classes, _, _ = data_utils.load_or_build_label(
                    self.datasets, self.config.label_file, self.config.include_test_labels, self.config.fewshot_threshold)

            self.label_desc_idx = None
            if self.config.label_file:
                self.label_desc_idx = data_utils.load_label_description(self.config.label_file, classes, word_dict, embedding=False)

            if self.config.val_metric not in self.config.monitor_metrics:
                logging.warn(
                    f'{self.config.val_metric} is not in `monitor_metrics`. Add {self.config.val_metric} to `monitor_metrics`.')
                self.config.monitor_metrics += [self.config.val_metric]

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
            data_workers=self.config.data_workers,
            tokenizer=self.tokenizer,
            label_desc_idx=self.label_desc_idx
        )

    def train(self):
        """Train model with pytorch lightning trainer. Set model to the best model after the training
        process is finished.
        """
        assert self.trainer is not None, "Please make sure the trainer is successfully initialized by `self._setup_trainer()`."
        train_loader = self._get_dataset_loader(split='train', shuffle=self.config.shuffle)

        if 'val' not in self.datasets:
            logging.info('No validation dataset is provided. Train without vaildation.')
            self.trainer.fit(self.model, train_loader)
        else:
            val_loader = self._get_dataset_loader(split='val')
            self.trainer.fit(self.model, train_loader, val_loader)

        # Set model to the best model. If the validation process is skipped during
        # training (i.e., val_size=0), the model is set to the last model.
        model_path = self.checkpoint_callback.best_model_path or self.checkpoint_callback.last_model_path
        if model_path:
            logging.info(f'Finished training. Load best model from {model_path}.')
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
        test_loader = self._get_dataset_loader(split=split)
        ###############################
        # from set_value import set_value
        # self.model = set_value(self.model)
        ###############################
        metric_dict = self.trainer.test(self.model, test_dataloaders=test_loader)[0]


        if self.config.save_k_predictions > 0:
            self.model.predict_top_k = self.config.save_k_predictions
            self._save_predictions(test_loader, self.config.predict_out_path)

        if self.config.fewshot_threshold > 0:
            classes, fewshot_start, zeroshot_start = data_utils.load_or_build_label(
                self.datasets, self.config.label_file, self.config.include_test_labels, self.config.fewshot_threshold)
            self.eval_zero_few(test_loader, classes, fewshot_start, zeroshot_start)

        return metric_dict

    def eval_zero_few(self, dataloader, classes, fewshot_start, zeroshot_start):
        model_eval_metric = self.model.eval_metric

        # Evaluate few-shot labels
        logging.info(f'=== Few-shot Labels ===')
        self.model.eval_label_set = (fewshot_start, zeroshot_start)
        n_eval_classes = zeroshot_start - fewshot_start
        self.model.eval_metric = get_metrics(self.config.metric_threshold, self.config.fewshot_monitor_metrics, n_eval_classes)
        metric_dict = self.trainer.test(self.model, test_dataloaders=dataloader)

        # Evaluate zero-shot labels
        logging.info(f'=== Zero-shot Labels ===')
        self.model.eval_label_set = (zeroshot_start, len(classes))
        n_eval_classes = len(classes) - zeroshot_start
        self.model.eval_metric = get_metrics(self.config.metric_threshold, self.config.fewshot_monitor_metrics, n_eval_classes)
        metric_dict = self.trainer.test(self.model, test_dataloaders=dataloader)

        self.model.eval_label_set = None
        self.model.eval_metric = model_eval_metric
        return

    def _save_predictions(self, dataloader, predict_out_path):
        """Save top k label results.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader for the test or valid dataset.
            predict_out_path (str): Path to the an output file holding top k label results.
        """
        batch_predictions = self.trainer.predict(self.model, dataloaders=dataloader)
        ####################
        # pred_scores = np.vstack([batch['pred_scores']
                                # for batch in batch_predictions])
        # np.save(predict_out_path, pred_scores)
        # logging.info(f'Saved predictions to: {predict_out_path}')
        # return
        ############################
        pred_labels = np.vstack([batch['top_k_pred']
                                for batch in batch_predictions])
        pred_scores = np.vstack([batch['top_k_pred_scores']
                                for batch in batch_predictions])
        with open(predict_out_path, 'w') as fp:
            for pred_label, pred_score in zip(pred_labels, pred_scores):
                out_str = ' '.join([f'{self.model.classes[label]}:{score:.4}' for label, score in zip(
                    pred_label, pred_score)])
                fp.write(out_str+'\n')
        pred_labels = np.vectorize(lambda x: self.model.classes[x])(pred_labels)
        np.savez(predict_out_path.replace('.txt', '.npz'), pred_labels=pred_labels, pred_scores=pred_scores)
        logging.info(f'Saved predictions to: {predict_out_path}')
