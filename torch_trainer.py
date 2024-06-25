import logging
import os

import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoTokenizer

from libmultilabel.common_utils import dump_log, is_multiclass_dataset
from libmultilabel.nn import data_utils
from libmultilabel.nn.model import Model
from libmultilabel.nn.nn_utils import init_device, init_model, init_trainer, set_seed
from libmultilabel.nn.attentionxml import PLTTrainer


class TorchTrainer:
    """A wrapper for training neural network models with pytorch lightning trainer.

    Args:
        config (AttributeDict): Config of the experiment.
        datasets (dict, optional): Datasets for training, validation, and test. Defaults to None.
        classes(list, optional): List of class names.
        word_dict(torchtext.vocab.Vocab, optional): A vocab object which maps tokens to indices.
        embed_vecs (torch.Tensor, optional): The pre-trained word vectors of shape (vocab_size, embed_dim).
        save_checkpoints (bool, optional): Whether to save the last and the best checkpoint or not.
            Defaults to True.
    """

    def __init__(
        self,
        config: dict,
        datasets: dict = None,
        classes: list = None,
        word_dict: dict = None,
        embed_vecs=None,
        save_checkpoints: bool = True,
        is_tune_mode: bool = False,
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
        tokenize_text = "lm_weight" not in config.network_config
        if not tokenize_text:
            self.tokenizer = AutoTokenizer.from_pretrained(config.network_config["lm_weight"], use_fast=True)
        # Load dataset
        if datasets is None:
            self.datasets = data_utils.load_datasets(
                training_data=config.training_file,
                test_data=config.test_file,
                val_data=config.val_file,
                val_size=config.val_size,
                merge_train_val=config.merge_train_val,
                tokenize_text=tokenize_text,
                remove_no_label_data=config.remove_no_label_data,
            )
        else:
            self.datasets = datasets

        self.config.multiclass = is_multiclass_dataset(self.datasets["train"] + self.datasets.get("val", list()))

        if self.config.model_name.lower() == "attentionxml":
            # Note that AttentionXML produces two models. checkpoint_path directs to model_1
            if config.checkpoint_path is None:
                if self.config.embed_file is not None:
                    logging.info("Load word dictionary ")
                    word_dict, embed_vecs = data_utils.load_or_build_text_dict(
                        dataset=self.datasets["train"] + self.datasets["val"],
                        vocab_file=config.vocab_file,
                        min_vocab_freq=config.min_vocab_freq,
                        embed_file=config.embed_file,
                        silent=config.silent,
                        normalize_embed=config.normalize_embed,
                        embed_cache_dir=config.embed_cache_dir,
                    )

                if not classes:
                    classes = data_utils.load_or_build_label(
                        self.datasets, self.config.label_file, self.config.include_test_labels
                    )

                if self.config.early_stopping_metric not in self.config.monitor_metrics:
                    logging.warning(
                        f"{self.config.early_stopping_metric} is not in `monitor_metrics`. "
                        f"Add {self.config.early_stopping_metric} to `monitor_metrics`."
                    )
                    self.config.monitor_metrics += [self.config.early_stopping_metric]

                if self.config.val_metric not in self.config.monitor_metrics:
                    logging.warn(
                        f"{self.config.val_metric} is not in `monitor_metrics`. "
                        f"Add {self.config.val_metric} to `monitor_metrics`."
                    )
                    self.config.monitor_metrics += [self.config.val_metric]
            self.trainer = PLTTrainer(self.config, classes=classes, embed_vecs=embed_vecs, word_dict=word_dict)
            return
        self._setup_model(
            classes=classes,
            word_dict=word_dict,
            embed_vecs=embed_vecs,
            log_path=self.log_path,
            checkpoint_path=config.checkpoint_path,
        )
        self.trainer = init_trainer(
            checkpoint_dir=self.checkpoint_dir,
            epochs=config.epochs,
            patience=config.patience,
            early_stopping_metric=config.early_stopping_metric,
            val_metric=config.val_metric,
            silent=config.silent,
            use_cpu=config.cpu,
            limit_train_batches=config.limit_train_batches,
            limit_val_batches=config.limit_val_batches,
            limit_test_batches=config.limit_test_batches,
            save_checkpoints=save_checkpoints,
            is_tune_mode=is_tune_mode,
        )
        callbacks = [callback for callback in self.trainer.callbacks if isinstance(callback, ModelCheckpoint)]
        self.checkpoint_callback = callbacks[0] if callbacks else None

    def _setup_model(
        self,
        classes: list = None,
        word_dict: dict = None,
        embed_vecs=None,
        log_path: str = None,
        checkpoint_path: str = None,
    ):
        """Setup model from checkpoint if a checkpoint path is passed in or specified in the config.
        Otherwise, initialize model from scratch.

        Args:
            classes(list): List of class names.
            word_dict(torchtext.vocab.Vocab): A vocab object which maps tokens to indices.
            embed_vecs (torch.Tensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
            log_path (str): Path to the log file. The log file contains the validation
                results for each epoch and the test results. If the `log_path` is None, no performance
                results will be logged.
            checkpoint_path (str): The checkpoint to warm-up with.
        """
        if "checkpoint_path" in self.config and self.config.checkpoint_path is not None:
            checkpoint_path = self.config.checkpoint_path

        if checkpoint_path is not None:
            logging.info(f"Loading model from `{checkpoint_path}` with the previously saved hyper-parameter...")
            self.model = Model.load_from_checkpoint(checkpoint_path, log_path=log_path)
        else:
            logging.info("Initialize model from scratch.")
            if self.config.embed_file is not None:
                logging.info("Load word dictionary ")
                word_dict, embed_vecs = data_utils.load_or_build_text_dict(
                    dataset=self.datasets["train"],
                    vocab_file=self.config.vocab_file,
                    min_vocab_freq=self.config.min_vocab_freq,
                    embed_file=self.config.embed_file,
                    silent=self.config.silent,
                    normalize_embed=self.config.normalize_embed,
                    embed_cache_dir=self.config.embed_cache_dir,
                )
            if not classes:
                classes = data_utils.load_or_build_label(
                    self.datasets, self.config.label_file, self.config.include_test_labels
                )

            if self.config.early_stopping_metric not in self.config.monitor_metrics:
                logging.warn(
                    f"{self.config.early_stopping_metric} is not in `monitor_metrics`. "
                    f"Add {self.config.early_stopping_metric} to `monitor_metrics`."
                )
                self.config.monitor_metrics += [self.config.early_stopping_metric]

            if self.config.val_metric not in self.config.monitor_metrics:
                logging.warn(
                    f"{self.config.val_metric} is not in `monitor_metrics`. "
                    f"Add {self.config.val_metric} to `monitor_metrics`."
                )
                self.config.monitor_metrics += [self.config.val_metric]

            self.model = init_model(
                model_name=self.config.model_name,
                network_config=dict(self.config.network_config),
                classes=classes,
                word_dict=word_dict,
                embed_vecs=embed_vecs,
                init_weight=self.config.init_weight,
                log_path=log_path,
                learning_rate=self.config.learning_rate,
                optimizer=self.config.optimizer,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                lr_scheduler=self.config.lr_scheduler,
                scheduler_config=self.config.scheduler_config,
                val_metric=self.config.val_metric,
                metric_threshold=self.config.metric_threshold,
                monitor_metrics=self.config.monitor_metrics,
                multiclass=self.config.multiclass,
                loss_function=self.config.loss_function,
                silent=self.config.silent,
                save_k_predictions=self.config.save_k_predictions,
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
            classes=self.model.classes,
            device=self.device,
            max_seq_length=self.config.max_seq_length,
            batch_size=self.config.batch_size if split == "train" else self.config.eval_batch_size,
            shuffle=shuffle,
            data_workers=self.config.data_workers,
            word_dict=self.model.word_dict,
            tokenizer=self.tokenizer,
            add_special_tokens=self.config.add_special_tokens,
        )

    def train(self):
        """Train model with pytorch lightning trainer. Set model to the best model after the training
        process is finished.
        """
        if self.config.model_name.lower() == "attentionxml":
            self.trainer.fit(self.datasets)

            dump_log(self.log_path, config=self.config)
            return
        assert (
            self.trainer is not None
        ), "Please make sure the trainer is successfully initialized by `self._setup_trainer()`."
        train_loader = self._get_dataset_loader(split="train", shuffle=self.config.shuffle)

        if "val" not in self.datasets:
            logging.info("No validation dataset is provided. Train without vaildation.")
            self.trainer.fit(self.model, train_loader)
        else:
            val_loader = self._get_dataset_loader(split="val")
            self.trainer.fit(self.model, train_loader, val_loader)

        # Set model to the best model. If the validation process is skipped during
        # training (i.e., val_size=0), the model is set to the last model.
        model_path = self.checkpoint_callback.best_model_path or self.checkpoint_callback.last_model_path
        if model_path:
            logging.info(f"Finished training. Load best model from {model_path}.")
            self._setup_model(checkpoint_path=model_path, log_path=self.log_path)
        else:
            logging.info(
                "No model is saved during training. \
                If you want to save the best and the last model, please set `save_checkpoints` to True."
            )

        dump_log(self.log_path, config=self.config)

        # return best model score for ray
        return self.checkpoint_callback.best_model_score.item() if self.checkpoint_callback.best_model_score else None

    def test(self, split="test"):
        """Test model with pytorch lightning trainer. Top-k predictions are saved
        if `save_k_predictions` > 0.

        Args:
            split (str, optional): One of 'train', 'test', or 'val'. Defaults to 'test'.

        Returns:
            dict: Scores for all metrics in the dictionary format.
        """
        assert "test" in self.datasets and self.trainer is not None

        if self.config.model_name.lower() == "attentionxml":
            self.trainer.test(self.datasets["test"])
            return
        logging.info(f"Testing on {split} set.")
        test_loader = self._get_dataset_loader(split=split)
        metric_dict = self.trainer.test(self.model, dataloaders=test_loader, verbose=False)[0]

        if self.config.save_k_predictions > 0:
            self._save_predictions(test_loader, self.config.predict_out_path)

        dump_log(self.log_path, config=self.config)
        return metric_dict

    def _save_predictions(self, dataloader, predict_out_path):
        """Save top k label results.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader for the test or valid dataset.
            predict_out_path (str): Path to the an output file holding top k label results.
        """
        batch_predictions = self.trainer.predict(self.model, dataloaders=dataloader)
        pred_labels = np.vstack([batch["top_k_pred"] for batch in batch_predictions])
        pred_scores = np.vstack([batch["top_k_pred_scores"] for batch in batch_predictions])
        with open(predict_out_path, "w") as fp:
            for pred_label, pred_score in zip(pred_labels, pred_scores):
                out_str = " ".join(
                    [f"{self.model.classes[label]}:{score:.4}" for label, score in zip(pred_label, pred_score)]
                )
                fp.write(out_str + "\n")
        logging.info(f"Saved predictions to: {predict_out_path}")
