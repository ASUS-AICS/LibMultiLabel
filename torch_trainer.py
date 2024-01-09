from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_info, rank_zero_warn
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from libmultilabel.common_utils import dump_log, is_multiclass_dataset, AttributeDict
from libmultilabel.nn import data_utils
from libmultilabel.nn.data_utils import UNK
from libmultilabel.nn.datasets import MultiLabelDataset
from libmultilabel.nn.model import Model, BaseModel
from libmultilabel.nn.nn_utils import init_device, init_model, init_trainer, set_seed
from libmultilabel.nn.plt import PLTTrainer


class TorchTrainer:
    """A wrapper for training neural network models with pytorch lightning trainer.

    Args:
        config (AttributeDict): Config of the experiment.
        datasets (dict, optional): Datasets for training, validation, and test. Defaults to None.
        classes(list, optional): List of class names.
        word_dict(torchtext.vocab.Vocab, optional): A vocab object which maps tokens to indices.
        embed_vecs (torch.Tensor, optional): The pre-trained word vectors of shape (vocab_size, embed_dim).
        search_params (bool, optional): Enable pytorch-lightning trainer to report the results to ray tune
            on validation end during hyperparameter search. Defaults to False.
        save_checkpoints (bool, optional): Whether to save the last and the best checkpoint or not.
            Defaults to True.
    """

    def __init__(
        self,
        config: AttributeDict,
        datasets: dict = None,
        classes: list = None,
        word_dict: dict = None,
        embed_vecs=None,
        search_params: bool = False,
        save_checkpoints: bool = True,
    ):
        self.run_name = config.run_name
        self.checkpoint_dir = config.checkpoint_dir
        self.log_path = config.log_path
        self.classes = classes
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Set up seed & device
        set_seed(seed=config.seed)
        self.device = init_device(use_cpu=config.cpu)
        self.config = config

        if self.config.model_name.lower() == "FastAttentionXML":
            if datasets is None:
                self.datasets = data_utils.load_datasets(
                    training_data=config.training_file,
                    training_sparse_data=config.training_sparse_file,
                    test_data=config.test_file,
                    val_data=config.val_file,
                    val_size=config.val_size,
                    merge_train_val=config.merge_train_val,
                    tokenize_text=True,
                    tokenizer=config.get("tokenizer", "regex"),
                    remove_no_label_data=config.remove_no_label_data,
                    random_state=config.get("random_state", 1270),
                )
            if not (Path(config.test_file).parent / "word_dict.vocab").exists():
                rank_zero_info("Calculating word dictionary and embeddings.")
                word_dict, embed_vecs = data_utils.load_or_build_text_dict(
                    dataset=self.datasets["train"] + self.datasets["val"],
                    vocab_file=self.config.vocab_file,
                    # TODO: move to config
                    min_vocab_freq=self.config.min_vocab_freq,
                    embed_file=self.config.embed_file,
                    silent=self.config.silent,
                    normalize_embed=self.config.normalize_embed,
                    embed_cache_dir=self.config.embed_cache_dir,
                    max_tokens=self.config.get("max_tokens"),
                    unk_init=self.config.get("unk_init", "uniform"),
                    unk_init_param=self.config.get("unk_init_param", {-1, 1}),
                    apply_all=self.config.get("apply_all", True),
                )
                if word_dict is not None:
                    torch.save(word_dict, Path(config.test_file).parent / "word_dict.vocab")
                    torch.save(embed_vecs, Path(config.test_file).parent / "word_embeddings.tensor")
                # barrier
                while not (Path(config.test_file).parent / "word_dict.vocab").exists():
                    time.sleep(15)

            word_dict = torch.load(Path(config.test_file).parent / "word_dict.vocab")
            embed_vecs = torch.load(Path(config.test_file).parent / "word_embeddings.tensor")

            if not classes:
                classes = data_utils.load_or_build_label(
                    self.datasets, self.config.label_file, self.config.include_test_labels
                )
            self.trainer = PLTTrainer(config, classes=classes, word_dict=word_dict, embed_vecs=embed_vecs)
        else:
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
                    tokenizer=config.get("tokenizer", "regex"),
                    remove_no_label_data=config.remove_no_label_data,
                )
            else:
                self.datasets = datasets

            self.config.multiclass = is_multiclass_dataset(self.datasets["train"] + self.datasets.get("val", list()))
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
                search_params=search_params,
                save_checkpoints=save_checkpoints,
                config=self.config,
            )
            callbacks = [callback for callback in self.trainer.callbacks if isinstance(callback, ModelCheckpoint)]
            self.checkpoint_callback = callbacks[0] if callbacks else None

            # Dump config to log
            dump_log(self.log_path, config=config)

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
                if not (Path(self.config.test_file).parent / "word_dict.vocab").exists():
                    rank_zero_info("Calculating word dictionary and embeddings.")
                    rank_zero_info("Load word dictionary")
                    word_dict, embed_vecs = data_utils.load_or_build_text_dict(
                        dataset=self.datasets["train"]
                        if self.config.model_name.lower() != "AttentionXML"
                        else self.datasets["train"] + self.datasets["val"],
                        vocab_file=self.config.vocab_file,
                        min_vocab_freq=self.config.min_vocab_freq,
                        embed_file=self.config.embed_file,
                        silent=self.config.silent,
                        normalize_embed=self.config.normalize_embed,
                        embed_cache_dir=self.config.embed_cache_dir,
                        max_tokens=self.config.get("max_tokens"),
                        unk_init=self.config.get("unk_init"),
                        unk_init_param=self.config.get("unk_init_param"),
                        apply_all=self.config.get("apply_all", False),
                    )
                    if word_dict is not None:
                        torch.save(word_dict, Path(self.config.test_file).parent / "word_dict.vocab")
                        torch.save(embed_vecs, Path(self.config.test_file).parent / "word_embeddings.tensor")
                    # barrier
                    while not (Path(self.config.test_file).parent / "word_dict.vocab").exists():
                        time.sleep(15)

                word_dict = torch.load(Path(self.config.test_file).parent / "word_dict.vocab")
                embed_vecs = torch.load(Path(self.config.test_file).parent / "word_embeddings.tensor")
                self.word_dict = word_dict
                self.embed_vecs = embed_vecs

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
                config=self.config,
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
        assert (
            self.trainer is not None
        ), "Please make sure the trainer is successfully initialized by `self._setup_trainer()`."

        if self.config.model_name.lower() == "FastAttentionXML":
            self.trainer.fit(self.datasets)
        else:
            if self.config.model_name.lower() == "AttentionXML":
                if (
                    Path(self.config.dir_path) / f"{self.config.data_name}-{self.config.model_name}" / "Model.ckpt"
                ).exists():
                    return

                classes = data_utils.load_or_build_label(
                    self.datasets, self.config.label_file, self.config.include_test_labels
                )

                self.model.train_mlb = MultiLabelBinarizer(classes=classes, sparse_output=True)
                self.model.train_mlb.fit(None)

                train_x = self.reformat_text("train")
                valid_x = self.reformat_text("val")

                train_y = self.model.train_mlb.transform((i["label"] for i in self.datasets["train"]))
                valid_y = self.model.train_mlb.transform((i["label"] for i in self.datasets["val"]))

                train_dataloader = DataLoader(
                    MultiLabelDataset(train_x, train_y),
                    batch_size=self.config.batch_size,
                    num_workers=self.config.get("num_workers", 16),
                    pin_memory=True,
                    shuffle=True,
                )
                valid_dataloader = DataLoader(
                    MultiLabelDataset(valid_x, valid_y),
                    batch_size=self.config.batch_size,
                    num_workers=self.config.get("num_workers", 16),
                    pin_memory=True,
                )
                self.trainer.fit(self.model, train_dataloader, valid_dataloader)
                self.model = BaseModel.load_from_checkpoint(self.trainer.checkpoint_callback.best_model_path)
            else:
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

    @rank_zero_only
    def test(self, split="test"):
        """Test model with pytorch lightning trainer. Top-k predictions are saved
        if `save_k_predictions` > 0.

        Args:
            split (str, optional): One of 'train', 'test', or 'val'. Defaults to 'test'.

        Returns:
            dict: Scores for all metrics in the dictionary format.
        """
        assert "test" in self.datasets and self.trainer is not None

        if self.config.model_name == "FastAttentionXML":
            self.trainer.test(self.datasets["test"])
        else:
            if self.config.model_name.lower() == "attentionxml":
                if self.model is None:
                    self.model = BaseModel.load_from_checkpoint(
                        Path(self.config.dir_path)
                        / f"{self.config.data_name}-{self.config.model_name}-{0}"
                        / "Model.ckpt"
                    )
                classes = data_utils.load_or_build_label(
                    self.datasets, self.config.label_file, self.config.include_test_labels
                )

                self.model.train_mlb = MultiLabelBinarizer(classes=classes, sparse_output=True)
                self.model.train_mlb.fit(None)

                test_x = self.reformat_text("test")

                # self.model.test_mlb = MultiLabelBinarizer(sparse_output=True)
                # test_y = self.model.test_mlb.fit_transform((i["label"] for i in self.datasets["test"]))
                test_y = self.model.train_mlb.transform((i["label"] for i in self.datasets["test"]))

                test_dataloader = DataLoader(
                    MultiLabelDataset(test_x, test_y),
                    batch_size=self.config.batch_size,
                    num_workers=self.config.get("num_workers", 16),
                    pin_memory=True,
                )
                # use one and only one GPU for prediction
                if dist.is_initialized():
                    dist.destroy_process_group()
                    torch.cuda.empty_cache()
                trainer = pl.Trainer(
                    devices=1,
                    accelerator=self.config.accelerator,
                )

                trainer.test(self.model, test_dataloader)
            else:
                logging.info(f"Testing on {split} set.")
                test_loader = self._get_dataset_loader(split=split)
                metric_dict = self.trainer.test(self.model, dataloaders=test_loader, verbose=False)[0]

                if self.config.save_k_predictions > 0:
                    self._save_predictions(test_loader, self.config.predict_out_path)

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

    def reformat_text(self, split):
        encoded_text = list(
            map(
                lambda text: torch.tensor([self.word_dict[word] for word in text], dtype=torch.int)
                if text
                else torch.tensor([self.word_dict[UNK]], dtype=torch.int),
                [instance["text"][: self.config["max_seq_length"]] for instance in self.datasets[split]],
            )
        )
        # pad the first entry to be of length 500 if necessary
        encoded_text[0] = torch.cat(
            (
                encoded_text[0],
                torch.tensor(0, dtype=torch.int).repeat(self.config["max_seq_length"] - encoded_text[0].shape[0]),
            )
        )
        encoded_text = pad_sequence(encoded_text, batch_first=True)
        return encoded_text
