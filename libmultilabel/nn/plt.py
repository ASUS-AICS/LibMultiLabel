from __future__ import annotations

import logging
import time
from functools import reduce, partial
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import torch
from lightning import Trainer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from lightning.pytorch.callbacks import ModelCheckpoint

from .cluster import CLUSTER_NAME, FILE_EXTENSION as CLUSTER_FILE_EXTENSION, build_label_tree
from .data_utils import UNK
from .datasets_AttentionXML import MultiLabelDataset, PLTDataset
from .model_AttentionXML import PLTModel
from ..nn import networks
from ..nn.model import Model

__all__ = ["PLTTrainer"]

from .nn_utils import init_trainer, init_model

logger = logging.getLogger(__name__)


class PLTTrainer:
    CHECKPOINT_NAME = "model-level-"

    def __init__(
        self,
        config,
        classes: Optional[list] = None,
        embed_vecs: Optional[Tensor] = None,
        word_dict: Optional[dict] = None,
    ):
        # The number of levels is set to 2
        # In other words, there will be 2 models

        if config.multiclass:
            raise ValueError(
                "The label space of multi-class datasets is usually not large, so PLT training is unnecessary."
                "Please consider other methods."
                "If you have a multi-class set with numerous labels, please let us know"
            )
        self.is_multiclass = config.multiclass

        # cluster
        self.cluster_size = config.cluster_size
        # predict the top k clusters for deciding relevant/irrelevant labels of each instance in level 1 model training
        self.predict_top_k = config.top_k

        # dataset meta info
        self.embed_vecs = embed_vecs
        self.word_dict = word_dict
        self.classes = classes
        self.num_labels = len(classes)

        # cluster meta info
        self.cluster_size = config.cluster_size

        # network parameters
        self.network_config = config.network_config
        self.init_weight = "xavier_uniform"  # AttentionXML-specific setting
        self.loss_func = config.loss_func

        # optimizer parameters
        self.optimizer = config.optimizer
        self.optimizer_config = config.optimizer_config

        # Trainer parameters
        self.accelerator = config.accelerator
        self.use_cpu = config.cpu
        self.devices = 1
        self.num_nodes = 1
        self.epochs = config.epochs
        self.limit_train_batches = config.limit_train_batches
        self.limit_val_batches = config.limit_val_batches
        self.limit_test_batches = config.limit_test_batches

        # callbacks
        self.val_metric = config.val_metric
        self.silent = config.silent
        # EarlyStopping
        self.patience = config.patience
        # ModelCheckpoint
        self.result_dir = Path(config.result_dir)

        self.metrics = config.monitor_metrics

        # dataloader parameters
        # whether shuffle the training dataset or not during the training process
        self.shuffle = config.shuffle
        pin_memory = True if self.accelerator == "gpu" else False
        # training DataLoader
        self.dataloader = partial(
            DataLoader,
            batch_size=config.batch_size,
            num_workers=config.data_workers,
            pin_memory=pin_memory,
        )
        # evaluation DataLoader
        self.eval_dataloader = partial(
            DataLoader,
            batch_size=config.eval_batch_size,
            num_workers=config.data_workers,
            pin_memory=pin_memory,
        )

        # save path
        self.config = config

    def label2cluster(self, nodes, *ys) -> Generator[csr_matrix, ...]:
        """Map labels to their corresponding clusters in CSR sparse format.

        Suppose there are 6 labels and clusters are [(0, 1), (2, 3), (4, 5)] and ys of a given instance is [0, 1, 4].
        The clusters of the instance are [0, 2].

        Args:
            nodes: the nodes generated at a pre-defined level.
            *ys: true labels (leaf nodes) for train and/or valid datasets.

        Returns:
            Generator[csr_matrix]: the mapped labels (ancestor nodes) for train and/or valid datasets.
        """
        mapping = np.empty(self.num_labels, dtype=np.uint64)
        for idx, node_labels in enumerate(nodes):
            mapping[node_labels] = idx

        def _label2cluster(y: csr_matrix) -> csr_matrix:
            row = []
            col = []
            data = []
            for i in range(y.shape[0]):
                # n include all mapped ancestor nodes
                n = np.unique(mapping[y.indices[y.indptr[i] : y.indptr[i + 1]]])
                row += [i] * len(n)
                col += n.tolist()
                data += [1] * len(n)
            return csr_matrix((data, (row, col)), shape=(y.shape[0], len(nodes)))

        return (_label2cluster(y) for y in ys)

    def fit(self, datasets):
        """fit model to the training dataset

        Args:
            datasets: dict of train, val, and test
        """
        if self.get_best_model_path(level=1).exists():
            return

        train_data_full = datasets["train_full"]
        train_sparse_x = datasets["train_sparse_x"]
        # sparse training labels
        # TODO: remove workaround in future PR
        self.binarizer = MultiLabelBinarizer(classes=self.classes, sparse_output=True)
        self.binarizer.fit(None)
        train_sparse_y_full = self.binarizer.transform((i["label"] for i in train_data_full))

        train_x = self.reformat_text(datasets["train"])
        val_x = self.reformat_text(datasets["val"])

        train_y = self.binarizer.transform((i["label"] for i in datasets["train"]))
        val_y = self.binarizer.transform((i["label"] for i in datasets["val"]))

        # clustering
        build_label_tree(
            sparse_x=train_sparse_x,
            sparse_y=train_sparse_y_full,
            cluster_size=self.cluster_size,
            output_dir=self.result_dir,
        )

        # wait until the clustering process finishes
        cluster_path = self.get_cluster_path()
        while not cluster_path.exists():
            time.sleep(15)
        clusters = np.load(cluster_path, allow_pickle=True)

        # each y has been mapped to the node indices of its parent
        train_y_clustered, val_y_clustered = self.label2cluster(clusters, train_y, val_y)
        # regard each internal nodes as a "labels"
        num_labels = len(clusters)

        # trainer
        trainer = init_trainer(
            self.result_dir,
            epochs=self.epochs,
            patience=self.patience,
            early_stopping_metric=self.val_metric,
            val_metric=self.val_metric,
            silent=self.silent,
            use_cpu=self.use_cpu,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            limit_test_batches=self.limit_test_batches,
            save_checkpoints=True,
        )
        trainer.checkpoint_callback.file_name = f"{self.CHECKPOINT_NAME}0{ModelCheckpoint.FILE_EXTENSION}"

        best_model_path = self.get_best_model_path(level=0)
        if not best_model_path.exists():
            # train & valid dataloaders for training
            train_dataloader = self.dataloader(MultiLabelDataset(train_x, train_y_clustered), shuffle=self.shuffle)
            val_dataloader = self.dataloader(MultiLabelDataset(val_x, val_y_clustered))

            model = init_model(
                model_name="AttentionXML",
                network_config=self.config.network_config,
                classes=self.classes,
                word_dict=self.word_dict,
                embed_vecs=self.embed_vecs,
                init_weight=self.config.init_weight,
                log_path=self.config.log_path,
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

            logger.info(f"Training level 0. Number of labels: {num_labels}")
            trainer.fit(model, train_dataloader, val_dataloader)
            logger.info(f"Finish training level 0")

        logger.info(f"Best model loaded from {best_model_path}")
        model = Model.load_from_checkpoint(best_model_path, embed_vecs=self.embed_vecs)

        # Utilize single GPU to predict
        trainer = Trainer(
            num_nodes=1,
            devices=1,
            accelerator=self.accelerator,
            logger=False,
        )
        logger.info(
            f"Generating predictions for level 1. Number of possible predictions: {num_labels}. Top k: {self.predict_top_k}"
        )
        # load training and validation data and predict corresponding level 0 nodes
        train_dataloader = self.eval_dataloader(MultiLabelDataset(train_x))
        val_dataloader = self.eval_dataloader(MultiLabelDataset(val_x))

        train_pred = trainer.predict(model, train_dataloader)
        val_pred = trainer.predict(model, val_dataloader)

        # shape of node_pred: (n, 2, ~batch_size, top_k). n is floor(num_x / batch_size)
        # new shape: (2, num_x, top_k)
        _, train_clusters_pred = map(torch.vstack, list(zip(*train_pred)))
        val_scores_pred, val_clusters_pred = map(torch.vstack, list(zip(*val_pred)))

        logger.info("Selecting relevant/irrelevant clusters of each instance for level 1 training")
        clusters_selected = np.empty((len(train_x), self.predict_top_k), dtype=np.int64)
        for i, ys in enumerate(tqdm(train_clusters_pred, leave=False, desc="Sampling clusters")):
            # relevant clusters are positive
            pos = set(train_y_clustered.indices[train_y_clustered.indptr[i] : train_y_clustered.indptr[i + 1]])
            # Select relevant clusters first. Then from top-predicted clusters, sequentially include their labels until the total # of
            # until reaching top_k if the number of positive labels is less than top_k.
            if len(pos) <= self.predict_top_k:
                selected = pos
                for y in ys:
                    y = y.item()
                    if len(selected) == self.predict_top_k:
                        break
                    selected.add(y)
            # Regard positive (true) label as samples iff they appear in the predicted labels
            # if the number of positive labels is more than top_k. If samples are not of length top_k
            # add unseen predicted labels until reaching top_k.
            else:
                selected = set()
                for y in ys:
                    y = y.item()
                    if y in pos:
                        selected.add(y)
                    if len(selected) == self.predict_top_k:
                        break
                if len(selected) < self.predict_top_k:
                    selected = (list(selected) + list(pos - selected))[: self.predict_top_k]
            clusters_selected[i] = np.asarray(list(selected))

        # trainer
        trainer = init_trainer(
            self.result_dir,
            epochs=self.epochs,
            patience=self.patience,
            early_stopping_metric=self.val_metric,
            val_metric=self.val_metric,
            silent=self.silent,
            use_cpu=self.use_cpu,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            limit_test_batches=self.limit_test_batches,
            save_checkpoints=True,
        )
        trainer.checkpoint_callback.file_name = f"{self.CHECKPOINT_NAME}1{ModelCheckpoint.FILE_EXTENSION}"

        # train & valid dataloaders for training
        train_dataloader = self.dataloader(
            PLTDataset(
                train_x,
                train_y,
                num_classes=self.num_labels,
                mapping=clusters,
                clusters_selected=clusters_selected,
            ),
            shuffle=self.shuffle,
        )
        valid_dataloader = self.dataloader(
            PLTDataset(
                val_x,
                val_y,
                num_classes=self.num_labels,
                mapping=clusters,
                clusters_selected=val_clusters_pred,
                cluster_scores=val_scores_pred,
            ),
        )

        try:
            network = getattr(networks, "FastAttentionXML")(
                embed_vecs=self.embed_vecs, num_classes=len(self.classes), **dict(self.network_config)
            )
        except Exception:
            raise AttributeError("Failed to initialize AttentionXML")

        model = PLTModel(
            network=network,
            network_config=self.network_config,
            embed_vecs=self.embed_vecs,
            num_labels=self.num_labels,
            optimizer=self.optimizer,
            metrics=self.metrics,
            top_k=self.predict_top_k,
            val_metric=self.val_metric,
            is_multiclass=self.is_multiclass,
            loss_func=self.loss_func,
            optimizer_params=self.optimizer_config,
        )
        torch.nn.init.xavier_uniform_(model.network.attention.attention.weight)

        # initialize model with weights from level 0
        logger.info(f"Loading parameters of level 1 from level 0")
        state_dict = torch.load(self.get_best_model_path(level=0))["state_dict"]

        # remove the name prefix in state_dict starting with "network.xxx"
        embedding_state_dict = {}
        encoder_state_dict = {}
        output_state_dict = {}
        for n, p in state_dict.items():
            truncated_n = n.split(".", 2)[-1]
            if n.startswith("network.embedding"):
                embedding_state_dict[truncated_n] = p
            elif n.startswith("network.encoder"):
                encoder_state_dict[truncated_n] = p
            elif n.startswith("network.output"):
                output_state_dict[truncated_n] = p
        model.network.embedding.load_state_dict(embedding_state_dict)
        model.network.encoder.load_state_dict(encoder_state_dict)
        model.network.output.load_state_dict(output_state_dict)

        logger.info(
            f"Training level 1. Number of labels: {self.num_labels}."
            f"Number of clusters selected: {train_dataloader.dataset.num_clusters_selected}"
        )
        trainer.fit(model, train_dataloader, valid_dataloader)
        logger.info(f"Best model loaded from {best_model_path}")
        logger.info(f"Finish training level 1")

    def test(self, dataset):
        test_x = self.reformat_text(dataset)
        test_y = self.binarizer.transform((i["label"] for i in dataset))
        logger.info("Testing process started")
        trainer = Trainer(
            devices=1,
            accelerator=self.accelerator,
            logger=False,
        )

        # prediction starts from level 0
        model = BaseModel.load_from_checkpoint(
            self.get_best_model_path(level=0),
            embed_vecs=self.embed_vecs,
            top_k=self.predict_top_k,
            metrics=self.metrics,
        )

        test_dataloader = self.eval_dataloader(MultiLabelDataset(test_x))

        logger.info(f"Predicting level 0, Top: {self.predict_top_k}")
        node_pred = trainer.predict(model, test_dataloader)
        node_score_pred, node_label_pred = map(torch.vstack, list(zip(*node_pred)))

        clusters = np.load(self.get_cluster_path(), allow_pickle=True)

        model = PLTModel.load_from_checkpoint(
            self.get_best_model_path(level=1),
            embed_vecs=self.embed_vecs,
            top_k=self.predict_top_k,
            metrics=self.metrics,
        )

        test_dataloader = self.eval_dataloader(
            PLTDataset(
                test_x,
                test_y,
                num_classes=self.num_labels,
                mapping=clusters,
                clusters_selected=node_label_pred,
                cluster_scores=node_score_pred,
            ),
        )

        logger.info(f"Testing on level 1")
        trainer.test(model, test_dataloader)
        logger.info("Testing process finished")

    def reformat_text(self, dataset):
        encoded_text = list(
            map(
                lambda text: torch.tensor([self.word_dict[word] for word in text], dtype=torch.int64)
                if text
                else torch.tensor([self.word_dict[UNK]], dtype=torch.int64),
                [instance["text"][: self.config["max_seq_length"]] for instance in dataset],
            )
        )
        # pad the first entry to be of length 500 if necessary
        encoded_text[0] = torch.cat(
            (
                encoded_text[0],
                torch.tensor(0, dtype=torch.int64).repeat(self.config["max_seq_length"] - encoded_text[0].shape[0]),
            )
        )
        encoded_text = pad_sequence(encoded_text, batch_first=True)
        return encoded_text

    def get_best_model_path(self, level: int) -> Path:
        return self.result_dir / f"{self.CHECKPOINT_NAME}{level}{ModelCheckpoint.FILE_EXTENSION}"

    def get_cluster_path(self) -> Path:
        return self.result_dir / f"{CLUSTER_NAME}{CLUSTER_FILE_EXTENSION}"
