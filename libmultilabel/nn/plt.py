from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import torch
from scipy.special import expit
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
from .datasets_AttentionXML import PlainDataset, PLTDataset
from .model_AttentionXML import PLTModel
from ..linear.preprocessor import Preprocessor
from ..nn import networks
from ..nn.model import Model

__all__ = ["PLTTrainer"]

from .nn_utils import init_trainer, init_model

logger = logging.getLogger(__name__)


class PLTTrainer:
    CHECKPOINT_NAME = "model_"

    def __init__(
        self,
        config,
        classes: Optional[list] = None,
        embed_vecs: Optional[Tensor] = None,
        word_dict: Optional[dict] = None,
    ):
        # The number of levels is set to 2. In other words, there will be 2 models
        self.multiclass = config.multiclass
        if self.multiclass:
            raise ValueError(
                "The label space of multi-class datasets is usually not large, so PLT training is unnecessary."
                "Please consider other methods."
                "If you have a multi-class set with numerous labels, please let us know"
            )

        # cluster
        self.cluster_size = config.cluster_size
        # predict the top k clusters for deciding relevant/irrelevant labels of each instance in level 1 model training
        self.predict_top_k = config.save_k_predictions

        # dataset meta info
        self.embed_vecs = embed_vecs
        self.word_dict = word_dict
        self.classes = classes
        self.max_seq_length = config.max_seq_length
        self.num_classes = len(classes)

        # multilabel binarizer fitted to the datasets
        self.binarizer = None

        # cluster meta info
        self.cluster_size = config.cluster_size

        # network parameters
        self.network_config = config.network_config
        self.init_weight = "xavier_uniform"  # AttentionXML-specific setting
        self.loss_function = config.loss_function

        # optimizer parameters
        self.optimizer = config.optimizer
        self.learning_rate = config.learning_rate
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        # learning rate scheduler
        self.lr_scheduler = config.lr_scheduler
        self.scheduler_config = config.scheduler_config

        # Trainer parameters
        self.use_cpu = config.cpu
        self.accelerator = "cpu" if self.use_cpu else "gpu"
        self.devices = 1
        self.num_nodes = 1
        self.epochs = config.epochs
        self.limit_train_batches = config.limit_train_batches
        self.limit_val_batches = config.limit_val_batches
        self.limit_test_batches = config.limit_test_batches

        # callbacks
        self.silent = config.silent
        # EarlyStopping
        self.early_stopping_metric = config.early_stopping_metric
        self.patience = config.patience
        # ModelCheckpoint
        self.val_metric = config.val_metric
        self.checkpoint_dir = Path(config.checkpoint_dir)

        self.metrics = config.monitor_metrics
        self.metric_threshold = config.metric_threshold
        self.monitor_metrics = config.monitor_metrics

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
            self.dataloader,
            batch_size=config.eval_batch_size,
        )

        # save path
        self.log_path = config.log_path

    def label2cluster(self, cluster_mapping, *ys) -> Generator[csr_matrix, ...]:
        """Map labels to their corresponding clusters in CSR sparse format.

        Suppose there are 6 labels and clusters are [(0, 1), (2, 3), (4, 5)] and ys of a given instance is [0, 1, 4].
        The clusters of the instance are [0, 2].

        Args:
            cluster_mapping: mapping from clusters to labels.
            *ys: sparse labels.

        Returns:
            Generator[csr_matrix]: clusters generated from labels
        """
        mapping = np.empty(self.num_classes, dtype=np.uint64)
        for idx, clusters in enumerate(cluster_mapping):
            mapping[clusters] = idx

        def _label2cluster(y: csr_matrix) -> csr_matrix:
            row = []
            col = []
            data = []
            for i in range(y.shape[0]):
                # n include all mapped ancestor clusters
                n = np.unique(mapping[y.indices[y.indptr[i] : y.indptr[i + 1]]])
                row += [i] * len(n)
                col += n.tolist()
                data += [1] * len(n)
            return csr_matrix((data, (row, col)), shape=(y.shape[0], len(cluster_mapping)))

        return (_label2cluster(y) for y in ys)

    # def cluster2label(self, cluster_mapping, *ys):
    #     """Map clusters to their corresponding labels. Notice this function only deals with dense matrix.
    #
    #     Args:
    #         cluster_mapping: mapping from clusters to labels.
    #         *ys: sparse clusters.
    #
    #     Returns:
    #         Generator[csr_matrix]: labels generated from clusters
    #     """
    #
    #     def _cluster2label(y: csr_matrix) -> csr_matrix:
    #         self.labels_selected = [np.concatenate(cluster_mapping[labels]) for labels in y]
    #     return (_cluster2label(y) for y in ys)

    # def generate_goals(self, cluster_scores, y):
    #     if cluster_scores is not None:
    #         # label_scores are corresponding scores for selected labels and
    #         # look like [[0.1, 0.1, 0.1, 0.4, 0.4, 0.5, 0.5,...], ...]. shape: (len(x), cluster_size * top_k)
    #         # notice how scores repeat for each cluster.
    #         self.label_scores = [
    #             np.repeat(scores, [len(i) for i in cluster_mapping[labels]])
    #             for labels, scores in zip(y, cluster_scores)
    #         ]

    def fit(self, datasets):
        """fit model to the training dataset

        Args:
            datasets: dict containing training, validation, and/or test datasets
        """
        if self.get_best_model_path(level=1).exists():
            return

        # AttentionXML-specific data preprocessing
        train_val_dataset = datasets["train"] + datasets["val"]
        train_val_dataset = {
            "x": [" ".join(i["text"]) for i in train_val_dataset],
            "y": [i["label"] for i in train_val_dataset],
        }

        # Preprocessor does tf-idf vectorization and multilabel binarization
        # For details, see libmultilabel.linear.preprocessor.Preprocessor
        preprocessor = Preprocessor()
        datasets_temp = {"data_format": "txt", "train": train_val_dataset, "classes": self.classes}
        # Preprocessor requires the input dictionary to has a key named "train" and will return a new dictionary with
        # the same key.
        train_val_dataset_tf = preprocessor.fit_transform(datasets_temp)["train"]
        # save binarizer for testing
        self.binarizer = preprocessor.binarizer

        train_x = self.reformat_text(datasets["train"])
        val_x = self.reformat_text(datasets["val"])

        train_y = train_val_dataset_tf["y"][: len(datasets["train"])]
        val_y = train_val_dataset_tf["y"][len(datasets["train"]) :]

        # clusters are saved to the disk so that users doesn't need to provide the original training data when they want
        # to do predicting solely
        build_label_tree(
            sparse_x=train_val_dataset_tf["x"],
            sparse_y=train_val_dataset_tf["y"],
            cluster_size=self.cluster_size,
            output_dir=self.checkpoint_dir,
        )

        clusters = np.load(self.get_cluster_path(), allow_pickle=True)

        # each y has been mapped to the cluster indices of its parent
        train_y_clustered, val_y_clustered = self.label2cluster(clusters, train_y, val_y)

        trainer = init_trainer(
            self.checkpoint_dir,
            epochs=self.epochs,
            patience=self.patience,
            early_stopping_metric=self.early_stopping_metric,
            val_metric=self.val_metric,
            silent=self.silent,
            use_cpu=self.use_cpu,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            limit_test_batches=self.limit_test_batches,
            save_checkpoints=True,
        )
        trainer.checkpoint_callback.filename = f"{self.CHECKPOINT_NAME}0"

        train_dataloader = self.dataloader(PlainDataset(train_x, train_y_clustered), shuffle=self.shuffle)
        val_dataloader = self.dataloader(PlainDataset(val_x, val_y_clustered))

        best_model_path = self.get_best_model_path(level=0)
        if not best_model_path.exists():
            model_0 = init_model(
                model_name="AttentionXML_0",
                network_config=self.network_config,
                classes=clusters,
                word_dict=self.word_dict,
                embed_vecs=self.embed_vecs,
                init_weight=self.init_weight,
                log_path=self.log_path,
                learning_rate=self.learning_rate,
                optimizer=self.optimizer,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                lr_scheduler=self.lr_scheduler,
                scheduler_config=self.scheduler_config,
                val_metric=self.val_metric,
                metric_threshold=self.metric_threshold,
                monitor_metrics=self.monitor_metrics,
                multiclass=self.multiclass,
                loss_function=self.loss_function,
                silent=self.silent,
                save_k_predictions=self.predict_top_k,
            )

            logger.info(f"Training level 0. Number of clusters: {len(clusters)}")
            trainer.fit(model_0, train_dataloader, val_dataloader)
            logger.info(f"Finish training level 0")

        logger.info(f"Best model loaded from {best_model_path}")
        model_0 = Model.load_from_checkpoint(best_model_path)

        logger.info(f"Generating predictions for level 1. Will use the top {self.predict_top_k} predictions")
        # load training and validation data and predict corresponding level 0 clusters

        train_pred = trainer.predict(model_0, train_dataloader)
        val_pred = trainer.predict(model_0, val_dataloader)

        train_clusters_pred = np.vstack([i["top_k_pred"] for i in train_pred])
        val_scores_pred = expit(np.vstack([i["top_k_pred_scores"] for i in val_pred]))
        val_clusters_pred = np.vstack([i["top_k_pred"] for i in val_pred])

        logger.info(
            "Selecting relevant/irrelevant clusters of each instance for generating labels for level 1 training"
        )
        clusters_selected = np.empty((len(train_x), self.predict_top_k), dtype=np.int64)
        for i, ys in enumerate(tqdm(train_clusters_pred, leave=False, desc="Sampling clusters")):
            # relevant clusters are positive
            pos = set(train_y_clustered.indices[train_y_clustered.indptr[i] : train_y_clustered.indptr[i + 1]])
            # Select relevant clusters first. Then from top-predicted clusters, sequentially include them until
            # clusters reach top_k
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

        trainer = init_trainer(
            self.checkpoint_dir,
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
        trainer.checkpoint_callback.filename = f"{self.CHECKPOINT_NAME}1"

        # train & valid dataloaders for training
        train_dataloader = self.dataloader(
            PLTDataset(
                train_x,
                train_y,
                num_classes=self.num_classes,
                mapping=clusters,
                clusters_selected=clusters_selected,
            ),
            shuffle=self.shuffle,
        )
        val_dataloader = self.dataloader(
            PLTDataset(
                val_x,
                val_y,
                num_classes=self.num_classes,
                mapping=clusters,
                clusters_selected=val_clusters_pred,
                cluster_scores=val_scores_pred,
            ),
        )

        try:
            network = getattr(networks, "AttentionXML_1")(
                embed_vecs=self.embed_vecs, num_classes=len(self.classes), **dict(self.network_config)
            )
        except Exception:
            raise AttributeError("Failed to initialize AttentionXML")

        model_1 = PLTModel(
            classes=self.classes,
            word_dict=self.word_dict,
            embed_vecs=self.embed_vecs,
            network=network,
            log_path=self.log_path,
            learning_rate=self.learning_rate,
            optimizer=self.optimizer,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            lr_scheduler=self.lr_scheduler,
            scheduler_config=self.scheduler_config,
            val_metric=self.val_metric,
            metric_threshold=self.metric_threshold,
            monitor_metrics=self.monitor_metrics,
            multiclass=self.multiclass,
            loss_function=self.loss_function,
            silent=self.silent,
            save_k_predictions=self.predict_top_k,
        )
        torch.nn.init.xavier_uniform_(model_1.network.attention.attention.weight)

        logger.info(f"Initialize model with weights from the last level")
        # As the attention layer of model 1 is different from model 0, each layer needs to be initialized separately
        model_1.network.embedding.load_state_dict(model_0.network.embedding.state_dict())
        model_1.network.encoder.load_state_dict(model_0.network.encoder.state_dict())
        model_1.network.output.load_state_dict(model_0.network.output.state_dict())

        del model_0

        logger.info(
            f"Training level 1. Number of labels: {self.num_classes}."
            f"Number of labels selected: {train_dataloader.dataset.num_labels_selected}"
        )
        trainer.fit(model_1, train_dataloader, val_dataloader)
        logger.info(f"Best model loaded from {best_model_path}")
        logger.info(f"Finish training level 1")

    def test(self, dataset):
        # retrieve word_dict from model_1
        # prediction starts from level 0
        model_0 = Model.load_from_checkpoint(
            self.get_best_model_path(level=0),
            save_k_predictions=self.predict_top_k,
        )
        model_1 = PLTModel.load_from_checkpoint(
            self.get_best_model_path(level=1),
            save_k_predictions=self.predict_top_k,
            metrics=self.metrics,
        )
        self.word_dict = model_1.word_dict
        classes = model_1.classes

        test_x = self.reformat_text(dataset)

        if self.binarizer is None:
            binarizer = MultiLabelBinarizer(classes=classes, sparse_output=True)
            binarizer.fit(None)
            test_y = binarizer.transform((i["label"] for i in dataset))
        else:
            test_y = self.binarizer.transform((i["label"] for i in dataset))
        logger.info("Testing process started")
        trainer = Trainer(
            devices=1,
            accelerator=self.accelerator,
            logger=False,
        )

        test_dataloader = self.eval_dataloader(PlainDataset(test_x))

        logger.info(f"Predicting level 0, Top: {self.predict_top_k}")
        test_pred = trainer.predict(model_0, test_dataloader)
        test_pred_scores = expit(np.vstack([i["top_k_pred_scores"] for i in test_pred]))
        test_pred_cluters = np.vstack([i["top_k_pred"] for i in test_pred])

        clusters = np.load(self.get_cluster_path(), allow_pickle=True)

        test_dataloader = self.eval_dataloader(
            PLTDataset(
                test_x,
                test_y,
                num_classes=self.num_classes,
                mapping=clusters,
                clusters_selected=test_pred_cluters,
                cluster_scores=test_pred_scores,
            ),
        )

        logger.info(f"Testing on level 1")
        trainer.test(model_1, test_dataloader)
        logger.info("Testing process finished")

    def reformat_text(self, dataset):
        encoded_text = list(
            map(
                lambda text: torch.tensor([self.word_dict[word] for word in text], dtype=torch.int64)
                if text
                else torch.tensor([self.word_dict[UNK]], dtype=torch.int64),
                [instance["text"][: self.max_seq_length] for instance in dataset],
            )
        )
        # pad the first entry to be of length 500 if necessary
        encoded_text[0] = torch.cat(
            (
                encoded_text[0],
                torch.tensor(0, dtype=torch.int64).repeat(self.max_seq_length - encoded_text[0].shape[0]),
            )
        )
        encoded_text = pad_sequence(encoded_text, batch_first=True)
        return encoded_text

    def get_best_model_path(self, level: int) -> Path:
        return self.checkpoint_dir / f"{self.CHECKPOINT_NAME}{level}{ModelCheckpoint.FILE_EXTENSION}"

    def get_cluster_path(self) -> Path:
        return self.checkpoint_dir / f"{CLUSTER_NAME}{CLUSTER_FILE_EXTENSION}"
