from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import reduce, partial
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import torch
import torch.distributed as dist
from lightning import Trainer
from scipy.sparse import csr_matrix
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only, rank_zero_info

from .cluster import CLUSTER_NAME, FILE_EXTENSION as CLUSTER_FILE_EXTENSION, build_label_tree
from .data_utils import UNK
from .datasets import MultiLabelDataset, PLTDataset
from .model import PLTModel, BaseModel

__all__ = ["PLTTrainer"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PLTTrainer:
    CHECKPOINT_NAME = "model-level-"

    def __init__(
        self,
        config,
        classes: Optional[list] = None,  # TODO: removed in the future
        embed_vecs: Optional[Tensor] = None,
        word_dict: Optional[dict] = None,  # TODO: removed in the future
        mlb=None,  # TODO: removed in the future
    ):
        # The number of levels is set to 2
        # In other words, there will be 2 models

        # cluster
        self.cluster_size = config.cluster_size
        # predict the top k labels
        self.top_k = config.top_k

        # dataset meta info
        self.embed_vecs = embed_vecs
        self.word_dict = word_dict
        self.mlb = mlb
        self.num_labels = len(classes)
        self.is_multiclass = config.multiclass

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
        self.devices = 1
        self.num_nodes = 1
        self.max_epochs = config.epochs
        # callbacks
        self.val_metric = config.val_metric
        self.verbose = not config.silent
        # EarlyStopping
        self.patience = config.patience
        # ModelCheckpoint
        self.result_dir = Path(config.result_dir)
        # SWA/EMA
        # to understand how SWA work, see the pytorch doc and the following link
        # https://stackoverflow.com/questions/68726290/setting-learning-rate-for-stochastic-weight-averaging-in-pytorch
        # self.swa = config.get("swa")
        #
        # if (swa_config := config.get("swa")) is not None:
        #     self.swa_lr = swa_config.get("swa_lr", 5e-2)
        #     self.swa_epoch_start = swa_config.get("swa_epoch_start")
        #     self.annealing_epochs = swa_config.get("annealing_epochs", 10)
        #     self.annealing_strategy = swa_config.get("annealing_strategy", "cos")
        #     # TODO: SWA or EMA?
        #
        #     # self.avg_fn = None  # None == SWA
        #     def ema_avg_fn(averaged_model_parameter: Tensor, model_parameter: Tensor, num_averaged: Tensor) -> Tensor:
        #         decay = 1.0 - 1.0 / num_averaged
        #         return torch.optim.swa_utils.get_ema_avg_fn(decay=decay)
        #
        #     self.avg_fn = ema_avg_fn

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

    def label2node(self, nodes, *ys) -> Generator[csr_matrix, ...]:
        """Map labels (leaf nodes) to ancestor nodes at a certain level.

        If num_labels is 8 and nodes is [(0, 1), (2, 3), (4, 6), (5, 7)].
        Then the mapping is as follows: [0, 0, 1, 1, 2, 3, 2, 3]
        Suppose one element of ys is [0, 1, 7]. The results after mapping is [0, 3].

        Args:
            nodes: the nodes generated at a pre-defined level.
            *ys: true labels (leaf nodes) for train and/or valid datasets.

        Returns:
            Generator[csr_matrix]: the mapped labels (ancestor nodes) for train and/or valid datasets.
        """
        mapping = np.empty(self.num_labels, dtype=np.uint64)
        for idx, node_labels in enumerate(nodes):
            mapping[node_labels] = idx

        def _label2node(y: csr_matrix) -> csr_matrix:
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

        return (_label2node(y) for y in ys)

    def configure_trainer(self, level) -> Trainer:
        callbacks = []
        monitor = self.val_metric
        # loss cannot be calculated for PLTModel
        mode = "max"

        # ModelCheckpoint
        callbacks.append(
            ModelCheckpoint(
                dirpath=self.result_dir,
                filename=f"{self.CHECKPOINT_NAME}{level}",
                monitor=monitor,
                verbose=self.verbose,
                mode=mode,
                enable_version_counter=False,
                save_on_train_epoch_end=True,
            )
        )

        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                patience=self.patience,
                mode=mode,
                verbose=self.verbose,
            )
        )

        trainer = Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            num_nodes=self.num_nodes,
            callbacks=callbacks,
            max_epochs=self.max_epochs,
            # TODO: Decide whether to keep these parameters
            enable_progress_bar=True,
            default_root_dir=self.result_dir,
        )
        return trainer

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
        train_sparse_y_full = self.mlb.transform((i["label"] for i in train_data_full))

        train_x = self.reformat_text(datasets["train"])
        val_x = self.reformat_text(datasets["val"])

        train_y = self.mlb.transform((i["label"] for i in datasets["train"]))
        val_y = self.mlb.transform((i["label"] for i in datasets["val"]))

        # only do clustering on GPU 0
        @rank_zero_only
        def start_cluster():
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    build_label_tree,
                    sparse_x=train_sparse_x,
                    sparse_y=train_sparse_y_full,
                    cluster_size=self.cluster_size,
                    output_dir=self.result_dir,
                )
                future.result()

        start_cluster()

        # wait until the clustering process finishes
        cluster_path = self.get_cluster_path()
        while not cluster_path.exists():
            time.sleep(15)
        clusters = np.load(cluster_path, allow_pickle=True)

        # each y has been mapped to the node indices of its parent
        train_y_cluster, val_y_cluster = self.label2node(clusters, train_y, val_y)
        # regard each internal nodes as a "labels"
        num_labels = len(clusters)

        # trainer
        trainer = self.configure_trainer(level=0)

        best_model_path = self.get_best_model_path(level=0)
        if not best_model_path.exists():
            # train & valid dataloaders for training
            train_dataloader = self.dataloader(MultiLabelDataset(train_x, train_y_cluster), shuffle=self.shuffle)
            val_dataloader = self.dataloader(MultiLabelDataset(val_x, val_y_cluster))

            model = BaseModel(
                network="AttentionXML",
                network_config=self.network_config,
                embed_vecs=self.embed_vecs,
                num_labels=num_labels,
                optimizer=self.optimizer,
                metrics=self.metrics,
                val_metric=self.val_metric,
                top_k=self.top_k,
                is_multiclass=self.is_multiclass,
                init_weight=self.init_weight,
                loss_func=self.loss_func,
                optimizer_params=self.optimizer_config,
            )

            rank_zero_info(f"Training level 0. Number of labels: {num_labels}")
            trainer.fit(model, train_dataloader, val_dataloader)
            rank_zero_info(f"Finish training level 0")

        rank_zero_info(f"Best model loaded from {best_model_path}")
        model = BaseModel.load_from_checkpoint(best_model_path, embed_vecs=self.embed_vecs)

        # Utilize single GPU to predict
        trainer = Trainer(
            num_nodes=1,
            devices=1,
            accelerator=self.accelerator,
            logger=False,
        )
        rank_zero_info(
            f"Generating predictions for level 1. Number of possible predictions: {num_labels}. Top k: {self.top_k}"
        )
        # train & val dataloaders for prediction (without labels)
        train_dataloader = self.eval_dataloader(MultiLabelDataset(train_x))
        val_dataloader = self.eval_dataloader(MultiLabelDataset(val_x))

        # returned labels have been clustered into nodes (groups)
        train_node_pred = trainer.predict(model, train_dataloader)
        valid_node_pred = trainer.predict(model, val_dataloader)

        # shape of node_pred: (n, 2, ~batch_size, top_k). n is floor(num_x / batch_size)
        # new shape: (2, num_x, top_k)
        _, train_node_y_pred = map(torch.vstack, list(zip(*train_node_pred)))
        valid_node_score_pred, valid_node_y_pred = map(torch.vstack, list(zip(*valid_node_pred)))

        # The following process can be simplified using method from LightXML
        rank_zero_info("Getting Candidates")
        node_candidates = np.empty((len(train_x), self.top_k), dtype=np.int64)
        prog = rank_zero_only(tqdm)(train_node_y_pred, leave=False, desc="Parents")
        if prog is None:
            prog = train_node_y_pred
        for i, ys in enumerate(prog):
            # true nodes/labels are positive
            positive = set(train_y_cluster.indices[train_y_cluster.indptr[i] : train_y_cluster.indptr[i + 1]])
            # Regard positive nodes and predicted training nodes that are not in positive as candidates
            # until reaching top_k if the number of positive labels is less than top_k.
            if len(positive) <= self.top_k:
                candidates = positive
                for y in ys:
                    y = y.item()
                    if len(candidates) == self.top_k:
                        break
                    candidates.add(y)
            # Regard positive (true) label as candidates iff they appear in the predicted labels
            # if the number of positive labels is more than top_k. If candidates are not of length top_k
            # add unseen predicted labels until reaching top_k.
            else:
                candidates = set()
                for y in ys:
                    y = y.item()
                    if y in positive:
                        candidates.add(y)
                    if len(candidates) == self.top_k:
                        break
                if len(candidates) < self.top_k:
                    candidates = (list(candidates) + list(positive - candidates))[: self.top_k]
            node_candidates[i] = np.asarray(list(candidates))

        # mapping from the current nodes to leaf nodes.
        assert reduce(lambda a, b: a + len(b), clusters, 0) == self.num_labels

        # trainer
        trainer = self.configure_trainer(level=1)

        # train & valid dataloaders for training
        train_dataloader = self.dataloader(
            PLTDataset(
                train_x,
                train_y,
                num_labels=self.num_labels,
                mapping=clusters,
                node_label=node_candidates,
            ),
            shuffle=self.shuffle,
        )
        valid_dataloader = self.dataloader(
            PLTDataset(
                val_x,
                val_y,
                num_labels=self.num_labels,
                mapping=clusters,
                node_label=valid_node_y_pred,
                node_score=valid_node_score_pred,
            ),
        )

        model = PLTModel(
            network="FastAttentionXML",
            network_config=self.network_config,
            embed_vecs=self.embed_vecs,
            num_labels=self.num_labels,
            optimizer=self.optimizer,
            metrics=self.metrics,
            top_k=self.top_k,
            val_metric=self.val_metric,
            is_multiclass=self.is_multiclass,
            loss_func=self.loss_func,
            optimizer_params=self.optimizer_config,
        )
        torch.nn.init.xavier_uniform_(model.network.attention.attention.weight)

        # initialize model with weights from level 0
        rank_zero_info(f"Loading parameters of level 1 from level 0")
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

        rank_zero_info(
            f"Training level 1, Number of labels: {self.num_labels}, "
            f"Number of candidates: {train_dataloader.dataset.num_candidates}"
        )
        trainer.fit(model, train_dataloader, valid_dataloader)
        rank_zero_info(f"Best model loaded from {best_model_path}")
        rank_zero_info(f"Finish training level 1")

        # testing will hang forever without destroying process group
        if dist.is_initialized():
            dist.destroy_process_group()

    # why we want to test on a single GPU?
    # https://lightning.ai/docs/pytorch/stable/common/evaluation_intermediate.html
    @rank_zero_only
    def test(self, dataset):
        test_x = self.reformat_text(dataset)
        test_y = self.mlb.transform((i["label"] for i in dataset))
        rank_zero_info("Start predicting process.")
        trainer = Trainer(
            devices=1,
            accelerator=self.accelerator,
            logger=False,
        )

        # prediction starts from level 0
        model = BaseModel.load_from_checkpoint(
            self.get_best_model_path(level=0),
            embed_vecs=self.embed_vecs,
            top_k=self.top_k,
            metrics=self.metrics,
        )

        test_dataloader = self.eval_dataloader(MultiLabelDataset(test_x))

        rank_zero_info(f"Predicting level 0, Top: {self.top_k}")
        node_pred = trainer.predict(model, test_dataloader)
        node_score_pred, node_label_pred = map(torch.vstack, list(zip(*node_pred)))

        clusters = np.load(self.get_cluster_path(), allow_pickle=True)

        model = PLTModel.load_from_checkpoint(
            self.get_best_model_path(level=1), embed_vecs=self.embed_vecs, top_k=self.top_k, metrics=self.metrics
        )

        test_dataloader = self.eval_dataloader(
            PLTDataset(
                test_x,
                test_y,
                num_labels=self.num_labels,
                mapping=clusters,
                node_label=node_label_pred,
                node_score=node_score_pred,
            ),
        )

        rank_zero_info(f"Testing on level 1")
        trainer.test(model, test_dataloader)
        rank_zero_info("Testing process finished")

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
