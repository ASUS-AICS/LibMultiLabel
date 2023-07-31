from __future__ import annotations

import logging
import math
import time
from functools import reduce
from pathlib import Path
from typing import Generator
from datetime import datetime

import numpy as np
import torch
from multiprocessing import Process

import torch.distributed as dist
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import normalize, MultiLabelBinarizer
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .data_utils import MultiLabelDataset, PLTDataset
from .model import PLTModel, BaseModel
from .nn_utils import is_global_zero
from ..common_utils import GLOBAL_RANK

__all__ = ["PLTTrainer"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PLTTrainer:
    # TODO: adapt the trainer to a general one for PLT models

    def __init__(
        self,
        config,
        ensemble_id: str | int = "0",
        classes: list | None = None,
        word_dict: dict | None = None,
        embed_vecs=None,
        log_path: str | None = None,
    ):
        self.config = config
        # cluster
        self.num_levels = self.config.num_levels
        self.cluster_size = self.config.cluster_size
        self.top_k = self.config.get("top_k", 100)

        self.classes = classes
        self.word_dict = word_dict
        self.embed_vecs = embed_vecs
        self.log_path = log_path
        self.num_labels = len(classes)
        self.mlb = MultiLabelBinarizer(sparse_output=True)
        self.mlb.fit([classes])
        self.ensemble_id = ensemble_id

        # the height of the tree satisfies the following inequation:
        # 2^(tree_height - 1) * cluster_size < num_labels <= 2^tree_height * cluster_size
        self.cluster_size = self.config["cluster_size"]
        if np.log2(self.cluster_size) % 1 != 0:
            raise ValueError("cluster size must be a power of 2")

        self.num_levels = self.config["num_levels"]
        self.tree_height = np.ceil(np.log2(self.num_labels / self.cluster_size))

        # sanity check between num_labels and cluster_size
        q = {self.num_labels}
        while q:
            next_q = set()
            for i in q:
                next_q |= {math.ceil(i / 2), math.floor(i / 2)} if i % 2 != 0 else {i // 2}
            if any(i <= self.cluster_size for i in next_q):
                if not all(i <= self.cluster_size for i in next_q):
                    raise ValueError("The number of labels is incompatible with the cluster size")
                next_q = set()
            q = next_q

        # deduce levels from tree height, number of levels, and cluster size
        self.levels = [
            (self.tree_height - i * np.log2(self.cluster_size)).astype(int) for i in range(self.num_levels - 2, -1, -1)
        ]

        # network parameters
        self.loss_fn = self.config["loss_fn"]

        # optimizer parameters
        self.optimizer = self.config["optimizer"]

        # Trainer parameters
        self.accelerator = self.config["accelerator"]
        self.devices = self.config["devices"]
        self.num_nodes = self.config["num_nodes"]
        self.max_epochs = self.config["epochs"]
        self.strategy = "ddp" if (self.num_nodes > 1 or self.devices > 1) and self.accelerator == "gpu" else "auto"

        if GLOBAL_RANK == 0:
            logger.info(
                f"Accelerator: {self.accelerator}, devices: {self.devices}, num_nodes: {self.num_nodes} "
                f"max_epochs: {self.max_epochs}, strategy: {self.strategy}"
            )

        # dataloader parameters
        self.batch_size = self.config["batch_size"]
        self.pin_memory = True if self.accelerator == "gpu" else False
        self.num_workers = self.config.get("num_workers", 8)

        # model parameters
        self.metrics = self.config["monitor_metrics"]
        self.eval_metric = self.config["val_metric"]

        # save path
        self.dir_path = (
            Path(self.config["dir_path"]) / f"{self.config.data_name}-{self.config.model_name}-{self.ensemble_id}"
        )

    def label2node(self, nodes, *ys) -> Generator[csr_matrix, ...]:
        """Map true labels (leaf nodes) to ancestor nodes at a certain level.

        If num_labels is 8 and nodes is [(0, 1), (2, 3), (4, 6), (5, 7)].
        Then the mapping is as follows: [0, 0, 1, 1, 2, 3, 2, 3]
        Suppose one element of ys is [0, 1, 7]. The results after mapping is [0, 3].

        Args:
            nodes: the nodes generated at a pre-defined level.
            *ys: true labels (leaf nodes) for train and/or valid datasets.

        Returns:
            Generator[csr_matrix]: the mapped labels (ancestor nodes) for train and/or valid datasets.
        """
        mapping = np.empty(self.num_labels, dtype=np.uint32)
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
        monitor = "_".join(["valid", self.config["val_metric"].lower()])

        if level == 0:
            model_checkpoint = ModelCheckpoint(
                dirpath=self.dir_path,
                filename=f"Model-Level-{level}",
                save_top_k=1,
                monitor=monitor,
                verbose=True,
                mode="min" if monitor == "loss" else "max",
            )
            callbacks += [model_checkpoint]
        # else:
        #     model_checkpoint = ModelCheckpoint(
        #         dirpath=self.dir_path,
        #         filename=f"Model-Level-{level}",
        #         save_last=True,
        #         verbose=True,
        #     )
        #     model_checkpoint.CHECKPOINT_NAME_LAST = (f"Model-Level-{level}",)

        # callbacks += [
        #     StochasticWeightAveraging(
        #         swa_lrs=float(self.config.get("swa_lr", 1e-2)),
        #         swa_epoch_start=self.config["swa_epoch_start"][level],
        #     )
        # ]
        # TODO: Change monitoring variable name
        callbacks += [
            EarlyStopping(
                monitor=monitor,
                patience=self.config["patience"],
                mode="min" if monitor == "Loss" else "max",
            )
        ]

        strategy = self.strategy
        if self.strategy == "ddp":
            from pytorch_lightning.strategies import DDPStrategy

            strategy = DDPStrategy(find_unused_parameters=False)

        trainer = Trainer(
            num_nodes=self.num_nodes,
            devices=self.devices,
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            # TODO: Decide whether to keep these parameters
            check_val_every_n_epoch=None,
            val_check_interval=100,
            log_every_n_steps=100,
            strategy=strategy,
            deterministic=True,
            callbacks=callbacks,
            profiler="simple",
            enable_checkpointing=True if level == 0 else False,
            default_root_dir=self.dir_path,
            # gradient_clip_val=5,
            # precision=16,
        )
        return trainer

    def train_level(
        self, level: int, train_data: tuple, valid_data: tuple
    ) -> tuple[csr_matrix, tuple[Tensor], tuple[Tensor]]:
        """train levels from top to bottom.

        Args:
            level: current level
            train_data: (x, y)
            valid_data: (x, y)

        Returns:

        """
        train_x, train_y = train_data
        valid_x, valid_y = valid_data

        best_model_path = self.get_best_model_path(level)
        cluster_path = self.get_cluster_path(level)

        if level == 0:
            # wait for the clustering process for the first level to finish
            while not cluster_path.exists():
                time.sleep(15)
            nodes = np.load(cluster_path, allow_pickle=True)
            # each y has been mapped to the node indices of its parent
            train_y, valid_y = self.label2node(nodes, train_y, valid_y)
            # regard each internal nodes as a "labels"
            num_nodes = len(nodes)

            # trainer
            trainer = self.configure_trainer(level)

            # model
            if best_model_path.exists():
                # load existing best model
                if trainer.is_global_zero:
                    logger.info(f"Best model loaded from {best_model_path}")
                model = BaseModel.load_from_checkpoint(best_model_path, top_k=self.top_k)
            else:
                # train & valid dataloaders for training
                train_dataloader = DataLoader(
                    MultiLabelDataset(train_x, train_y),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    shuffle=True,
                )
                valid_dataloader = DataLoader(
                    MultiLabelDataset(valid_x, valid_y),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                )

                # model
                model = BaseModel(
                    network="AttentionRNN",
                    network_config=self.config["network_config"],
                    embed_vecs=self.embed_vecs,
                    optimizer=self.optimizer,
                    num_labels=num_nodes,
                    metrics=self.metrics,
                    top_k=self.top_k,
                    loss_fn=self.config["loss_fn"],
                    optimizer_params=self.config.get("optimizer_config"),
                    swa_epoch_start=self.config["swa_epoch_start"][level],
                )
                if GLOBAL_RANK == 0:
                    logger.info(f"Training level-{level}. Number of labels: {num_nodes}")
                trainer.fit(model, train_dataloader, valid_dataloader)
                # torch.cuda.empty_cache()
                if GLOBAL_RANK == 0:
                    logger.info(f"Best last loaded from {best_model_path}")
                # FIXME: I met a bug while experimenting with ModelCheckpoint
                trainer.strategy.barrier()
                model = BaseModel.load_from_checkpoint(best_model_path)
                # TODO: figure out why
                model.optimizer = None
            if GLOBAL_RANK == 0:
                logger.info(f"Finish Training Level-{level}")

            # Utilize single GPU to predict
            trainer = Trainer(
                num_nodes=1,
                devices=1,
                accelerator=self.accelerator,
            )
            print(f"trainer.is_global_zero: {GLOBAL_RANK} {trainer.is_global_zero}")
            if GLOBAL_RANK == 0:
                logger.info(
                    f"Generating predictions for Level-{level + 1}. "
                    f"Number of possible predictions: {num_nodes}. Top k: {self.top_k}"
                )
            # train & valid dataloaders for prediction (without labels)
            train_dataloader = DataLoader(
                MultiLabelDataset(train_x),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            valid_dataloader = DataLoader(
                MultiLabelDataset(valid_x),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

            return train_y, trainer.predict(model, train_dataloader), trainer.predict(model, valid_dataloader)
        # non-root nodes
        else:
            # returned labels have been clustered into nodes (groups)
            train_node_y, train_node_pred, valid_node_pred = self.train_level(level - 1, train_data, valid_data)

            # shape of node_pred: (n, 2, ~batch_size, top_k). n is floor(num_x / batch_size)
            # new shape: (2, num_x, top_k)
            train_node_score_pred, train_node_y_pred = map(torch.vstack, list(zip(*train_node_pred)))
            valid_node_score_pred, valid_node_y_pred = map(torch.vstack, list(zip(*valid_node_pred)))

            torch.cuda.empty_cache()
            if GLOBAL_RANK == 0:
                logger.info("Getting Candidates")
            node_candidates = np.empty((len(train_x), self.top_k), dtype=np.int32)

            prog = tqdm(train_node_y_pred, leave=False, desc="Parents") if GLOBAL_RANK == 0 else train_node_y_pred
            for i, ys in enumerate(prog):
                # true nodes/labels are positive
                positive = set(train_node_y.indices[train_node_y.indptr[i] : train_node_y.indptr[i + 1]])
                # Regard positive nodes and predicted training nodes that are not in positive as candidates
                # until reaching top_k if the number of positive labels is less than top_k.
                if len(positive) <= self.top_k:
                    candidates = positive
                    for y in ys:
                        if len(candidates) == self.top_k:
                            break
                        candidates |= {y}
                    candidates = list(candidates)
                # Regard positive (true) label as candidates iff they appear in the predicted labels
                # if the number of positive labels is more than top_k. If candidates are not of length top_k
                # add unseen predicted labels until reaching top_k.
                else:
                    candidates = set()
                    for y in ys:
                        if y in positive:
                            candidates |= {y}
                        if len(candidates) == self.top_k:
                            break
                    if len(candidates) < self.top_k:
                        candidates = (list(candidates) + list(positive - candidates))[: self.top_k]
                node_candidates[i] = np.asarray(candidates)

            # internal levels, i.e., levels that are neither the first nor last level
            if level < self.num_levels - 1:
                while not cluster_path.exists():
                    time.sleep(30)
                nodes = np.load(cluster_path, allow_pickle=True)
                # each y has been mapped to the node indices of its parent
                train_y, valid_y = self.label2node(nodes, train_y, valid_y)
                num_nodes = len(nodes)
                # mapping from the last level (level - 1) to the current level.
                # shape: (num_nodes of last level, cluster_size)
                parent2child = np.arange(num_nodes).reshape((-1, self.cluster_size))
            # the last level which connects to leaf nodes (true labels)
            else:
                num_nodes = train_y.shape[1]  # == reduce(lambda a, b: a + len(b), parent2child, 0)
                # mapping from the current nodes to leaf nodes.
                parent2child = np.load(self.get_cluster_path(level - 1), allow_pickle=True)

            assert reduce(lambda a, b: a + len(b), parent2child, 0) == num_nodes

            # trainer
            trainer = self.configure_trainer(level)

            trainer.strategy.barrier()

            print(f"{GLOBAL_RANK}: ")

            if best_model_path.exists():
                if trainer.is_global_zero:
                    logger.info(f"Best model loaded from {best_model_path}")
                model = PLTModel.load_from_checkpoint(best_model_path, top_k=self.top_k)
            else:
                # train & valid dataloaders for training
                train_dataloader = DataLoader(
                    PLTDataset(
                        train_x,
                        train_y,
                        num_nodes=num_nodes,
                        mapping=parent2child,
                        node_label=node_candidates,
                    ),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    shuffle=True,
                )
                valid_dataloader = DataLoader(
                    PLTDataset(
                        valid_x,
                        valid_y,
                        num_nodes=num_nodes,
                        mapping=parent2child,
                        node_label=valid_node_y_pred,
                        node_score=valid_node_score_pred,
                    ),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                )

                # model
                model = PLTModel(
                    network="FastAttentionRNN",
                    network_config=self.config["network_config"],
                    embed_vecs=self.embed_vecs,
                    optimizer=self.optimizer,
                    num_nodes=num_nodes,
                    metrics=self.config["monitor_metrics"],
                    top_k=self.top_k,
                    eval_metric=self.eval_metric,
                    loss_fn=self.config["loss_fn"],
                    optimizer_params=self.config.get("optimizer_config", None),
                    swa_epoch_start=self.config["swa_epoch_start"][level],
                )

                # initialize current layer with weights from last layer
                if GLOBAL_RANK == 0:
                    logger.info(f"Loading parameters of Level-{level} from Level-{level - 1}")
                # remove the name prefix in state_dict starting with "network"
                model.load_from_pretrained(torch.load(self.get_best_model_path(level - 1))["state_dict"])
                if GLOBAL_RANK == 0:
                    logger.info(
                        f"Training Level-{level}, "
                        f"Number of nodes: {num_nodes}, "
                        f"Number of candidates: {train_dataloader.dataset.num_candidates}"
                    )
                trainer.fit(model, train_dataloader, valid_dataloader)
                trainer.save_checkpoint(best_model_path)
                # FIXME: I met a bug while experimenting with ModelCheckpoint
                if GLOBAL_RANK == 0:
                    logger.info(f"Best model loaded from {best_model_path}")
                trainer.strategy.barrier()
                model = PLTModel.load_from_checkpoint(best_model_path)
            if GLOBAL_RANK == 0:
                logger.info(f"Finish training Level-{level}")
            # Utilize single GPU to predict
            trainer = Trainer(
                num_nodes=1,
                devices=1,
                accelerator=self.accelerator,
            )
            # end training if it is the last level
            if level == self.num_levels - 1:
                if GLOBAL_RANK == 0:
                    logger.info("Training process finished.")
                return
            if GLOBAL_RANK == 0:
                logger.info(
                    f"Generating predictions for Level-{level + 1}. "
                    f"Number of possible predictions: {num_nodes}, Top k: {self.top_k}"
                )

            # train & valid dataloaders for prediction
            train_dataloader = DataLoader(
                PLTDataset(
                    train_x,
                    num_nodes=num_nodes,
                    mapping=parent2child,
                    node_label=train_node_y_pred,
                    node_score=train_node_score_pred,
                ),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            valid_dataloader = DataLoader(
                PLTDataset(
                    valid_x,
                    num_nodes=num_nodes,
                    mapping=parent2child,
                    node_label=valid_node_y_pred,
                    node_score=valid_node_score_pred,
                ),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

            return train_y, trainer.predict(model, train_dataloader), trainer.predict(model, valid_dataloader)

    def predict_level(self, level, test_x, num_nodes, test_y=None):
        """
        If test_y provided, will log metrics on test dataset; otherwise, return predicted results.

        Args:
            level:
            test_x:
            num_nodes:
            test_y:

        Returns:

        """
        trainer = Trainer(
            devices=1,
            accelerator=self.accelerator,
        )

        # prediction starts from level 0
        if level == 0:
            model = BaseModel.load_from_checkpoint(self.get_best_model_path(level), top_k=self.top_k)

            test_dataloader = DataLoader(
                MultiLabelDataset(test_x),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

            if GLOBAL_RANK == 0:
                logger.info(f"Predicting Level-{level}, Top: {self.top_k}")
            node_pred = trainer.predict(model, test_dataloader)
            if GLOBAL_RANK == 0:
                logger.info(f"node_pred: {len(node_pred)}")
            node_score_pred, node_label_pred = map(torch.vstack, list(zip(*node_pred)))

            return node_score_pred, node_label_pred
        else:
            # last level
            if level == self.num_levels - 1:
                nodes = np.load(self.dir_path / f"Level-{len(self.levels)}-{level - 1}.npy", allow_pickle=True)
            else:
                nodes = np.arange(num_nodes).reshape((-1, self.cluster_size))
            node_score_pred, node_label_pred = self.predict_level(level - 1, test_x, len(nodes))
            torch.cuda.empty_cache()

            model = PLTModel.load_from_checkpoint(self.get_best_model_path(level), top_k=self.top_k)

            test_dataloader = DataLoader(
                PLTDataset(
                    test_x,
                    None if level < self.num_levels - 1 else test_y,
                    num_nodes=self.num_labels,
                    mapping=nodes,
                    node_label=node_label_pred,
                    node_score=node_score_pred,
                ),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

            if level == self.num_levels - 1 and test_y is not None:
                trainer.test(model, test_dataloader)
                return

            if GLOBAL_RANK == 0:
                logger.info(f"Predicting Level-{level}, Top: {self.top_k}")
            node_pred = trainer.predict(model, test_dataloader)
            if GLOBAL_RANK == 0:
                logger.info(f"node_pred: {len(node_pred)}")
            node_score_pred, node_label_pred = map(torch.vstack, list(zip(*node_pred)))

            return node_score_pred, node_label_pred

    def fit(self, datasets):
        """fit model
        Args:
            datasets: dict of train, valid, test
        """
        if self.get_best_model_path(self.num_levels - 1).exists():
            return

        train_data_full = datasets["train_full"]
        train_sparse_x = datasets["train_sparse_x"]
        # sparse training labels
        # TODO: remove workaround
        train_sparse_y_full = self.mlb.transform((i["label"] for i in train_data_full))
        cluster_process = Process(
            target=build_shallow_and_wide_plt,
            args=(train_sparse_x, train_sparse_y_full, self.levels, self.cluster_size, self.dir_path),
        )

        # TODO: remove workaround
        # TODO: we assume <unk> is 0. Remove this assumption or not?
        train_x = list(
            map(
                lambda x: torch.IntTensor([self.word_dict[i] for i in x]),
                [i["text"][: self.config["max_seq_length"]] for i in datasets["train"]],
            )
        )
        valid_x = list(
            map(
                lambda x: torch.IntTensor([self.word_dict[i] for i in x]),
                [i["text"][: self.config["max_seq_length"]] for i in datasets["val"]],
            )
        )
        # pad the first entry to be of length 500 if necessary
        train_x[0] = torch.cat(
            (
                train_x[0],
                torch.tensor(0, dtype=torch.long).repeat(self.config["max_seq_length"] - train_x[0].shape[0]),
            )
        )
        valid_x[0] = torch.cat(
            (
                valid_x[0],
                torch.tensor(0, dtype=torch.long).repeat(self.config["max_seq_length"] - valid_x[0].shape[0]),
            )
        )
        train_x = pad_sequence(train_x, batch_first=True)
        valid_x = pad_sequence(valid_x, batch_first=True)

        train_data = (train_x, self.mlb.transform((i["label"] for i in datasets["train"])))
        valid_data = (valid_x, self.mlb.transform((i["label"] for i in datasets["val"])))

        # only do clustering on the main process
        if GLOBAL_RANK == 0:
            try:
                cluster_process.start()
                self.train_level(self.num_levels - 1, train_data, valid_data)
                cluster_process.join()
            finally:
                # TODO: How to close process properly?
                cluster_process.terminate()
                cluster_process.close()
        else:
            self.train_level(self.num_levels - 1, train_data, valid_data)

        if dist.is_initialized():
            dist.destroy_process_group()
            torch.cuda.empty_cache()

    @is_global_zero
    def test(self, dataset):
        # why we want to test on a single GPU?
        # https://lightning.ai/docs/pytorch/stable/common/evaluation_intermediate.html
        if GLOBAL_RANK != 0:
            return

        test_x = list(
            map(
                lambda x: torch.tensor([self.word_dict[i] for i in x], dtype=torch.int),
                [i["text"][: self.config["max_seq_length"]] for i in dataset],
            )
        )
        test_x[0] = torch.cat(
            (
                test_x[0],
                torch.tensor(0, dtype=torch.long).repeat(self.config["max_seq_length"] - test_x[0].shape[0]),
            )
        )
        test_x = pad_sequence(test_x, batch_first=True)
        test_y = self.mlb.transform((i["label"] for i in dataset))
        logger.info("Start predicting process.")
        self.predict_level(self.num_levels - 1, test_x, self.num_labels, test_y)
        logger.info("Predicting process finishes.")

    def get_best_model_path(self, level: int) -> Path:
        return self.dir_path / f"Model-Level-{level}.ckpt"

    def get_cluster_path(self, level: int) -> Path:
        return self.dir_path / f"Level-{len(self.levels)}-{level}.npy"


def build_shallow_and_wide_plt(
    sparse_x,
    sparse_y: np.ndarray,
    levels: list[int],
    cluster_size: int,
    cluster_path: Path,
):
    """Build shallow and wide Probabilistic Label Tree(PLT).

    Args:
        sparse_x:
        sparse_y:
        levels:
        cluster_size:
        cluster_path:

    """
    if GLOBAL_RANK != 0:
        return
    logger.info(f"Number of levels: {len(levels) + 1}")
    logger.info(f"Internal level at depth: {levels}")
    logger.info(f"Cluster size: {cluster_size}")

    # check if the last level of clusters exist
    # we assume any clusters whose numbers are smaller than the cluster found exist
    if (cluster_path / f"Level-{len(levels)}-{len(levels) - 1}.npy").exists():
        logger.info("Clustering has finished in previous runs")
        return
    logger.info("Start clustering")

    # create directory for storing clusters
    cluster_path.mkdir(parents=True, exist_ok=True)

    # label representations
    label_repr = normalize(csr_matrix(sparse_y.T) @ csc_matrix(sparse_x))

    # number of nodes in each level
    levels = [2**x for x in levels]
    q = None
    # start from the latest cluster if previous clustering results found
    # We assume any clusters whose numbers are smaller than the cluster found exist

    for level in range(len(levels) - 1, -1, -1):
        if (cluster_path / f"Level-{len(levels)}-{level}.npy").exists():
            logger.info(f"load Level-{level} from existing file.")
            q = np.load(cluster_path / f"Level-{len(levels)}-{level}.npy", allow_pickle=True).tolist()
            break
    # start from the root node if no previous clustering results found
    if q is None:
        # the label indices stored in each node
        q = [np.arange(label_repr.shape[0])]
    # train leaf nodes
    while q:
        # the sum of the number of label indices in each node should be equal to the number of labels
        assert sum(len(idx_in_node) for idx_in_node in q) == label_repr.shape[0]
        if len(q) in levels:
            level = levels.index(len(q))
            logger.info(f"Have clustered {len(q)} nodes. Finish clustering Level-{level}")
            np.save(cluster_path / f"Level-{len(levels)}-{level}.npy", np.asarray(q, dtype=object))
        else:
            logger.info(f"Have clustered {len(q)} nodes")
        next_q = []
        for idx_in_node in q:
            if len(idx_in_node) > cluster_size:
                c0_labels, c1_labels = _kmeans(idx_in_node, label_repr)
                # kmeans = KMeans(2, random_state=42, max_iter=300, verbose=True).fit(label_repr[idx_in_node])
                # c0_idx, c1_idx = np.nonzero(kmeans.labels_ == 0), np.nonzero(kmeans.labels_ == 1)
                # c0_labels, c1_labels = idx_in_node[c0_idx], idx_in_node[c1_idx]
                next_q += [c0_labels, c1_labels]
        q = next_q
        logger.info("Finish clustering")


def _kmeans(idx_in_node: np.ndarray, label_repr: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """A variant of KMeans implemented in AttentionXML. Its main differences with sklearn.KMeans are:
    1. the distance metric is cosine similarity as all label representations are normalized.
    2. the end-of-loop criterion is the difference between the new and old average in-cluster distance to centroids.
    Possible drawbacks:
        Random initialization.
        cluster_size matters.
    """
    # Some possible hyperparameters
    tol = 1e-4
    if tol <= 0 or tol > 1:
        raise ValueError(f"tol should be a positive number that is less than 1, got {tol} instead.")

    # the corresponding label representations in the node
    tgt_repr = label_repr[idx_in_node]

    # the number of leaf labels in the node
    n = len(idx_in_node)

    # randomly choose two points as initial centroids
    centroids = tgt_repr[np.random.choice(n, 2, replace=False)].toarray()

    # initialize distances (cosine similarity)
    old_dist = -2.0
    new_dist = -1.0

    # "c" denotes clusters
    c0_idx = None
    c1_idx = None

    while new_dist - old_dist >= tol:
        # each points' distance (cosine similarity) to the two centroids
        dist = tgt_repr @ centroids.T  # shape: (n, 2)

        # generate clusters
        # let a = dist[:, 1] - dist[:, 0], the larger the element in a is, the closer the point is to the c1
        c_idx = np.argsort(dist[:, 1] - dist[:, 0])
        c0_idx = c_idx[: n // 2]
        c1_idx = c_idx[n // 2 :]

        # update distances
        # the distance is the average in-cluster distance to the centroids
        old_dist = new_dist
        new_dist = (dist[c0_idx, 0].sum() + dist[c1_idx, 1].sum()) / n

        # update centroids
        # the new centroid is the average of the points in the cluster
        centroids = normalize(
            np.asarray(
                [
                    np.squeeze(np.asarray(tgt_repr[c0_idx].sum(axis=0))),
                    np.squeeze(np.asarray(tgt_repr[c1_idx].sum(axis=0))),
                ]
            )
        )
    return idx_in_node[c0_idx], idx_in_node[c1_idx]
