import logging
import os
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint, warning_cache
from pytorch_lightning.utilities.seed import seed_everything
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch import Tensor

from ..nn import networks
from ..nn.model import Model


def init_device(use_cpu=False):
    """Initialize device to CPU if `use_cpu` is set to True otherwise GPU.

    Args:
        use_cpu (bool, optional): Whether to use CPU or not. Defaults to False.

    Returns:
        torch.device: One of cuda or cpu.
    """

    if not use_cpu and torch.cuda.is_available():
        # Set a debug environment variable CUBLAS_WORKSPACE_CONFIG to ":16:8" (may limit overall performance) or ":4096:8" (will increase library footprint in GPU memory by approximately 24MiB).
        # https://docs.nvidia.com/cuda/cublas/index.html
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy("file_system")
    logging.info(f"Using device: {device}")
    return device


def init_model(
    model_name,
    network_config,
    classes,
    word_dict=None,
    embed_vecs=None,
    init_weight=None,
    log_path=None,
    learning_rate=0.0001,
    optimizer="adam",
    momentum=0.9,
    weight_decay=0,
    lr_scheduler=None,
    scheduler_config=None,
    val_metric=None,
    metric_threshold=0.5,
    monitor_metrics=None,
    multiclass=False,
    loss_function="binary_cross_entropy_with_logits",
    silent=False,
    save_k_predictions=0,
):
    """Initialize a `Model` class for initializing and training a neural network.

    Args:
        model_name (str): Model to be used such as KimCNN.
        network_config (dict): Configuration for defining the network.
        classes (list): List of class names.
        word_dict (torchtext.vocab.Vocab, optional): A vocab object for word tokenizer to
            map tokens to indices. Defaults to None.
        embed_vecs (torch.Tensor, optional): The pre-trained word vectors of shape
            (vocab_size, embed_dim). Defaults to None.
        init_weight (str): Weight initialization method from `torch.nn.init`.
            For example, the `init_weight` of `torch.nn.init.kaiming_uniform_`
            is `kaiming_uniform`. Defaults to None.
        log_path (str): Path to a directory holding the log files and models.
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 0.0001.
        optimizer (str, optional): Optimizer name (i.e., sgd, adam, or adamw). Defaults to 'adam'.
        momentum (float, optional): Momentum factor for SGD only. Defaults to 0.9.
        weight_decay (int, optional): Weight decay factor. Defaults to 0.
        lr_scheduler (str, optional): Name of the learning rate scheduler. Defaults to None.
        scheduler_config (dict, optional): The configuration for learning rate scheduler. Defaults to None.
        val_metric (str, optional): The metric to select the best model for testing. Used by some of the schedulers. Defaults to None.
        metric_threshold (float, optional): The decision value threshold over which a label is predicted as positive. Defaults to 0.5.
        monitor_metrics (list, optional): Metrics to monitor while validating. Defaults to None.
        multiclass (bool, optional): Enable multiclass mode. Defaults to False.
        silent (bool, optional): Enable silent mode. Defaults to False.
        loss_function (str, optional): Loss function name (i.e., binary_cross_entropy_with_logits,
            cross_entropy). Defaults to 'binary_cross_entropy_with_logits'.
        save_k_predictions (int, optional): Save top k predictions on test set. Defaults to 0.

    Returns:
        Model: A class that implements `MultiLabelModel` for initializing and training a neural network.
    """

    try:
        network = getattr(networks, model_name)(embed_vecs=embed_vecs, num_classes=len(classes), **dict(network_config))
    except:
        raise AttributeError(f"Failed to initialize {model_name}.")

    if init_weight is not None:
        init_weight = networks.get_init_weight_func(init_weight=init_weight)
        network.apply(init_weight)

    model = Model(
        classes=classes,
        word_dict=word_dict,
        embed_vecs=embed_vecs,
        network=network,
        log_path=log_path,
        learning_rate=learning_rate,
        optimizer=optimizer,
        momentum=momentum,
        weight_decay=weight_decay,
        lr_scheduler=lr_scheduler,
        scheduler_config=scheduler_config,
        val_metric=val_metric,
        metric_threshold=metric_threshold,
        monitor_metrics=monitor_metrics,
        multiclass=multiclass,
        loss_function=loss_function,
        silent=silent,
        save_k_predictions=save_k_predictions,
    )
    return model


def init_trainer(
    checkpoint_dir,
    epochs=10000,
    patience=5,
    early_stopping_metric="P@1",
    val_metric="P@1",
    silent=False,
    use_cpu=False,
    limit_train_batches=1.0,
    limit_val_batches=1.0,
    limit_test_batches=1.0,
    search_params=False,
    save_checkpoints=True,
):
    """Initialize a torch lightning trainer.

    Args:
        checkpoint_dir (str): Directory for saving models and log.
        epochs (int): Number of epochs to train. Defaults to 10000.
        patience (int): Number of epochs to wait for improvement before early stopping. Defaults to 5.
        early_stopping_metric (str): The metric to monitor for early stopping. Defaults to 'P@1'.
        val_metric (str): The metric to select the best model for testing. Defaults to 'P@1'.
        silent (bool): Enable silent mode. Defaults to False.
        use_cpu (bool): Disable CUDA. Defaults to False.
        limit_train_batches (Union[int, float]): Percentage of training dataset to use. Defaults to 1.0.
        limit_val_batches (Union[int, float]): Percentage of validation dataset to use. Defaults to 1.0.
        limit_test_batches (Union[int, float]): Percentage of test dataset to use. Defaults to 1.0.
        search_params (bool): Enable pytorch-lightning trainer to report the results to ray tune
            on validation end during hyperparameter search. Defaults to False.
        save_checkpoints (bool): Whether to save the last and the best checkpoint or not. Defaults to True.

    Returns:
        pl.Trainer: A torch lightning trainer.
    """

    # The value of `mode` equals to 'min' only when the metric is 'Loss'
    # because now for other supported metrics such as F1 or Precision, we maximize them in the training process.
    # But if in the future, we further support other metrics that need to be minimized,
    # we may need a dictionary that records a metric-mode mapping for a better practice.
    # Set strict to False to prevent EarlyStopping from crashing the training if no validation data are provided
    early_stopping_callback = EarlyStopping(
        patience=patience,
        monitor=early_stopping_metric,
        mode="min" if early_stopping_metric == "Loss" else "max",
        strict=False,
    )
    callbacks = [early_stopping_callback]
    if save_checkpoints:
        callbacks += [
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="best_model",
                save_last=True,
                save_top_k=1,
                monitor=val_metric,
                mode="min" if val_metric == "Loss" else "max",
            )
        ]
    if search_params:
        from ray.tune.integration.pytorch_lightning import TuneReportCallback

        callbacks += [TuneReportCallback({f"val_{val_metric}": val_metric}, on="validation_end")]

    trainer = pl.Trainer(
        logger=False,
        num_sanity_val_steps=0,
        accelerator="cpu" if use_cpu else "gpu",
        devices=None if use_cpu else 1,
        enable_progress_bar=False if silent else True,
        max_epochs=epochs,
        callbacks=callbacks,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        deterministic="warn",
    )
    return trainer


def set_seed(seed):
    """Set seeds for numpy and pytorch.

    Args:
        seed (int): Random seed.
    """

    if seed is not None:
        if seed >= 0:
            seed_everything(seed=seed, workers=True)
        else:
            logging.warning("the random seed should be a non-negative integer")


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        monitor: str,
        dirpath=None,
        filename: str = None,
        save_weights_only: bool = False,
        verbose: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
    ):
        """Cache the best model during validation and save it at the end of training"""
        if monitor is None:
            raise ValueError("Monitor has to be set")

        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_top_k=1,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
        )
        # As we only want the top-1 model, these values are equal.
        self.best_model_score = self.kth_value
        self.best_state_dict = {}
        # For compatibility reasons, we use 'saved' instead of 'cached'.
        self._last_epoch_saved = 0
        self.best_monitor_candidates = None

        # variables of our interest initialized in the parent class
        # self.monitor = monitor
        # self.verbose = verbose
        # self.save_weights_only = save_weights_only
        # self.current_score = None # Tensor
        # self.best_model_path = ""
        # self._last_global_step_saved = 0

    @property
    def state_key(self) -> str:
        return self._generate_state_key(monitor=self.monitor, mode=self.mode)

    def on_train_start(self, trainer, pl_module):
        return

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        return

    def on_train_end(self, trainer, pl_module):
        """Save the checkpoint with the best validation result at the end of training."""
        self.best_model_path = self._get_metric_interpolated_filepath_name(self.best_monitor_candidates, trainer)
        if self.save_weights_only:
            checkpoint = self.best_state_dict
        else:
            checkpoint = trainer._checkpoint_connector.dump_checkpoint(weights_only=False)
            checkpoint["state_dict"] = self.best_state_dict
            checkpoint["epoch"] = self._last_epoch_saved
            checkpoint["global_step"] = self._last_global_step_saved
        trainer.strategy.save_checkpoint(checkpoint, self.best_model_path)
        trainer.strategy.barrier("Trainer.save_checkpoint")

    def on_validation_end(self, trainer, pl_module):
        """Cache the checkpoint with the best validation result at the end of validation."""
        if not self._should_skip_saving_checkpoint(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self.monitor not in monitor_candidates:
                m = (
                    f"`ModelCheckpoint(monitor={self.monitor!r})` could not find the monitored key in the returned"
                    f" metrics: {list(monitor_candidates)}."
                    f" HINT: Did you call `log({self.monitor!r}, value)` in the `LightningModule`?"
                )
                if trainer.fit_loop.epoch_loop.val_loop._has_run:
                    raise MisconfigurationException(m)
                warning_cache.warn(m)
            self.current_score = monitor_candidates.get(self.monitor)
            assert self.current_score is not None

            monitor_op = {"min": torch.lt, "max": torch.gt}[self.mode]
            should_update_best_and_save = monitor_op(self.current_score, self.best_model_score)

            # If using multiple devices, make sure all processes are unanimous on the decision.
            should_update_best_and_save = trainer.strategy.reduce_boolean_decision(bool(should_update_best_and_save))

            if should_update_best_and_save:
                # do not save nan, replace with +/- inf
                if isinstance(self.current_score, Tensor) and torch.isnan(self.current_score):
                    self.current_score = torch.tensor(
                        float("inf" if self.mode == "min" else "-inf"), device=self.current_score.device
                    )

                self.best_model_score = self.current_score

                if self.verbose:
                    rank_zero_info(
                        f"Epoch {self._last_epoch_saved:d}, global step {self._last_global_step_saved:d}: "
                        f"{repr(self.monitor)} reached {self.current_score:.5f}"
                    )

                # cache checkpoint
                state_dict = trainer._checkpoint_connector.dump_checkpoint(weights_only=True)["state_dict"]
                for k in state_dict:
                    self.best_state_dict[k] = state_dict[k].detach().cpu()

                self._last_epoch_saved = monitor_candidates["epoch"]
                self._last_global_step_saved = monitor_candidates["step"]
                self.best_monitor_candidates = monitor_candidates

                # skip notifying logger because their behaviors are not clear
            elif self.verbose:
                epoch = monitor_candidates["epoch"]
                step = monitor_candidates["step"]
                rank_zero_info(f"Epoch {epoch:d}, global step {step:d}: {repr(self.monitor)} was not the best")

    def state_dict(self):
        return {
            "monitor": self.monitor,
            "best_model_score": self.best_model_score,
            "best_model_path": self.best_model_path,
            "current_score": self.current_score,
            "dirpath": self.dirpath,
        }

    def load_state_dict(self, state_dict):
        dirpath_from_ckpt = state_dict.get("dirpath", self.dirpath)

        if self.dirpath == dirpath_from_ckpt:
            self.best_model_score = state_dict["best_model_score"]
        else:
            warnings.warn(
                f"The dirpath has changed from {repr(dirpath_from_ckpt)} to {repr(self.dirpath)},"
                " therefore `best_model_score`, `kth_best_model_path`, `kth_value`, `last_model_path` and"
                " `best_k_models` won't be reloaded. Only `best_model_path` will be reloaded."
            )

        self.best_model_path = state_dict["best_model_path"]
