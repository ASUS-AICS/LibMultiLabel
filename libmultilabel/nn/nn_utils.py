import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

from ..nn import networks
from ..nn.model import Model, BaseModel
from ..common_utils import GLOBAL_RANK


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
        if GLOBAL_RANK == 0:
            logging.info(f"Available GPUs: {torch.cuda.device_count()}")
        # Sets the internal precision of float32 matrix multiplications.
        # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device("cpu")
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy("file_system")
    if GLOBAL_RANK == 0:
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
    config=None,
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

    if config.model_name == "AttentionXML":
        model = BaseModel(
            network=config.model_name,
            network_config=config["network_config"],
            embed_vecs=embed_vecs,
            optimizer=config.optimizer,
            num_labels=len(classes),
            metrics=config.monitor_metrics,
            top_k=config.get("top_k", 100),
            loss_fn=config.loss_fn,
            optimizer_params=config.get("optimizer_config"),
            swa_epoch_start=config["swa_epoch_start"],
        )
    else:
        try:
            network = getattr(networks, model_name)(
                embed_vecs=embed_vecs, num_classes=len(classes), **dict(network_config)
            )
        except Exception as e:
            logging.error(e)
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
    config=None,
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
        model_name=None

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

    if config.model_name == "AttentionXML":
        dir_path = Path(config.dir_path) / f"{config.data_name}-{config.model_name}-{0}"
        callbacks = []
        monitor = "_".join(["valid", config["val_metric"].lower()])
        callbacks += [
            ModelCheckpoint(
                dirpath=dir_path,
                filename=f"Model",
                save_top_k=1,
                monitor=monitor,
                verbose=True,
                mode="max",
            )
        ]
        callbacks += [
            EarlyStopping(
                monitor=monitor,
                patience=config["patience"],
                mode="max",
            )
        ]
        trainer = pl.Trainer(
            num_nodes=config.num_nodes,
            devices=config.devices,
            max_epochs=config.max_epochs,
            accelerator=config.accelerator,
            # TODO: Decide whether to keep these parameters
            check_val_every_n_epoch=None,
            val_check_interval=100,
            log_every_n_steps=100,
            # strategy= "ddp" if ,
            deterministic=True,
            callbacks=callbacks,
            profiler="simple",
            default_root_dir=dir_path,
            # gradient_clip_val=5,
            # precision=16,
        )
    else:
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
    # Dongli Suggestion: Rename to setup_reproducibility
    # random seed if not specified
    Args:
        seed (int): Random seed.
    """

    if seed is not None:
        if seed >= 0:
            seed_everything(seed=seed, workers=True)
        else:
            logging.warning("the random seed should be a non-negative integer")


def is_global_zero(func):
    def wrapper(*args, **kwargs):
        if GLOBAL_RANK == 0:
            return func(*args, **kwargs)

    return wrapper
