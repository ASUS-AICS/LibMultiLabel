import logging
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

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
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy('file_system')
    logging.info(f'Using device: {device}')
    return device


def init_model(model_name,
               network_config,
               classes,
               word_dict,
               init_weight=None,
               log_path=None,
               learning_rate=0.0001,
               optimizer='adam',
               momentum=0.9,
               weight_decay=0,
               metric_threshold=0.5,
               monitor_metrics=None,
               silent=False,
               save_k_predictions=0):
    """Initialize a `Model` class for initializing and training a neural network.

    Args:
        model_name (str): Model to be used such as KimCNN.
        network_config (dict): Configuration for defining the network.
        classes(list): List of class names.
        word_dict(torchtext.vocab.Vocab): A vocab object which maps tokens to indices.
        init_weight (str): Weight initialization method from `torch.nn.init`.
            For example, the `init_weight` of `torch.nn.init.kaiming_uniform_`
            is `kaiming_uniform`. Defaults to None.
        log_path (str): Path to a directory holding the log files and models.
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 0.0001.
        optimizer (str, optional): Optimizer name (i.e., sgd, adam, or adamw). Defaults to 'adam'.
        momentum (float, optional): Momentum factor for SGD only. Defaults to 0.9.
        weight_decay (int, optional): Weight decay factor. Defaults to 0.
        metric_threshold (float, optional): Threshold to monitor for metrics. Defaults to 0.5.
        monitor_metrics (list, optional): Metrics to monitor while validating. Defaults to None.
        silent (bool, optional): Enable silent mode. Defaults to False.
        save_k_predictions (int, optional): Save top k predictions on test set. Defaults to 0.

    Returns:
        Model: A class that implements `MultiLabelModel` for initializing and training a neural network.
    """
    network = getattr(networks, model_name)(
        embed_vecs=word_dict.vectors,
        num_classes=len(classes),
        **dict(network_config)
    )
    if init_weight is not None:
        init_weight = networks.get_init_weight_func(
            init_weight=init_weight)
        network.apply(init_weight)

    model = Model(
        classes=classes,
        word_dict=word_dict,
        network=network,
        log_path=log_path,
        learning_rate=learning_rate,
        optimizer=optimizer,
        momentum=momentum,
        weight_decay=weight_decay,
        metric_threshold=metric_threshold,
        monitor_metrics=monitor_metrics,
        silent=silent,
        save_k_predictions=save_k_predictions
    )
    return model


def init_trainer(checkpoint_dir,
                 epochs=10000,
                 patience=5,
                 mode='max',
                 val_metric='P@1',
                 silent=False,
                 use_cpu=False,
                 limit_train_batches=1.0,
                 limit_val_batches=1.0,
                 limit_test_batches=1.0,
                 search_params=False,
                 save_checkpoints=True):
    """Initialize a torch lightning trainer.

    Args:
        checkpoint_dir (str): Directory for saving models and log.
        epochs (int): Number of epochs to train. Defaults to 10000.
        patience (int): Number of epochs to wait for improvement before early stopping. Defaults to 5.
        mode (str): One of [min, max]. Decides whether the val_metric is minimizing or maximizing.
        val_metric (str): The metric to monitor for early stopping. Defaults to 'P@1'.
        silent (bool): Enable silent mode. Defaults to False.
        use_cpu (bool): Disable CUDA. Defaults to False.
        limit_train_batches(Union[int, float]): Percentage of training dataset to use. Defaults to 1.0.
        limit_val_batches(Union[int, float]): Percentage of validation dataset to use. Defaults to 1.0.
        limit_test_batches(Union[int, float]): Percentage of test dataset to use. Defaults to 1.0.
        search_params (bool): Enable pytorch-lightning trainer to report the results to ray tune
            on validation end during hyperparameter search. Defaults to False.
        save_checkpoints (bool): Whether to save the last and the best checkpoint or not. Defaults to True.
    Returns:
        pl.Trainer: A torch lightning trainer.
    """

    callbacks = [EarlyStopping(patience=patience, monitor=val_metric, mode=mode)]
    if save_checkpoints:
        callbacks += [ModelCheckpoint(dirpath=checkpoint_dir, filename='best_model',
                                      save_last=True, save_top_k=1,
                                      monitor=val_metric, mode=mode)]
    if search_params:
        from ray.tune.integration.pytorch_lightning import TuneReportCallback
        callbacks += [TuneReportCallback({f'val_{val_metric}': val_metric}, on="validation_end")]

    trainer = pl.Trainer(logger=False, num_sanity_val_steps=0,
                         gpus=0 if use_cpu else 1,
                         progress_bar_refresh_rate=0 if silent else 1,
                         max_epochs=epochs,
                         callbacks=callbacks,
                         limit_train_batches=limit_train_batches,
                         limit_val_batches=limit_val_batches,
                         limit_test_batches=limit_test_batches)
    return trainer


def set_seed(seed):
    """Set seeds for numpy and pytorch.

    Args:
        seed (int): Random seed.
    """

    if seed is not None:
        if seed >= 0:
            seed_everything(seed=seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        else:
            logging.warning('the random seed should be a non-negative integer')
