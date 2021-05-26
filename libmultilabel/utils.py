import copy
import json
import logging
import os
import time
import torch

import numpy as np


class ArgDict(dict):
    def __init__(self, *args, **kwargs):
        super(ArgDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elasped time."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total


def pad_sequence(sequences, batch_first=False, padding_value=0.0, max_len=None):
    # type: (List[Tensor], bool, float) -> Tensor
    r"""Pad a list of variable length Tensors with ``padding_value``

    Modified from pytorch.nn.utils.rnn.pad_sequence to support length specification
    https://pytorch.org/docs/stable/_modules/torch/nn/utils/rnn.html#pad_sequence

    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.
        max_len (int, optional): length to pad if given, or calculate from
			sequences otherwise. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def dump_log(config, metrics, split):
    """Write log including config and the evaluation scores.

    Args:
        config (dict): config to save
        metrics (dict): metric and scores in dictionary format
        split (str): val or test
    """
    log_path = os.path.join(config.result_dir, config.run_name, 'logs.json')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.isfile(log_path):
        with open(log_path) as fp:
            result = json.load(fp)
    else:
        config_to_save = copy.deepcopy(config.__dict__)
        del config_to_save['device']
        result = {'config': config_to_save}

    if split in result:
        result[split].append(metrics)
    else:
        result[split] = [metrics]
    with open(log_path, 'w') as fp:
        json.dump(result, fp)

    logging.info(f'Finish writing log to {log_path}.')


def save_top_k_predictions(class_names, y_pred, predict_out_path, k=100):
    """Save top k predictions to the predict_out_path. The format of this file is:
    <label1>:<value1> <label2>:<value2> ...

    Args:
        class_names (list): list of class names
        y_pred (ndarray): predictions (shape: number of samples * number of classes)
        k (int): number of classes considered as the correct labels
    """
    assert predict_out_path, "Please specify the output path to the prediction results."

    logging.info(f'Save top {k} predictions to {predict_out_path}.')
    with open(predict_out_path, 'w') as fp:
        for pred in y_pred:
            label_ids = np.argsort(-pred).tolist()[:k]
            out_str = ' '.join([f'{class_names[i]}:{pred[i]:.4}' for i in label_ids])
            fp.write(out_str+'\n')


def set_seed(seed):
    """Set seeds for numpy and pytorch."""
    if seed is not None:
        if seed >= 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.set_deterministic(True)
            torch.backends.cudnn.benchmark = False
        else:
            logging.warning(
                f'the random seed should be a non-negative integer')


def init_device(use_cpu=False):
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
