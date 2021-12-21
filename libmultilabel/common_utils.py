import copy
import json
import logging
import os
import time

import numpy as np


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


def dump_log(log_path, metrics=None, split=None, config=None):
    """Write log including config and the evaluation scores.

    Args:
        log_path(str): path to log path
        metrics (dict): metric and scores in dictionary format, defaults to None
        split (str): val or test, defaults to None
        config (dict): config to save, defaults to None
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.isfile(log_path):
        with open(log_path) as fp:
            result = json.load(fp)
    else:
        result = dict()

    if config:
        config_to_save = copy.deepcopy(dict(config))
        config_to_save.pop('device', None)  # delete if device exists
        result['config'] = config_to_save
    if split and metrics:
        if split in result:
            result[split].append(metrics)
        else:
            result[split] = [metrics]

    with open(log_path, 'w') as fp:
        json.dump(result, fp)

    logging.info(f'Finish writing log to {log_path}.')


def argsort_top_k(vals, k, axis=-1):
    unsorted_top_k_idx = np.argpartition(vals, -k, axis=axis)[:, -k:]
    unsorted_top_k_scores = np.take_along_axis(
        vals, unsorted_top_k_idx, axis=axis)
    sorted_order = np.argsort(-unsorted_top_k_scores, axis=axis)
    sorted_top_k_idx = np.take_along_axis(
        unsorted_top_k_idx, sorted_order, axis=axis)
    return sorted_top_k_idx


class AttributeDict(dict):
    """AttributeDict is an extended dict that can access
    stored items as attributes.

    >>> ad = AttributeDict({'ans': 42})
    >>> ad.ans
    >>> 42
    """

    def __getattr__(self, key: str) -> any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f'Missing attribute "{key}"')

    def __setattr__(self, key: str, value: any) -> None:
        self[key] = value
