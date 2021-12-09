import re

import numpy as np
import scipy.sparse as sparse

__all__ = ['RPRecision',
           'Precision',
           'F1',
           'MetricCollection',
           'get_metrics',
           'tabulate_metrics']


class RPrecision:
    def __init__(self, top_k: int) -> None:
        self.top_k = top_k
        self.score = 0
        self.num_sample = 0

    def update(self, preds: np.ndarray, target: np.ndarray) -> None:
        assert preds.shape == target.shape
        top_k_ind = np.argpartition(preds, -self.top_k)[:, -self.top_k:]
        num_relevant = target[top_k_ind].sum(axis=-1)
        top_ks = np.full_like(preds, self.top_k)
        self.score += np.nan_to_num(
            num_relevant / np.min(top_ks, target.sum(axis=-1)),
            posinf=0.
        ).sum()
        self.num_sample += preds.shape[0]

    def compute(self) -> float:
        return self.score / self.num_sample


class Precision:
    def __init__(self, num_classes: int, average: str, top_k: int) -> None:
        self.top_k = top_k
        self.score = 0
        self.num_sample = 0

    def update(self, preds: np.ndarray, target: np.ndarray) -> None:
        assert preds.shape == target.shape
        top_k_ind = np.argpartition(preds, -self.top_k)[:, -self.top_k:]
        num_relevant = np.take_along_axis(target, top_k_ind, -1).sum()
        self.score += num_relevant / self.top_k
        self.num_sample += preds.shape[0]

    def compute(self) -> float:
        return self.score / self.num_sample


class F1:
    def __init__(self, num_classes: int, metric_threshold: float, average: str) -> None:
        self.num_classes = num_classes
        self.metric_threshold = metric_threshold
        if average not in {'macro', 'micro'}:
            raise ValueError('unsupported average')
        self.average = average
        self.tp = self.fp = self.fn = 0

    def update(self, preds: np.ndarray, target: np.ndarray) -> None:
        assert preds.shape == target.shape
        preds = preds > self.metric_threshold
        self.tp += np.logical_and(target == 1, preds == 1).sum(axis=0)
        self.fn += np.logical_and(target == 1, preds == 0).sum(axis=0)
        self.fp += np.logical_and(target == 0, preds == 1).sum(axis=0)

    def compute(self) -> float:
        if self.average == 'macro':
            return np.nansum(2*self.tp / (2*self.tp + self.fp + self.fn)) / self.num_classes
        if self.average == 'micro':
            return np.nan_to_num(2*np.sum(self.tp) / np.sum(2*self.tp + self.fp + self.fn))


class MetricCollection(dict):
    def __init__(self, metrics) -> None:
        self.metrics = metrics

    def update(self, preds: np.ndarray, target: np.ndarray) -> None:
        assert preds.shape == target.shape
        for metric in self.metrics.values():
            metric.update(preds, target)

    def compute(self) -> "dict[str, float]":
        ret = {}
        for name, metric in self.metrics.items():
            ret[name] = metric.compute()
        return ret


def get_metrics(metric_threshold: float, monitor_metrics: list, num_classes: int):
    """Get a collection of metrics by their names.

    Args:
        metric_threshold (float): The decision value threshold over which a label
        is predicted as positive.

        monitor_metrics (list): A list of strings naming the metrics.

        num_classes (int): The number of classes.

    Returns:
        MetricCollection: A metric collection of the list of metrics.
    """
    if monitor_metrics is None:
        monitor_metrics = []
    metrics = {
        'Micro-F1': F1(num_classes, metric_threshold, average='micro'),
        'Macro-F1': F1(num_classes, metric_threshold, average='macro'),
    }
    for metric in monitor_metrics:
        if re.match('P@\d+', metric):
            metrics[metric] = Precision(
                num_classes, average='samples', top_k=int(metric[2:]))
        elif re.match('RP@\d+', metric):
            metrics[metric] = RPrecision(top_k=int(metric[3:]))

        elif metric not in ['Micro-Precision', 'Micro-Recall', 'Micro-F1', 'Macro-F1', 'Another-Macro-F1']:
            raise ValueError(f'Invalid metric: {metric}')

    return MetricCollection(metrics)


def tabulate_metrics(metric_dict, split):
    msg = f'====== {split} dataset evaluation result =======\n'
    header = '|'.join([f'{k:^18}' for k in metric_dict.keys()])
    values = '|'.join([f'{x * 100:^18.4f}' if isinstance(x, (np.floating,
                      float)) else f'{x:^18}' for x in metric_dict.values()])
    msg += f"|{header}|\n|{'-----------------:|' * len(metric_dict)}\n|{values}|\n"
    return msg
