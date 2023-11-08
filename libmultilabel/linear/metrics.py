from __future__ import annotations

import re

import numpy as np

__all__ = ["get_metrics", "compute_metrics", "tabulate_metrics", "MetricCollection"]


def _sorted_top_k_idx(preds: np.ndarray, k: int) -> np.ndarray:
    """Sorts the top k indices in O(n + k log k) time.
    The sorting order is ascending to be consistent with np.sort.
    This means the last element is the largest, the first element is the kth largest.
    """
    top_k_idx = np.argpartition(preds, -k)[:, -k:]
    argsort_top_k = np.argsort(np.take_along_axis(preds, top_k_idx, axis=-1))
    return np.take_along_axis(top_k_idx, argsort_top_k, axis=-1)


def _DCG(preds: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    """Compute the discounted cumulative gains (DCG)."""
    # Self-implemented dcg is used here. scikit-learn's implementation has
    # an average time complexity O(NlogN) as it directly applies quicksort
    # to preds regardless of k. Here the average time complexity is reduced to
    # O(N + klogk) by first partitioning off the k largest elements and then
    # applying quicksort to the subarray.
    row_idx = np.arange(preds.shape[0])[:, np.newaxis]
    preds_unsorted_idx = np.argpartition(preds, -k)[:, -k:]
    preds_sorted_idx = preds_unsorted_idx[row_idx, np.argsort(-preds[row_idx, preds_unsorted_idx])]

    # target sorted by the top k preds in non-increasing order
    gains = target[row_idx, preds_sorted_idx]

    # the discount factor
    discount = 1 / (np.log2(np.arange(gains.shape[1]) + 2))

    # get the sum product over the last axis of gains and discount
    dcg = gains.dot(discount)
    return dcg


def _DCG_argsort(argsort_preds: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    top_k_idx = argsort_preds[:, -k:][:, ::-1]
    gains = np.take_along_axis(target, top_k_idx, axis=-1)
    discount = 1 / (np.log2(np.arange(gains.shape[1]) + 2))
    dcg = gains.dot(discount)
    return dcg


def _IDCG(target: np.ndarray, k: int) -> np.ndarray:
    labels = target.sum(axis=1, dtype="i")
    max_labels = np.max(labels)
    discount = 1 / (np.log2(np.arange(max_labels) + 2))
    gains = discount.cumsum()
    indices = np.minimum(labels - 1, k - 1)
    return gains[indices]


def _batchDCG_argsort(argsort_preds: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    top_k_idx = argsort_preds[:, -k:][:, ::-1]
    gains = np.take_along_axis(target, top_k_idx, axis=-1)
    discount = 1 / (np.log2(np.arange(gains.shape[1]) + 2))
    dcg = (gains * discount).cumsum(axis=1)
    return dcg


def _batchIDCG(target: np.ndarray, k: int) -> np.ndarray:
    labels = target.sum(axis=1, dtype="i")
    max_labels = np.max(labels)
    discount = 1 / (np.log2(np.arange(max_labels) + 2))
    gains = discount.cumsum()
    indices = np.tile(np.arange(k), target.shape[0]).reshape(-1, k)
    indices = np.minimum(labels[:, np.newaxis] - 1, indices)
    return gains[indices]


class NDCG:
    def __init__(self, top_k: int):
        """Compute the normalized DCG@k (nDCG@k).

        Args:
            top_k: Consider only the top k elements for each query.
        """
        _check_top_k(top_k)

        self.top_k = top_k
        self.score = 0
        self.num_sample = 0

    def update(self, preds: np.ndarray, target: np.ndarray):
        assert preds.shape == target.shape  # (batch_size, num_classes)

        # DCG
        dcgs = _DCG(preds, target, self.top_k)
        # ideal DCG
        idcgs = _DCG(target, target, self.top_k)
        # normalized DCG
        ndcg_score = dcgs / idcgs

        # deal with instances with all 0 labels and add up the results
        self.score += np.nan_to_num(ndcg_score, nan=0.0).sum()
        # remember the total number of instances
        self.num_sample += preds.shape[0]

    def update_argsort(self, argsort_preds: np.ndarray, target: np.ndarray):
        dcgs = _DCG_argsort(argsort_preds, target, self.top_k)
        idcgs = _IDCG(target, self.top_k)
        ndcg_score = dcgs / idcgs
        self.score += np.nan_to_num(ndcg_score, nan=0.0).sum()
        self.num_sample += argsort_preds.shape[0]

    def compute(self) -> float:
        return self.score / self.num_sample

    def reset(self):
        self.score = 0
        self.num_sample = 0


class BatchNDCG:
    def __init__(self, top_k: int):
        """Compute the normalized DCG@k (nDCG@k) for k = 1 to top_k.

        Args:
            top_k: Consider up to the top k elements for each query.
        """
        _check_top_k(top_k)

        self.top_k = top_k
        self.score = np.zeros(top_k)
        self.num_sample = 0

    def update(self, preds: np.ndarray, target: np.ndarray):
        assert preds.shape == target.shape  # (batch_size, num_classes)
        return self.update_argsort(_sorted_top_k_idx(preds, self.top_k), target)

    def update_argsort(self, argsort_preds: np.ndarray, target: np.ndarray):
        dcgs = _batchDCG_argsort(argsort_preds, target, self.top_k)
        idcgs = _batchIDCG(target, self.top_k)
        ndcg_score = dcgs / idcgs
        self.score += np.nan_to_num(ndcg_score, nan=0.0).sum(axis=0)
        self.num_sample += argsort_preds.shape[0]

    def compute(self) -> dict[str, float]:
        return {f"NDCG@{k}": v for k, v in zip(range(1, self.top_k + 1), self.score / self.num_sample)}

    def reset(self):
        self.score = 0
        self.num_sample = 0


class RPrecision:
    def __init__(self, top_k: int):
        """Compute the R-Precision@K.

        Args:
            top_k: Consider only the top k elements for each query.
        """
        _check_top_k(top_k)

        self.top_k = top_k
        self.score = 0
        self.num_sample = 0

    def update(self, preds: np.ndarray, target: np.ndarray):
        assert preds.shape == target.shape  # (batch_size, num_classes)
        return self.update_argsort(np.argpartition(preds, -self.top_k), target)

    def update_argsort(self, argsort_preds: np.ndarray, target: np.ndarray):
        top_k_idx = argsort_preds[:, -self.top_k :]
        num_relevant = np.take_along_axis(target, top_k_idx, axis=-1).sum(axis=-1)  # (batch_size, )
        self.score += np.nan_to_num(num_relevant / np.minimum(self.top_k, target.sum(axis=-1)), nan=0.0).sum()
        self.num_sample += argsort_preds.shape[0]

    def compute(self) -> float:
        return self.score / self.num_sample

    def reset(self):
        self.score = 0
        self.num_sample = 0


class Precision:
    def __init__(self, num_classes: int, average: str, top_k: int):
        """Compute the Precision@K.

        Args:
            num_classes: The number of classes.
            average: Define the reduction that is applied over labels. Currently only "samples" is supported.
            top_k: Consider only the top k elements for each query.
        """
        if average != "samples":
            raise ValueError("unsupported average")

        _check_top_k(top_k)

        self.top_k = top_k
        self.score = 0
        self.num_sample = 0

    def update(self, preds: np.ndarray, target: np.ndarray):
        assert preds.shape == target.shape  # (batch_size, num_classes)
        return self.update_argsort(np.argpartition(preds, -self.top_k), target)

    def update_argsort(self, argsort_preds: np.ndarray, target: np.ndarray):
        top_k_idx = argsort_preds[:, -self.top_k :]
        num_relevant = np.take_along_axis(target, top_k_idx, -1).sum()
        self.score += num_relevant / self.top_k
        self.num_sample += argsort_preds.shape[0]

    def compute(self) -> float:
        return self.score / self.num_sample

    def reset(self):
        self.score = 0
        self.num_sample = 0


class BatchPrecision:
    def __init__(self, num_classes: int, average: str, top_k: int):
        """Compute Precision@k for K = 1 to top_k.

        Args:
            num_classes: The number of classes.
            average: Define the reduction that is applied over labels. Currently only "samples" is supported.
            top_k: Consider up to the top k elements for each query.
        """
        if average != "samples":
            raise ValueError("unsupported average")

        _check_top_k(top_k)

        self.top_k = top_k
        self.score = np.zeros(top_k)
        self.num_sample = 0

    def update(self, preds: np.ndarray, target: np.ndarray):
        assert preds.shape == target.shape  # (batch_size, num_classes)
        return self.update_argsort(_sorted_top_k_idx(preds, self.top_k), target)

    def update_argsort(self, argsort_preds: np.ndarray, target: np.ndarray):
        top_k_idx = argsort_preds[:, -self.top_k :][:, ::-1]
        num_relevant = np.take_along_axis(target, top_k_idx, -1).cumsum(axis=1)
        scores = num_relevant / np.arange(1, self.top_k + 1).reshape(1, -1)
        self.score += scores.sum(axis=0)
        self.num_sample += argsort_preds.shape[0]

    def compute(self) -> dict[str, float]:
        return {f"P@{k}": v for k, v in zip(range(1, self.top_k), self.score / self.num_sample)}

    def reset(self):
        self.score = 0
        self.num_sample = 0


class Recall:
    def __init__(self, num_classes: int, average: str, top_k: int):
        """Compute the Recall@K.

        Args:
            num_classes: The number of classes.
            average: Define the reduction that is applied over labels. Currently only "samples" is supported.
            top_k: Consider only the top k elements for each query.
        """
        if average != "samples":
            raise ValueError("unsupported average")

        _check_top_k(top_k)

        self.top_k = top_k
        self.score = 0
        self.num_sample = 0

    def update(self, preds: np.ndarray, target: np.ndarray):
        assert preds.shape == target.shape  # (batch_size, num_classes)
        return self.update_argsort(np.argpartition(preds, -self.top_k), target)

    def update_argsort(self, argsort_preds: np.ndarray, target: np.ndarray):
        top_k_idx = argsort_preds[:, -self.top_k :]
        num_relevant = np.take_along_axis(target, top_k_idx, -1).sum(axis=-1)
        self.score += np.sum(np.nan_to_num(num_relevant / target.sum(axis=-1), nan=1.0))
        self.num_sample += argsort_preds.shape[0]

    def compute(self) -> float:
        return self.score / self.num_sample

    def reset(self):
        self.score = 0
        self.num_sample = 0


class F1:
    def __init__(self, num_classes: int, average: str, multiclass=False):
        """Compute the F1 score.

        Args:
            num_classes: The number of labels.
            average: Define the reduction that is applied over labels. Should be one of "macro", "micro",
            "another-macro".
            multiclass: Whether the task is a multiclass task.
        """
        self.num_classes = num_classes
        if average not in {"macro", "micro", "another-macro"}:
            raise ValueError("unsupported average")
        self.average = average
        self.multiclass = multiclass
        self.tp = self.fp = self.fn = 0

    def update(self, preds: np.ndarray, target: np.ndarray):
        assert preds.shape == target.shape  # (batch_size, num_classes)
        if self.multiclass:
            max_idx = np.argmax(preds, axis=1).reshape(-1, 1)
            preds = np.zeros(preds.shape)
            np.put_along_axis(preds, max_idx, 1, axis=1)
        else:
            preds = preds > 0
        self.tp += np.logical_and(target == 1, preds == 1).sum(axis=0)
        self.fn += np.logical_and(target == 1, preds == 0).sum(axis=0)
        self.fp += np.logical_and(target == 0, preds == 1).sum(axis=0)

    def compute(self) -> float:
        prev_settings = np.seterr("ignore")

        if self.average == "macro":
            score = np.nansum(2 * self.tp / (2 * self.tp + self.fp + self.fn)) / self.num_classes
        elif self.average == "micro":
            score = np.nan_to_num(2 * np.sum(self.tp) / np.sum(2 * self.tp + self.fp + self.fn))
        elif self.average == "another-macro":
            macro_prec = np.nansum(self.tp / (self.tp + self.fp)) / self.num_classes
            macro_recall = np.nansum(self.tp / (self.tp + self.fn)) / self.num_classes
            score = np.nan_to_num(2 * macro_prec * macro_recall / (macro_prec + macro_recall))

        np.seterr(**prev_settings)
        return score

    def reset(self):
        self.tp = self.fp = self.fn = 0


class MetricCollection(dict):
    """A collection of metrics created by get_metrics.
    MetricCollection computes metric values in two steps. First, batches of
    decision values and labels are added with update(). After all instances have been
    added, compute() computes the metric values from the accumulated batches.
    """

    def __init__(self, metrics):
        self.metrics = metrics
        self.max_k = max(getattr(metric, "top_k", 0) for metric in self.metrics.values())

    def update(self, preds: np.ndarray, target: np.ndarray):
        """Adds a batch of decision values and labels.

        Args:
            preds (np.ndarray): A matrix of decision values with dimensions number of instances * number of classes.
            target (np.ndarray): A 0/1 matrix of labels with dimensions number of instances * number of classes.
        """
        assert preds.shape == target.shape  # (batch_size, num_classes)

        # update_argsort only requires there to be max_k indices
        if self.max_k > 0:
            argsort_preds = _sorted_top_k_idx(preds, self.max_k)

        for metric in self.metrics.values():
            if hasattr(metric, "update_argsort"):
                metric.update_argsort(argsort_preds, target)
            else:
                metric.update(preds, target)

    def compute(self) -> dict[str, float]:
        """Computes the metrics from the accumulated batches of decision values and labels.

        Returns:
            dict[str, float]: A dictionary of metric values.
        """
        ret = {}
        for name, metric in self.metrics.items():
            value = metric.compute()
            if isinstance(value, dict):
                ret.update(value)
            else:
                ret[name] = value
        return ret

    def reset(self):
        """Clears the accumulated batches of decision values and labels."""
        for metric in self.metrics.values():
            metric.reset()


def get_metrics(monitor_metrics: list[str], num_classes: int, multiclass: bool = False) -> MetricCollection:
    """Get a collection of metrics by their names.
    See MetricCollection for more details.

    Args:
        monitor_metrics (list[str]): A list of metric names.
        num_classes (int): The number of classes.
        multiclass (bool, optional): Enable multiclass mode. Defaults to False.

    Returns:
        MetricCollection: A metric collection of the list of metrics.
    """
    if monitor_metrics is None:
        monitor_metrics = []
    metrics = {}
    for metric in monitor_metrics:
        if re.match("P@\d+", metric):
            metrics[metric] = Precision(num_classes, average="samples", top_k=int(metric[2:]))
        elif re.match("R@\d+", metric):
            metrics[metric] = Recall(num_classes, average="samples", top_k=int(metric[2:]))
        elif re.match("RP@\d+", metric):
            metrics[metric] = RPrecision(top_k=int(metric[3:]))
        elif re.match("NDCG@\d+", metric):
            metrics[metric] = NDCG(top_k=int(metric[5:]))
        elif metric in {"Another-Macro-F1", "Macro-F1", "Micro-F1"}:
            metrics[metric] = F1(num_classes, average=metric[:-3].lower(), multiclass=multiclass)
        else:
            raise ValueError(f"invalid metric: {metric}")

    return MetricCollection(metrics)


def compute_metrics(
    preds: np.ndarray, target: np.ndarray, monitor_metrics: list[str], multiclass: bool = False
) -> dict[str, float]:
    """Compute metrics with decision values and labels.
    See get_metrics and MetricCollection if decision values and labels are too
    large to hold in memory.


    Args:
        preds (np.ndarray): A matrix of decision values with dimensions number of instances * number of classes.
        target (np.ndarray): A 0/1 matrix of labels with dimensions number of instances * number of classes.
        monitor_metrics (list[str]): A list of metric names.
        multiclass (bool, optional): Enable multiclass mode. Defaults to False.

    Returns:
        dict[str, float]: A dictionary of metric values.
    """
    assert preds.shape == target.shape

    metric = get_metrics(monitor_metrics, preds.shape[1], multiclass)
    metric.update(preds, target)
    return metric.compute()


def tabulate_metrics(metric_dict: dict[str, float], split: str) -> str:
    """Convert a dictionary of metric values into a pretty formatted string for printing.

    Args:
        metric_dict (dict[str, float]): A dictionary of metric values.
        split (str): Name of the data split.

    Returns:
        str: Pretty formatted string.
    """
    msg = f"====== {split} dataset evaluation result =======\n"
    header = "|".join([f"{k:^18}" for k in metric_dict.keys()])
    values = "|".join(
        [f"{x:^18.4f}" if isinstance(x, (np.floating, float)) else f"{x:^18}" for x in metric_dict.values()]
    )
    msg += f"|{header}|\n|{'-----------------:|' * len(metric_dict)}\n|{values}|\n"
    return msg


def _check_top_k(k):
    if not (isinstance(k, int) and k > 0):
        raise ValueError('"k" has to be a positive integer')
