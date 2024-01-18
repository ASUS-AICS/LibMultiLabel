from __future__ import annotations

import re

import numpy as np

__all__ = ["get_metrics", "compute_metrics", "tabulate_metrics", "MetricCollection"]


def _argsort_top_k(preds: np.ndarray, top_k: int) -> np.ndarray:
    """Sorts the top k indices in O(n + k log k) time.
    The sorting order is ascending to be consistent with np.sort.
    This means the last element is the largest, the first element is the kth largest.
    """
    top_k_idx = np.argpartition(preds, -top_k)[:, -top_k:]
    argsort_top_k = np.argsort(np.take_along_axis(preds, top_k_idx, axis=-1))
    return np.take_along_axis(top_k_idx, argsort_top_k, axis=-1)


def _DCG_argsort(argsort_preds: np.ndarray, target: np.ndarray, top_k: int) -> np.ndarray:
    """Computes DCG@k with a sorted preds array and a target array."""
    top_k_idx = argsort_preds[:, -top_k:][:, ::-1]
    gains = np.take_along_axis(target, top_k_idx, axis=-1)
    discount = 1 / (np.log2(np.arange(top_k) + 2))
    # get the sum product over the last axis of gains and discount
    dcgs = gains.dot(discount)
    return dcgs


def _IDCG(target: np.ndarray, top_k: int) -> np.ndarray:
    """Computes IDCG@k for a 0/1 target array. A 0/1 target is a special case that
    doesn't require sorting. If IDCG is computed with DCG,
    then target will need to be sorted, which incurs a large overhead.
    """
    num_relevant_labels = target.sum(axis=1, dtype="i")
    discount = 1 / (np.log2(np.arange(top_k) + 2))
    cum_discount = discount.cumsum()
    # NDCG for an instance with no relevant labels is defined to be 0.
    # If an instance has no relevant labels, its DCG will be 0,
    # thus we can return any non-zero IDCG for that instance.
    # Here we return the first element of cum_discount, which is 1.
    idx = np.clip(num_relevant_labels - 1, 0, top_k - 1)
    return cum_discount[idx]


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
        return self.update_argsort(_argsort_top_k(preds, self.top_k), target)

    def update_argsort(self, argsort_preds: np.ndarray, target: np.ndarray):
        dcg = _DCG_argsort(argsort_preds, target, self.top_k)
        idcg = _IDCG(target, self.top_k)
        ndcg_score = dcg / idcg
        # by convention, ndcg is 0 for zero label instances
        self.score += np.nan_to_num(ndcg_score, nan=0.0).sum()
        self.num_sample += argsort_preds.shape[0]

    def compute(self) -> float:
        return self.score / self.num_sample

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
        # by convention, rprecision is 0 for zero label instances
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
        # by convention, recall is 0 for zero label instances
        self.score += np.nan_to_num(num_relevant / target.sum(axis=-1), nan=0.0).sum()
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

        # F1 is 0 for the cases where there are no positive instances
        if self.average == "macro":
            score = np.nansum(2 * self.tp / (2 * self.tp + self.fp + self.fn)) / self.num_classes
        elif self.average == "micro":
            score = np.nan_to_num(2 * np.sum(self.tp) / np.sum(2 * self.tp + self.fp + self.fn), nan=0.0)
        elif self.average == "another-macro":
            macro_prec = np.nansum(self.tp / (self.tp + self.fp)) / self.num_classes
            macro_recall = np.nansum(self.tp / (self.tp + self.fn)) / self.num_classes
            score = np.nan_to_num(2 * macro_prec * macro_recall / (macro_prec + macro_recall), nan=0.0)

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

        # The main bottleneck when computing metrics is sorting the top k indices.
        # As an optimization, we sort only once and pass the sorted predictions to metrics that needs them.
        # Top k ranking metrics only requires the sorted top k predictions, so we don't need to fully sort the predictions.
        if self.max_k > 0:
            argsort_preds = _argsort_top_k(preds, self.max_k)

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
            ret[name] = metric.compute()
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
