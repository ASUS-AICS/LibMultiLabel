import re

import torch
import numpy as np
import torchmetrics.classification
from torchmetrics import Metric, MetricCollection, Precision, Recall, RetrievalNormalizedDCG
from torchmetrics.utilities.data import select_topk


class RPrecision(Metric):
    """R-precision calculates precision at k by adjusting k to the minimum value of the number of
    relevant labels and k. Please find the definition here:
    https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html

    Args:
        top_k (int): the top k relevant labels to evaluate.
    """
    def __init__(
        self,
        top_k
    ):
        super().__init__()
        self.top_k = top_k
        self.add_state("score", default=torch.tensor(0., dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("num_sample", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        assert preds.shape == target.shape
        binary_topk_preds = select_topk(preds, self.top_k)
        target = target.to(dtype=torch.int)
        num_relevant = torch.sum(binary_topk_preds & target, dim=-1)
        top_ks = torch.tensor([self.top_k]*preds.shape[0]).to(preds.device)
        self.score += torch.nan_to_num(
            num_relevant / torch.min(top_ks, target.sum(dim=-1)),
            posinf=0.
        ).sum()
        self.num_sample += len(preds)

    def compute(self):
        return self.score / self.num_sample


def get_metrics(metric_threshold, monitor_metrics, num_classes):
    """Map monitor metrics to the corresponding classes defined in `torchmetrics.Metric`
    (https://torchmetrics.readthedocs.io/en/latest/references/modules.html).

    Args:
        metric_threshold (float): Thresholds to monitor for metrics.
        monitor_metrics (list): Metrics to monitor while validating.
        num_classes (int): Total number of classes.

    Raises:
        ValueError: The metric is invalid if:
            (1) It is not one of 'P@k', 'R@k', 'RP@k', 'nDCG@k', 'Micro-Precision',
                'Micro-Recall', 'Micro-F1', 'Macro-F1', 'Another-Macro-F1', or a
                `torchmetrics.Metric`.
            (2) Metric@k: k is greater than `num_classes`.

    Returns:
        torchmetrics.MetricCollection: A collections of `torchmetrics.Metric` for evaluation.
    """
    if monitor_metrics is None:
        monitor_metrics = []

    metrics = dict()
    for metric in monitor_metrics:
        if isinstance(metric, Metric):  # customized metric
            metrics[type(metric).__name__] = metric
            continue

        match_top_k = re.match(r'\b(P|R|RP|nDCG)\b@(\d+)', metric)
        match_metric = re.match(r'\b(Micro|Macro)\b-\b(Precision|Recall|F1)\b', metric)

        if match_top_k:
            metric_abbr = match_top_k.group(1) # P, R, PR, or nDCG
            top_k = int(match_top_k.group(2))
            if top_k >= num_classes:
                raise ValueError(
                    f'Invalid metric: {metric}. {top_k} is greater than {num_classes}.')
            if metric_abbr == 'P':
                metrics[metric] = Precision(num_classes, average='samples', top_k=top_k)
            elif metric_abbr == 'R':
                metrics[metric] = Recall(num_classes, average='samples', top_k=top_k)
            elif metric_abbr == 'RP':
                metrics[metric] = RPrecision(top_k=top_k)
            elif metric_abbr == 'nDCG':
                metrics[metric] = RetrievalNormalizedDCG(k=top_k)
        elif metric == 'Another-Macro-F1':
            # The f1 value of macro_precision and macro_recall. This variant of
            # macro_f1 is less preferred but is used in some works. Please
            # refer to Opitz et al. 2019 [https://arxiv.org/pdf/1911.03347.pdf]
            macro_prec = Precision(num_classes, metric_threshold, average='macro')
            macro_recall = Recall(num_classes, metric_threshold, average='macro')
            metrics[metric] = 2 * (macro_prec * macro_recall) / \
                (macro_prec + macro_recall + 1e-10)
        elif match_metric:
            average_type = match_metric.group(1).lower() # Micro or Macro
            metric_type = match_metric.group(2) # Precision, Recall, or F1
            metrics[metric] = getattr(torchmetrics.classification, metric_type)(
                num_classes, metric_threshold, average=average_type)
        else:
            raise ValueError(
                f'Invalid metric: {metric}. Make sure the metric is in the right format: Macro/Micro-Precision/Recall/F1 (ex. Micro-F1)')

    return MetricCollection(metrics)


def tabulate_metrics(metric_dict, split):
    msg = f'====== {split} dataset evaluation result =======\n'
    header = '|'.join([f'{k:^18}' for k in metric_dict.keys()])
    values = '|'.join([f'{x * 100:^18.4f}' if isinstance(x, (np.floating, float)) else f'{x:^18}' for x in metric_dict.values()])
    msg += f"|{header}|\n|{'-----------------:|' * len(metric_dict)}\n|{values}|\n"
    return msg
