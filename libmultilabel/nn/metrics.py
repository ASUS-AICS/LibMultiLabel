import re

import numpy as np
import torch
import torchmetrics.classification
from torchmetrics import Metric, MetricCollection, Precision, Recall
from torchmetrics.functional.retrieval.ndcg import retrieval_normalized_dcg
from torchmetrics.utilities.data import select_topk


class Loss(Metric):
    """Loss records the batch-wise losses
    and then obtains a mean loss from the recorded losses.
    """
    # If the metric state of one batch is independent of the state of other batches,
    # full_state_update can be set to False,
    # which leads to more efficient computation with calling update() only once.
    # Please find the detailed explanation here:
    # https://torchmetrics.readthedocs.io/en/stable/pages/implement.html
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("loss", default=torch.tensor(
            0., dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("num_sample", default=torch.tensor(0),
                       dist_reduce_fx="sum")

    def update(self, preds, target, loss):
        assert preds.shape == target.shape
        self.loss += loss * len(preds)
        self.num_sample += len(preds)

    def compute(self):
        return self.loss / self.num_sample


class NDCG(Metric):
    """NDCG (Normalized Discounted Cumulative Gain) sums the true scores
    ranked in the order induced by the predicted scores after applying a logarithmic discount,
    and then divides by the best possible score (Ideal DCG, obtained for a perfect ranking)
    to obtain a score between 0 and 1.
    The definition is quoted from:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html
    Please find the formal definition here:
    https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html

    Args:
        top_k (int): the top k relevant labels to evaluate.
    """
    # If the metric state of one batch is independent of the state of other batches,
    # full_state_update can be set to False,
    # which leads to more efficient computation with calling update() only once.
    # Please find the detailed explanation here:
    # https://torchmetrics.readthedocs.io/en/stable/pages/implement.html
    full_state_update = False

    def __init__(
        self,
        top_k
    ):
        super().__init__()
        self.top_k = top_k
        self.add_state("ndcg", default=[], dist_reduce_fx="cat")

    def update(self, preds, target):
        assert preds.shape == target.shape
        # implement batch-wise calculations instead of storing results of all batches
        self.ndcg += [self._metric(p, t) for p, t in zip(preds, target)]

    def compute(self):
        return torch.stack(self.ndcg).mean()

    def _metric(self, preds, target):
        return retrieval_normalized_dcg(preds, target, k=self.top_k)


class RPrecision(Metric):
    """R-precision calculates precision at k by adjusting k to the minimum value of the number of
    relevant labels and k. The definition is given at Appendix C equation (3) of
    https://aclanthology.org/P19-1636.pdf

    Args:
        top_k (int): the top k relevant labels to evaluate.
    """
    # If the metric state of one batch is independent of the state of other batches,
    # full_state_update can be set to False,
    # which leads to more efficient computation with calling update() only once.
    # Please find the detailed explanation here:
    # https://torchmetrics.readthedocs.io/en/stable/pages/implement.html
    full_state_update = False

    def __init__(
        self,
        top_k
    ):
        super().__init__()
        self.top_k = top_k
        self.add_state("score", default=torch.tensor(
            0., dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("num_sample", default=torch.tensor(0),
                       dist_reduce_fx="sum")

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


class MacroF1(Metric):
    """The macro-f1 score computes the average f1 scores of all labels in the dataset.

    Args:
        num_classes (int): Total number of classes.
        metric_threshold (float): Threshold to monitor for metrics.
        another_macro_f1 (bool, optional): Whether to compute the 'Another-Macro-F1' score.
            The 'Another-Macro-F1' is the f1 value of macro-precision and macro-recall.
            This variant of macro-f1 is less preferred but is used in some works.
            Please refer to Opitz et al. 2019 [https://arxiv.org/pdf/1911.03347.pdf].
            Defaults to False.
    """
    # If the metric state of one batch is independent of the state of other batches,
    # full_state_update can be set to False,
    # which leads to more efficient computation with calling update() only once.
    # Please find the detailed explanation here:
    # https://torchmetrics.readthedocs.io/en/stable/pages/implement.html
    full_state_update = False

    def __init__(
        self,
        num_classes,
        metric_threshold,
        another_macro_f1=False,
        top_k=None
    ):
        super().__init__()
        self.metric_threshold = metric_threshold
        self.another_macro_f1 = another_macro_f1
        self.top_k = top_k
        self.add_state("preds_sum", default=torch.zeros(
            num_classes, dtype=torch.double))
        self.add_state("target_sum", default=torch.zeros(
            num_classes, dtype=torch.double))
        self.add_state("tp_sum", default=torch.zeros(
            num_classes, dtype=torch.double))

    def update(self, preds, target):
        assert preds.shape == target.shape
        if self.top_k:
            preds = select_topk(preds, self.top_k)
        else:
            preds = torch.where(preds > self.metric_threshold, 1, 0)

        self.preds_sum = torch.add(self.preds_sum, preds.sum(dim=0))
        self.target_sum = torch.add(self.target_sum, target.sum(dim=0))
        self.tp_sum = torch.add(self.tp_sum, (preds & target).sum(dim=0))

    def compute(self):
        if self.another_macro_f1:
            macro_prec = torch.mean(torch.nan_to_num(
                self.tp_sum / self.preds_sum, posinf=0.))
            macro_recall = torch.mean(torch.nan_to_num(
                self.tp_sum / self.target_sum, posinf=0.))
            return 2 * (macro_prec * macro_recall) / (macro_prec + macro_recall + 1e-10)
        else:
            label_f1 = 2 * self.tp_sum / \
                (self.preds_sum + self.target_sum + 1e-10)
            return torch.mean(label_f1)


def get_metrics(metric_threshold, monitor_metrics, num_classes, top_k=None):
    """Map monitor metrics to the corresponding classes defined in `torchmetrics.Metric`
    (https://torchmetrics.readthedocs.io/en/latest/references/modules.html).

    Args:
        metric_threshold (float): Threshold to monitor for metrics.
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
        match_metric = re.match(
            r'\b(Micro|Macro)\b-\b(Precision|Recall|F1)\b', metric)

        if match_top_k:
            metric_abbr = match_top_k.group(1)  # P, R, PR, or nDCG
            top_k = int(match_top_k.group(2))
            if top_k >= num_classes:
                raise ValueError(
                    f'Invalid metric: {metric}. top_k ({top_k}) is greater than num_classes({num_classes}).')
            if metric_abbr == 'P':
                metrics[metric] = Precision(
                    num_classes, average='samples', top_k=top_k)
            elif metric_abbr == 'R':
                metrics[metric] = Recall(
                    num_classes, average='samples', top_k=top_k)
            elif metric_abbr == 'RP':
                metrics[metric] = RPrecision(top_k=top_k)
            elif metric_abbr == 'nDCG':
                metrics[metric] = NDCG(top_k=top_k)
                # The implementation in torchmetrics stores the prediction/target of all batches,
                # which can lead to CUDA out of memory.
                # metrics[metric] = RetrievalNormalizedDCG(k=top_k)
        elif metric == 'Another-Macro-F1':
            metrics[metric] = MacroF1(
                num_classes, metric_threshold, another_macro_f1=True, top_k=top_k)
        elif metric == 'Macro-F1':
            metrics[metric] = MacroF1(
                num_classes, metric_threshold, top_k=top_k)
        elif metric == 'Loss':
            metrics[metric] = Loss()
        elif match_metric:
            average_type = match_metric.group(1).lower()  # Micro
            metric_type = match_metric.group(2)  # Precision, Recall, or F1
            metric_type = metric_type.replace(
                'F1', 'F1Score')  # to be determined
            metrics[metric] = getattr(torchmetrics.classification, metric_type)(
                num_classes, metric_threshold, average=average_type, top_k=top_k)
        else:
            raise ValueError(
                f'Invalid metric: {metric}. Make sure the metric is in the right format: Macro/Micro-Precision/Recall/F1 (ex. Micro-F1)')

    # If compute_groups is set to True (default), incorrect results may be calculated.
    # Please refer to https://github.com/Lightning-AI/metrics/issues/746 for more details.
    return MetricCollection(metrics, compute_groups=False)


def tabulate_metrics(metric_dict, split):
    msg = f'====== {split} dataset evaluation result =======\n'
    header = '|'.join([f'{k:^18}' for k in metric_dict.keys()])
    values = '|'.join([f'{x * 100:^18.4f}' if isinstance(x, (np.floating,
                      float)) else f'{x:^18}' for x in metric_dict.values()])
    msg += f"|{header}|\n|{'-----------------:|' * len(metric_dict)}\n|{values}|\n"
    return msg
