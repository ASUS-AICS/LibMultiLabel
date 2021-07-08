import re

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

from .utils import argsort_top_k


def f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-10)


def precision_recall_at_ks(y_true, y_pred_vals, top_ks):
    max_k = max(top_ks)
    top_idx = argsort_top_k(y_pred_vals, max_k, axis=1)
    n_pos = y_true.sum(axis=1)
    scores = {}
    for k in top_ks:
        n_pos_in_top_k = np.take_along_axis(y_true, top_idx[:,:k], axis=1).sum(axis=1)
        scores[f'P@{k}'] = np.mean(n_pos_in_top_k / k).item()  # precision at k
        scores[f'R@{k}'] = np.mean(n_pos_in_top_k / (n_pos + 1e-10)).item()  # recall at k
    return scores


class MultiLabelMetrics():
    def __init__(self, config):
        self.monitor_metrics = config.get('monitor_metrics', [])
        self.metric_threshold = config.get('metric_threshold', 0.5)

        self.n_eval = 0
        self.multilabel_confusion_matrix = 0.
        self.metric_stats = {}

        self.top_ks = set()
        self.prec_recall_metrics = []
        for metric in self.monitor_metrics:
            if re.match('[P|R]@\d+', metric):
                top_k = int(metric[2:])
                self.top_ks.add(top_k)
                self.metric_stats[metric] = 0.
                self.prec_recall_metrics.append(metric)
            elif metric not in ['Micro-Precision', 'Micro-Recall', 'Micro-F1', 'Macro-F1', 'Another-Macro-F1']:
                raise ValueError(f'Invalid metric: {metric}')

    def reset(self):
        self.n_eval = 0
        self.multilabel_confusion_matrix = 0.
        for metric in self.metric_stats:
            self.metric_stats[metric] = 0.

    def add_values(self, y_true, y_pred):
        """Add batch of y_true and y_pred.

        Args:
            y_true (ndarray): an array with ground truth labels (shape: batch_size * number of classes)
            y_pred (ndarray): an array with predicted label values (shape: batch_size * number of classes)
        """
        y_pred_pos = y_pred > self.metric_threshold

        n_eval = len(y_true)
        self.n_eval += n_eval
        self.multilabel_confusion_matrix += multilabel_confusion_matrix(y_true, y_pred_pos)

        scores = precision_recall_at_ks(y_true, y_pred, top_ks=self.top_ks)
        for metric in self.prec_recall_metrics:
            self.metric_stats[metric] += (scores[metric] * n_eval)

    def get_metric_dict(self):
        """Get evaluation results."""

        result = {}
        cm = self.multilabel_confusion_matrix
        cum_cm = cm.sum(axis=0)
        cum_tp, cum_fp, cum_fn = cum_cm[1,1], cum_cm[0,1], cum_cm[1,0]
        micro_precision = cum_tp / (cum_tp + cum_fp + 1e-10)
        micro_recall = cum_tp / (cum_tp + cum_fn + 1e-10)

        ml_tp, ml_fp, ml_fn = cm[:,1,1], cm[:,0,1], cm[:,1,0]
        labelwise_precision = ml_tp / (ml_tp + ml_fp + 1e-10)
        labelwise_recall = ml_tp / (ml_tp + ml_fn + 1e-10)
        macro_precision = labelwise_precision.mean()
        macro_recall = labelwise_recall.mean()

        result = {
            'Micro-Precision': micro_precision,
            'Micro-Recall': micro_recall,
            'Micro-F1': f1(micro_precision, micro_recall),
            'Macro-F1': f1(labelwise_precision, labelwise_recall).mean(),
            'Another-Macro-F1': f1(macro_precision, macro_recall),
        }
        for metric, val in self.metric_stats.items():
            result[metric] = val / self.n_eval
        return result

    def __repr__(self):
        """Return evaluation results in markdown."""
        result = self.get_metric_dict()
        header = '|'.join([f'{k:^18}' for k in result.keys()])
        values = '|'.join([f'{x * 100:^18.4f}' if isinstance(x, (np.floating, float)) else f'{x:^18}' for x in result.values()])
        return f"|{header}|\n|{'-----------------:|' * len(result)}\n|{values}|"
