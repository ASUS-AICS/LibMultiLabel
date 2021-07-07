"""Metrics modified from CAML-MIMIC are tentatively placed here.
(https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py)
They are used for the internal need to compare with CAML."""


import re

import numpy as np
from sklearn.metrics import classification_report, multilabel_confusion_matrix


def f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-10)


def precision_recall_at_ks(y_true, y_pred_vals, top_ks):
    y_pred_ranked_idx = np.argsort(-y_pred_vals)
    n_pos = y_true.sum(axis=1)
    scores = {}
    for k in top_ks:
        n_pos_in_top_k = np.take_along_axis(y_true, y_pred_ranked_idx[:,:k], axis=1).sum(axis=1)
        scores[f'P@{k}'] = np.mean(n_pos_in_top_k / k).item()  # precision at k
        scores[f'R@{k}'] = np.mean(n_pos_in_top_k / (n_pos + 1e-10)).item()  # recall at k
    return scores


class MultiLabelMetrics():
    def __init__(self, config):
        self.monitor_metrics = config.get('monitor_metrics', [])
        self.metric_threshold = config.get('metric_threshold', 0.5)

        self.n_eval = 0
        self.metric_stats = {
            'Micro-Precision': 0.,
            'Micro-Recall': 0.,
            'Micro-F1': 0.,
            'confusion_matrix': 0.,
        }

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

    def add_values(self, y_true, y_pred):
        """Add batch of y_true and y_pred.

        Args:
            y_true (ndarray): an array with ground truth labels (shape: batch_size * number of classes)
            y_pred (ndarray): an array with predicted label values (shape: batch_size * number of classes)
        """
        y_pred_pos = y_pred > self.metric_threshold
        report_dict = classification_report(y_true, y_pred_pos, output_dict=True, zero_division=0)

        n_eval = len(y_true)
        self.n_eval += n_eval
        self.metric_stats['Micro-Precision'] += (report_dict['micro avg']['precision'] * n_eval)
        self.metric_stats['Micro-Recall'] += (report_dict['micro avg']['recall'] * n_eval)
        self.metric_stats['Micro-F1'] += (report_dict['micro avg']['f1-score'] * n_eval)
        self.metric_stats['confusion_matrix'] += multilabel_confusion_matrix(y_true, y_pred_pos)

        scores = precision_recall_at_ks(y_true, y_pred, top_ks=self.top_ks)
        for metric in self.prec_recall_metrics:
            self.metric_stats[metric] += (scores[metric] * n_eval)

    def get_metric_dict(self):
        """Get evaluation results."""

        result = {}
        for metric, val in self.metric_stats.items():
            if metric == 'confusion_matrix':
                labelwise_precision = val[:,1,1] / (val[:,1,1] + val[:,0,1] + 1e-10)
                labelwise_recall = val[:,1,1] / (val[:,1,1] + val[:,1,0] + 1e-10)
                labelwise_f1 = f1(labelwise_precision, labelwise_recall)
                result['Macro-F1'] = labelwise_f1.mean()

                # The f1 value of macro_precision and macro_recall. This
                # variant of # macro_f1 is less preferred but is used in
                # some works.
                macro_precision = labelwise_precision.mean()
                macro_recall = labelwise_recall.mean()
                result['Another-Macro-F1'] = f1(macro_precision, macro_recall)
            else:
                result[metric] = val / self.n_eval
        return result

    def __repr__(self):
        """Return evaluation results in markdown."""
        result = self.get_metric_dict()
        header = '|'.join([f'{k:^18}' for k in result.keys()])
        values = '|'.join([f'{x * 100:^18.4f}' if isinstance(x, (np.floating, float)) else f'{x:^18}' for x in result.values()])
        return f"|{header}|\n|{'-----------------:|' * len(result)}\n|{values}|"
