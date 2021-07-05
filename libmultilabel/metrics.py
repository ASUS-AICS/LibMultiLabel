"""Metrics modified from CAML-MIMIC are tentatively placed here.
(https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py)
They are used for the internal need to compare with CAML."""


import re

import numpy as np
from sklearn.metrics import classification_report


def another_macro_f1(macro_prec, macro_rec):
    # The f1 value of macro_precision and macro_recall. This variant of
    # macro_f1 is less preferred but is used in some works
    f1 = 2 * (macro_prec * macro_rec) / (macro_prec + macro_rec + 1e-10)
    return f1


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
    def __init__(
        self,
        metric_threshold=0.5,
        monitor_metrics=None
    ):
        self.monitor_metrics = monitor_metrics if monitor_metrics else []
        self.metric_threshold = metric_threshold
        self.y_true = []
        self.y_pred = []
        self.cached_results = {}

        self.top_ks = set()
        for metric in self.monitor_metrics:
            if re.match('[P|R]@\d+', metric):
                top_k = int(metric[2:])
                self.top_ks.add(top_k)
            elif metric not in ['Micro-Precision', 'Micro-Recall', 'Micro-F1', 'Macro-F1', 'Another-Macro-F1']:
                raise ValueError(f'Invalid metric: {metric}')

    def add_values(self, y_true, y_pred):
        """Add batch of y_true and y_pred.

        Args:
            y_true (ndarray): an array with ground truth labels (shape: batch_size * number of classes)
            y_pred (ndarray): an array with predicted labels (shape: batch_size * number of classes)
        """
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def eval(self):
        """Evaluate precision, recall, micro-f1, macro-f1, and P@k/R@k listed in the monitor_metrics."""
        y_true = np.vstack(self.y_true)
        y_pred = np.vstack(self.y_pred)
        report_dict = classification_report(y_true, y_pred > self.metric_threshold, output_dict=True, zero_division=0)
        result = {
            'Micro-Precision': report_dict['micro avg']['precision'],
            'Micro-Recall': report_dict['micro avg']['recall'],
            'Micro-F1': report_dict['micro avg']['f1-score'],
            'Macro-F1': report_dict['macro avg']['f1-score'],
            'Another-Macro-F1': another_macro_f1(
                macro_prec=report_dict['macro avg']['precision'],
                macro_rec=report_dict['macro avg']['recall'])  # caml's macro-f1
        }
        # add metrics like P@k, R@k to the result dict
        scores = precision_recall_at_ks(y_true, y_pred, top_ks=self.top_ks)
        precision_recall_monitored = sorted(set(self.monitor_metrics) & scores.keys())
        result.update({metric: scores[metric] for metric in precision_recall_monitored})
        self.cached_results = result

    def get_y_pred(self):
        """Convert 3D array (shape: number of batches * batch_size * number of classes
        to 2D array (shape: number of samples * number of classes).
        """
        return np.vstack(self.y_pred)

    def get_metric_dict(self, use_cache=False):
        """Evaluate or get score dictionary from cache.

        Args:
            threshold (float, optional): threshold to evaluate precision, recall, and f1 score. Defaults to 0.5.
            use_cache (bool, optional): return cached results or not. Defaults to False.
        """
        if not use_cache:
            self.eval()
        return self.cached_results

    def __repr__(self):
        """Return cache results in markdown."""
        header = '|'.join([f'{k:^18}' for k in self.cached_results.keys()])
        values = '|'.join([f'{x * 100:^18.4f}' if isinstance(x, (np.floating, float)) else f'{x:^18}' for x in self.cached_results.values()])
        return f"|{header}|\n|{'-----------------:|' * len(self.cached_results)}\n|{values}|"
