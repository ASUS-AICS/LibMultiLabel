import os
import re

import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

from .metrics import another_macro_f1, precision_recall_at_ks


def evaluate(model, dataset_loader, monitor_metrics, label_key='label', silent=False):
    """Evaluate model and add the predictions to MultiLabelMetrics.

    Args:
        model (Model): a high level class used to initialize network, predict examples and load/save model
        dataset_loader (DataLoader): pytorch dataloader (torch.utils.data.DataLoader)
        monitor_metrics (list): metrics to monitor while validating
        label_key (str, optional): the key to label in the dataset. Defaults to 'label'.
    """
    progress_bar = tqdm(dataset_loader, disable=silent)
    eval_metric = MultiLabelMetrics(monitor_metrics=monitor_metrics)

    for batch in progress_bar:
        batch_labels = batch[label_key]
        predict_results = model.predict(batch)
        batch_label_scores = predict_results['scores']

        batch_labels = batch_labels.cpu().detach().numpy()
        batch_label_scores = batch_label_scores.cpu().detach().numpy()
        eval_metric.add_values(batch_labels, batch_label_scores)
    return eval_metric


class MultiLabelMetrics():
    def __init__(self, monitor_metrics):
        self.monitor_metrics = monitor_metrics
        self.y_true = []
        self.y_pred = []
        self.cached_results = {}

        self.top_ks = set()
        for metric in self.monitor_metrics:
            if re.match('[P|R]@\d+', metric):
                top_k = int(metric[2:])
                self.top_ks.add(top_k)
            else:
                raise ValueError(f'Invalid metric: {metric}')

    def add_values(self, y_true, y_pred):
        """Add batch of y_true and y_pred.

        Args:
            y_true (ndarray): an array with ground truth labels (shape: batch_size * number of classes)
            y_pred (ndarray): an array with predicted labels (shape: batch_size * number of classes)
        """
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def eval(self, threshold=0.5):
        """Evaluate precision, recall, micro-f1, macro-f1, and P@k/R@k listed in the monitor_metrics."""
        y_true = np.vstack(self.y_true)
        y_pred = np.vstack(self.y_pred)
        report_dict = classification_report(y_true, y_pred > threshold, output_dict=True, zero_division=0)
        result = {
            'Micro-Precision': report_dict['micro avg']['precision'],
            'Micro-Recall': report_dict['micro avg']['recall'],
            'Micro-F1': report_dict['micro avg']['f1-score'],
            'Macro-F1': report_dict['macro avg']['f1-score'],
            'Another-Macro-F1': another_macro_f1(y_true, y_pred > threshold) # caml's macro-f1
        }
        # add metrics like P@k, R@k to the result dict
        scores = precision_recall_at_ks(y_true, y_pred, top_ks=self.top_ks)
        result.update({metric: scores[metric] for metric in self.monitor_metrics})
        self.cached_results = result

    def get_y_pred(self):
        """Convert 3D array (shape: number of batches * batch_size * number of classes
        to 2D array (shape: number of samples * number of classes).
        """
        return np.vstack(self.y_pred)

    def get_metric_dict(self, threshold=0.5, use_cache=True):
        """Evaluate or get score dictionary from cache.

        Args:
            threshold (float, optional): threshold to evaluate precision, recall, and f1 score. Defaults to 0.5.
            use_cache (bool, optional): return a cached results or not. Defaults to True.
        """
        if not use_cache:
            self.eval(threshold)
        return self.cached_results

    def __repr__(self):
        """Return cache results in markdown."""
        header = '|'.join([f'{k:^18}' for k in self.cached_results.keys()])
        values = '|'.join([f'{x * 100:^18.4f}' if isinstance(x, (np.floating, float)) else f'{x:^18}' for x in self.cached_results.values()])
        return f"|{header}|\n|{'-----------------:|' * len(self.cached_results)}\n|{values}|"
