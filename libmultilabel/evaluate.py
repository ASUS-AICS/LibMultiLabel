import logging
import re

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
from tqdm import tqdm

from .metrics import another_macro_f1, precision_recall_at_ks
from .utils import Timer, dump_log, save_top_k_prediction


def evaluate(model, dataset_loader, num_classes, monitor_metrics, label_key='label', predict_out_path=None, save_k_predictions=0):
    """Evaluate model and save top-k prediction to file if the number of top_k_prediction > 0.

    Args:
        model (Model): a high level class used to initialize network, predict examples and load/save model
        dataset_loader (DataLoader)
        num_classes (int): the number of the labels
        monitor_metrics (list): metrics to monitor while validating
        label_key (str, optional): the key to label in the dataset. Defaults to 'label'.
        predict_out_path (str, optional): path to file for saving the top k predictions. Defaults to None.
        save_k_predictions (int, optional): top k label scores to save for each sample. Defaults to 0.
    """
    timer = Timer()
    progress_bar = tqdm(dataset_loader)
    eval_metric = MultiLabelMetrics(num_classes, monitor_metrics=monitor_metrics)

    for batch in progress_bar:
        batch_labels = batch[label_key] # set label_key in os.env or sync naming
        predict_results = model.predict(batch)
        batch_label_scores = predict_results['scores']

        batch_labels = batch_labels.cpu().detach().numpy()
        batch_label_scores = batch_label_scores.cpu().detach().numpy()
        eval_metric.add_values(batch_labels, batch_label_scores)

    metrics = eval_metric.get_metrics()
    print(eval_metric)
    logging.info(f'Time for evaluating = {timer.time():.2f} (s)')

    if save_k_predictions > 0:
        save_top_k_prediction(model.classes, eval_metric.get_y_pred(), predict_out_path, k=save_k_predictions)

    return metrics


class MultiLabelMetrics():
    def __init__(self, num_classes, monitor_metrics):
        self.num_classes = num_classes
        self.monitor_metrics = monitor_metrics
        self.y_true = []
        self.y_pred = []
        self.cache_result = {}

    def add_values(self, y_true, y_pred):
        """Add batch of y_true and y_pred.

        Args:
            y_true (ndarray): a 2D array with ground truth labels (shape: batch_size * number of classes)
            y_pred (ndarray): a 2d array with predicted labels (shape: batch_size * number of classes)
        """
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def eval(self, y_true, y_pred, threshold=0.5):
        """Evaluate precision, recall, micro-f1, macro-f1, and P@k/R@k listed in the monitor_metrics."""
        y_true = np.vstack(self.y_true)
        y_pred = np.vstack(self.y_pred)
        precision, recall, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred > threshold, average='micro')
        result = {
            'Precision': precision,
            'Recall': recall,
            'Micro-F1': micro_f1,
            'Macro-F1': f1_score(y_true, y_pred > threshold, average='macro'),
            'Another-Macro-F1': another_macro_f1(y_true, y_pred > threshold) # caml's macro-f1
        }
        # add metrics like P@k, R@k to the result dict
        top_ks = set()
        for metric in self.monitor_metrics:
            if re.match('[P|R]@\d+', metric):
                top_k = int(metric[2:])
                top_ks.add(top_k)
            else:
                raise ValueError(f'Invalid metric: {metric}')
        scores = precision_recall_at_ks(y_true, y_pred, top_ks=top_ks)
        result.update({metric: scores[metric] for metric in self.monitor_metrics})
        self.cache_result = result

    def get_y_pred(self):
        """Convert 3D array (shape: number of batches * batch_size * number of classes
        to 2D array (shape: number of samples * number of classes).
        """
        return np.vstack(self.y_pred)

    def get_metrics(self, use_cache=False):
        if not use_cache:
            self.eval(self.y_true, self.y_pred)
        return self.cache_result

    def __repr__(self):
        """Return cache results in markdown."""
        header = '|'.join([f'{k:^20}' for k in self.cache_result.keys()])
        values = '|'.join([f'{x * 100:^20.4f}' if isinstance(x, (np.floating, float)) else f'{x:^20}' for x in self.cache_result.values()])
        return f"|{header}|\n|{'-------------------:|' * len(self.cache_result)}\n|{values}|"
