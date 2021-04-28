import logging
import re

import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from metrics import macro_f1, precision_at_k, recall_at_k
from utils import Timer, dump_log, dump_top_k_prediction


def evaluate(config, model, dataset_loader, split='val', dump=True):
    timer = Timer()
    progress_bar = tqdm(dataset_loader)
    eval_metric = MultiLabelMetrics(config)

    for idx, batch in enumerate(progress_bar):
        batch_labels = batch['label']
        predict_results = model.predict(batch)
        batch_label_scores = predict_results['scores']

        batch_labels = batch_labels.cpu().detach().numpy()
        batch_label_scores = batch_label_scores.cpu().detach().numpy()
        eval_metric.add_values(batch_labels, batch_label_scores)

    logging.info(f'Time for evaluating {split} set = {timer.time():.2f} (s)')
    print(eval_metric)
    metrics = eval_metric.get_metrics()

    if dump:
        print(metrics)
        dump_log(config, metrics, split)

    if split == 'test':
        dump_top_k_prediction(config, model.classes, eval_metric.get_y_pred())

    return metrics


class MultiLabelMetrics():
    def __init__(self, config):
        self.config = config
        self.y_true = []
        self.y_pred = []

    def add_values(self, y_true, y_pred):
        """Add batch of y_true and y_pred.

        Parameters:
        y_true (ndarray): a 2D array with ground truth labels (shape: batch_size * number of classes)
        y_pred (ndarray): a 2d array with predicted labels (shape: batch_size * number of classes)
        """
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def eval(self, y_true, y_pred, threshold=0.5):
        result = {
            'Label Size': self.config.num_classes,
            '# Instance': len(y_true),
            'Micro-F1': f1_score(y_true, y_pred > threshold, average='micro'),
            'Macro-F1': f1_score(y_true, y_pred > threshold, average='macro'),
            'Macro*-F1': macro_f1(y_true, y_pred > threshold) # caml's macro-f1
        }

        # add metrics like P@k, R@k to the result dict
        precision_top_ks = list()
        recall_top_ks = list()

        for metric in self.config.monitor_metrics:
            if re.match('P@\d+', metric):
                top_k = int(metric[2:])
                precision_top_ks.append(top_k)
            elif re.match('R@\d+', metric):
                top_k = int(metric[2:])
                recall_top_ks.append(top_k)
            else:
                raise ValueError(f'Invalid metric: {metric}')

        precision_at_k_scores = precision_at_k(y_true, y_pred, top_ks=precision_top_ks)
        recall_at_k_scores = recall_at_k(y_true, y_pred, top_ks=recall_top_ks)

        for i, top_k_score in enumerate(precision_at_k_scores):
            result[f'P@{precision_top_ks[i]}'] = top_k_score
        for i, top_k_score in enumerate(recall_at_k_scores):
            result[f'R@{recall_top_ks[i]}'] = top_k_score

        return result

    def get_y_pred(self):
        """Convert 3D array (shape: number of batches * batch_size * number of classes
        to 2D array (shape: number of samples * number of classes).
        """
        return np.vstack(self.y_pred)

    def get_metrics(self):
        y_true = np.vstack(self.y_true)
        y_pred = np.vstack(self.y_pred)
        return self.eval(y_true, y_pred)

    def __repr__(self):
        result = self.get_metrics()
        header = '|'.join([f'{k:^13}' for k in result.keys()])
        values = '|'.join([f'{x * 100:^13.4f}' if isinstance(x, (np.floating, float)) else f'{x:^13}' for x in result.values()])
        return f"|{header}|\n|{'------------:|' * len(result)}\n|{values}|"
