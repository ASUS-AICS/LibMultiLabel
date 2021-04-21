import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

from metrics import macro_f1, precision_at_k, recall_at_k
from utils import log
from utils.utils import Timer, dump_log


def evaluate(config, model, dataset_loader, eval_metric, split='val', dump=True):
    timer = Timer()
    eval_metric.clear()
    progress_bar = tqdm(dataset_loader)

    for idx, batch in enumerate(progress_bar):
        batch_labels = batch['label']
        predict_results = model.predict(batch)
        batch_label_scores = predict_results['scores']

        batch_labels = batch_labels.cpu().detach().numpy()
        batch_label_scores = batch_label_scores.cpu().detach().numpy()
        eval_metric.add_batch(batch_labels, batch_label_scores)

    log.info(f'Time for evaluating {split} set = {timer.time():.2f} (s)')
    print(eval_metric)
    metrics = eval_metric.get_metrics()
    if dump:
        dump_log(config, metrics, split)

    return metrics


class FewShotMetrics():
    def __init__(self, config, dataset):
        self.config = config
        self.clear()

    def clear(self):
        self.y_true = []
        self.y_pred = []

    def add(self, y_true, y_pred):
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def add_batch(self, y_true, y_pred):
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def eval(self, y_true, y_pred, threshold=0.5):
        result = {
            'Label Size': self.config.num_classes,
            '# Instance': len(y_true),
            'Micro-F1': f1_score(y_true, y_pred > threshold, average='micro'), # sklearn's micro-f1
            'Macro-F1': macro_f1(y_true, y_pred > threshold) # caml's macro-f1
        }

        # add metric like P@k, R@k to the result dict
        pattern = re.compile('(?:P|R)@\d+')
        for metric in self.config.monitor_metrics:
            for pr_metric in re.findall(pattern, metric):
                metric_type, top_k = pr_metric.split('@')
                top_k = int(top_k)
                metric_at_k = precision_at_k(y_true, y_pred, k=top_k) if metric_type == 'P' \
                    else recall_at_k(y_true, y_pred, k=top_k)
                result[pr_metric] = metric_at_k

        return result

    def get_metrics(self):
        y_true = np.vstack(self.y_true)
        y_pred = np.vstack(self.y_pred)
        return self.eval(y_true, y_pred)

    def __repr__(self):
        results = self.get_metrics()
        print(f'results: {results}')
        df = pd.DataFrame([results]).applymap(
            lambda x: f'{x * 100:.4f}' if isinstance(x, (np.floating, float)) else x)
        return df.to_markdown(index=False)
