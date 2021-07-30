import re

import torch
import numpy as np
from torchmetrics import Metric, MetricCollection, F1, Precision, Recall


class RPrecision(Metric):
    def __init__(
        self,
        top_k,
        dist_sync_on_step=False
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.top_k = top_k
        self.add_state("score", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("num_sample", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        assert preds.shape == target.shape
        for y_pred, y_true in zip(preds, target):
            self.score += self.ranking_rprecision_score(y_pred, y_true)
        self.num_sample += len(preds)

    def compute(self):
        return self.score.float() / self.num_sample

    def ranking_rprecision_score(self, y_pred, y_true):
        n_pos = torch.sum(y_true == 1)
        order = torch.argsort(-y_pred)
        y_true = torch.take(y_true, order[:self.top_k])
        n_relevant = torch.sum(y_true == 1)

        return 0. if min(self.top_k, n_pos) == 0 else n_relevant / min(self.top_k, n_pos)


class NDCG(Metric):
    def __init__(
        self,
        top_k,
        dist_sync_on_step=False
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.top_k = top_k
        self.add_state("score", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("num_sample", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        assert preds.shape == target.shape
        for y_pred, y_true in zip(preds, target):
            self.score += self.ndcg_score(y_pred, y_true)
        self.num_sample += len(preds)

    def compute(self):
        return self.score.float() / self.num_sample

    def ndcg_score(self, y_pred, y_true):
        best = self.dcg_score(y_true, y_true)
        actual = self.dcg_score(y_pred, y_true)
        return actual / best

    def dcg_score(self, y_pred, y_true):
        order = torch.argsort(-y_pred)
        y_true = torch.take(y_true, order[:self.top_k])
        gains = 2 ** y_true - 1
        discounts = torch.log2(torch.arange(len(y_true)) + 2).cuda()
        return torch.sum(gains / discounts)


def get_metrics(metric_threshold, monitor_metrics, num_classes):
    macro_prec = Precision(num_classes, metric_threshold, average='macro')
    macro_recall = Recall(num_classes, metric_threshold, average='macro')
    another_macro_f1 = 2 * (macro_prec * macro_recall) / (macro_prec + macro_recall + 1e-10)
    metrics = {
        'Micro-Precision': Precision(num_classes, metric_threshold, average='micro'),
        'Micro-Recall': Recall(num_classes, metric_threshold, average='micro'),
        'Micro-F1': F1(num_classes, metric_threshold, average='micro'),
        'Macro-F1': F1(num_classes, metric_threshold, average='macro'),
        # The f1 value of macro_precision and macro_recall. This variant of
        # macro_f1 is less preferred but is used in some works. Please
        # refer to Opitz et al. 2019 [https://arxiv.org/pdf/1911.03347.pdf]
        'Another-Macro-F1': another_macro_f1,
    }
    for metric in monitor_metrics:
        if isinstance(metric, Metric): # customized metric
            metrics[type(metric).__name__] = metric
        elif re.match('P@\d+', metric):
            metrics[metric] = Precision(num_classes, average='samples', top_k=int(metric[2:]))
        elif re.match('R@\d+', metric):
            metrics[metric] = Recall(num_classes, average='samples', top_k=int(metric[2:]))
        elif re.match('RP@\d+', metric):
            metrics[metric] = RPrecision(top_k=int(metric[3:]))
        elif re.match('nDCG@\d+', metric):
            metrics[f'{metric}'] = NDCG(top_k=int(metric[5:]))

        elif metric not in ['Micro-Precision', 'Micro-Recall', 'Micro-F1', 'Macro-F1', 'Another-Macro-F1']:
            raise ValueError(f'Invalid metric: {metric}')

    return  MetricCollection(metrics)


def tabulate_metrics(metric_dict, split):
    msg = f'====== {split} dataset evaluation result =======\n'
    header = '|'.join([f'{k:^18}' for k in metric_dict.keys()])
    values = '|'.join([f'{x * 100:^18.4f}' if isinstance(x, (np.floating, float)) else f'{x:^18}' for x in metric_dict.values()])
    msg += f"|{header}|\n|{'-----------------:|' * len(metric_dict)}\n|{values}|\n"
    return msg
