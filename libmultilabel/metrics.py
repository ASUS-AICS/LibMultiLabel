import re

import torch
import numpy as np
from torchmetrics import Metric, MetricCollection, F1, Precision, Recall, RetrievalNormalizedDCG
from torchmetrics.utilities.data import select_topk


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
        binary_topk_preds = select_topk(preds, self.top_k)
        target = target.to(dtype=torch.int)
        n_relevant = torch.sum(binary_topk_preds & target, dim=-1)
        top_ks = torch.tensor([self.top_k]*preds.shape[0]).to(preds.device)
        self.score += torch.nan_to_num(
            n_relevant / torch.min(top_ks, target.sum(dim=-1)),
            posinf=0.
        ).sum()
        self.num_sample += len(preds)

    def compute(self):
        return self.score.float() / self.num_sample


def get_metrics(metric_threshold, monitor_metrics, num_classes):
    macro_prec = Precision(num_classes, metric_threshold, average='macro')
    macro_recall = Recall(num_classes, metric_threshold, average='macro')
    another_macro_f1 = 2 * (macro_prec * macro_recall) / \
        (macro_prec + macro_recall + 1e-10)
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
        if isinstance(metric, Metric):  # customized metric
            metrics[type(metric).__name__] = metric
        elif re.match('P@\d+', metric):
            metrics[metric] = Precision(
                num_classes, average='samples', top_k=int(metric[2:]))
        elif re.match('R@\d+', metric):
            metrics[metric] = Recall(
                num_classes, average='samples', top_k=int(metric[2:]))
        elif re.match('RP@\d+', metric):
            metrics[metric] = RPrecision(top_k=int(metric[3:]))
        elif re.match('nDCG@\d+', metric):
            metrics[metric] = RetrievalNormalizedDCG(k=int(metric[5:]))

        elif metric not in ['Micro-Precision', 'Micro-Recall', 'Micro-F1', 'Macro-F1', 'Another-Macro-F1']:
            raise ValueError(f'Invalid metric: {metric}')

    return MetricCollection(metrics)


def tabulate_metrics(metric_dict, split):
    msg = f'====== {split} dataset evaluation result =======\n'
    header = '|'.join([f'{k:^18}' for k in metric_dict.keys()])
    values = '|'.join([f'{x * 100:^18.4f}' if isinstance(x, (np.floating, float)) else f'{x:^18}' for x in metric_dict.values()])
    msg += f"|{header}|\n|{'-----------------:|' * len(metric_dict)}\n|{values}|\n"
    return msg
