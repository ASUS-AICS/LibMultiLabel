import re

import numpy as np
from torchmetrics import Metric, MetricCollection, F1, Precision, Recall


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
            metrics[metric] = Precision(num_classes, metric_threshold,
                                        average='samples', top_k=int(metric[2:]))
        elif re.match('R@\d+', metric):
            metrics[metric] = Recall(num_classes, metric_threshold,
                                        average='samples', top_k=int(metric[2:]))
        elif metric not in ['Micro-Precision', 'Micro-Recall', 'Micro-F1', 'Macro-F1', 'Another-Macro-F1']:
            raise ValueError(f'Invalid metric: {metric}')

    return  MetricCollection(metrics)


def tabulate_metrics(metric_dict, split):
    msg = f'====== {split} dataset evaluation result ======='
    header = '|'.join([f'{k:^18}' for k in metric_dict.keys()])
    values = '|'.join([f'{x * 100:^18.4f}' if isinstance(x, (np.floating, float)) else f'{x:^18}' for x in metric_dict.values()])
    msg += f"|{header}|\n|{'-----------------:|' * len(metric_dict)}\n|{values}|\n"
    return msg
