import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, multilabel_confusion_matrix, ndcg_score, roc_curve, auc
from tqdm import tqdm

from utils import log
from utils.utils import Timer, dump_log


def evaluate(config, model, dataset_loader, eval_metric, split='val', dump=True):
    timer = Timer()
    metrics = MultiLabelMetric(config.num_class, thresholds=config.metrics_thresholds)
    eval_metric.clear()
    progress_bar = tqdm(dataset_loader)

    for idx, batch in enumerate(progress_bar):
        batch_labels = batch['label']
        predict_results = model.predict(batch)
        batch_label_scores = predict_results['scores']

        batch_labels = batch_labels.cpu().detach().numpy()
        batch_label_scores = batch_label_scores.cpu().detach().numpy()
        metrics.add_batch(batch_labels, batch_label_scores)
        eval_metric.add_batch(batch_labels, batch_label_scores)

        if not config.display_iter or idx % config.display_iter == 0:
            last_metrics = metrics.get_metrics()
            progress_bar.set_postfix(**last_metrics)

    log.info(f'Time for evaluating {split} set = {timer.time():.2f} (s)')
    print(eval_metric)
    metrics = eval_metric.get_metrics()
    if dump:
        dump_log(config, metrics, split)

    return metrics


class FewShotMetrics():
    def __init__(self, config, dataset=None, few_shot_limit=5, target_label='label'):
        # read train / test labels
        # dev is not considered for now
        # test_labels = np.hstack([instance[target_label]
                                 # for instance in dataset['test']])
        # train_labels = np.hstack([instance[target_label]
                                  # for instance in dataset['train']])

        self.config = config
        self.num_class = config.num_class
        # get ALL, Z, F, S
        # unique, counts = np.unique(train_labels, return_counts=True)
        # self.frequent_labels_idx = unique[counts > few_shot_limit].astype(int).tolist()
        # self.few_shot_labels_idx = unique[counts <= few_shot_limit].astype(int).tolist()
        # self.zero_shot_labels_idx = list(set(test_labels) - set(train_labels))

        # label groups
        self.label_groups = [[list(range(self.num_class)), 'ALL']]
        # if len(self.few_shot_labels_idx) > 0:
            # self.label_groups.extend([
                # [self.frequent_labels_idx, 'S'],
                # [self.few_shot_labels_idx, 'F'],
                # [self.zero_shot_labels_idx, 'Z']
            # ])

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

    def filter_instances(self, label_idxs, y_true, y_pred):
        """1. Instances that do not contains labels idxs are removed
           2. Labels that do not contains in the label idxs are removed
        """
        mask_np = np.zeros(self.num_class)
        mask_np[label_idxs] = 1

        valid_instances_idxs = list()
        for i, y in enumerate(y_true):
            if (y * mask_np).sum() > 0:
                valid_instances_idxs.append(i)
        return y_true[valid_instances_idxs][:, label_idxs], y_pred[valid_instances_idxs][:, label_idxs]

    def eval(self, y_true, y_pred, threshold=0.5):
        results = []

        for group_idxs, group_name in self.label_groups:
            result = {'Label Group': group_name, 'Label Size': len(group_idxs)}
            target_y_true, target_y_pred = self.filter_instances(
                group_idxs, y_true, y_pred)
            result['# Instance'] = len(target_y_true)

            # micro/macro f1 of the target groups
            # micro_f1 = f1_score(y_true=target_y_true, y_pred=target_y_pred > threshold, average='micro')
            # macro_f1 = f1_score(y_true=target_y_true, y_pred=target_y_pred > threshold, average='macro')

            result['Micro-F1'] = micro_f1((target_y_pred > threshold).ravel(), target_y_true.ravel())
            result['Macro-F1'] = macro_f1(target_y_pred > threshold, target_y_true)

            # find all metric starts with P(Precition) or R(Recall)
            pattern = re.compile('(?:P|R)@\d+')
            for metric in self.config.monitor_metrics:
                for pr_metric in re.findall(pattern, metric):
                    metric_type, top_k = pr_metric.split('@')
                    top_k = int(top_k)
                    metric_at_k = precision_at_k(target_y_pred, target_y_true, k=top_k) if metric_type == 'P' \
                                    else recall_at_k(target_y_pred, target_y_true, k=top_k)
                    result[pr_metric] = metric_at_k

            results.append(result)

        return results

    def get_metrics(self):
        y_true = np.vstack(self.y_true)
        y_pred = np.vstack(self.y_pred)
        return self.eval(y_true, y_pred)

    def __repr__(self):
        results = self.get_metrics()
        df = pd.DataFrame(results).applymap(
            lambda x: f'{x * 100:.4f}' if isinstance(x, (np.floating, float)) else x)
        return df.to_markdown(index=False)


class MetricTypes:
    F1 = 'F1'
    PRECISION = 'Precision'
    RECALL = 'Recall'
    NDCG = 'NDCG'


class MultiLabelMetric(object):
    def __init__(self, num_class, thresholds=None, n_workers=1):
        if thresholds is None:
            thresholds = [0.5]
        self.num_class = num_class
        self.thresholds = thresholds
        self.n_workers = n_workers
        self.reset()

    def reset(self):
        # multilabel confusion matrix
        self.mcm = [np.zeros((self.num_class, 2, 2), dtype=np.uint32)
                    for _ in self.thresholds]
        self.count = 0
        self.preds = []
        self.targets = []
        self._last_metrics = [{MetricTypes.PRECISION: 0, MetricTypes.RECALL: 0,
                               MetricTypes.F1: 0, MetricTypes.NDCG: 0} for _ in self.thresholds]
        self._ndcg = 0

    @staticmethod
    def ndcg_worker(targets, preds, procnum, return_dict):
        score = ndcg_score(targets, preds)
        return_dict[procnum] = score * len(targets)

    def eval(self, beta=1):
        if not self.preds:
            return self._last_metrics

        # stack collect instances
        preds = np.vstack(self.preds)
        target_one_hot = np.vstack(self.targets)

        self.preds, self.targets = [], []

        batch_size = preds.shape[0]
        worker_batch_size = max(batch_size // self.n_workers, 500)
        self.count += batch_size

        # compute mcm for each thresholds
        for idx, threshold in enumerate(self.thresholds):
            pred_one_hot = preds >= threshold
            batch_mcm = multilabel_confusion_matrix(
                target_one_hot, pred_one_hot)
            self.mcm[idx] += batch_mcm.astype(np.uint64)
            mcm_ = self.mcm[idx]

            metrics = {}

            # with mcm, precision, recall and f1 is trivial
            # https://github.com/scikit-learn/scikit-learn/blob/b194674c4/sklearn/metrics/_classification.py#L1488
            tp_sum = mcm_[:, 1, 1]
            pred_sum = tp_sum + mcm_[:, 0, 1]
            true_sum = tp_sum + mcm_[:, 1, 0]

            # 'micro' average
            tp_sum = np.array([tp_sum.sum()])
            pred_sum = np.array([pred_sum.sum()])
            true_sum = np.array([true_sum.sum()])

            precision = _prf_divide(tp_sum, pred_sum, 'precision',
                                    'predicted', average='micro', warn_for={}, zero_division='warn')
            recall = _prf_divide(tp_sum, true_sum, 'recall',
                                 'true', average='micro', warn_for={}, zero_division='warn')
            beta2 = beta ** 2
            denom = beta2 * precision + recall
            denom[denom == 0.] = 1  # avoid division by 0
            f_score = (1 + beta2) * precision * recall / denom

            metrics[MetricTypes.PRECISION], metrics[MetricTypes.RECALL], metrics[MetricTypes.F1] = float(precision), float(recall), float(f_score)
            self._last_metrics[idx] = metrics  # cache metrics

        return self._last_metrics

    def add_batch(self, targets, preds):
        """
        Add a batch of instances to calculate metrics
        """
        assert len(preds) == len(
            targets), f'Different number of pred({len(preds)}) and target({len(targets)})'
        self.preds.append(np.array(preds))
        self.targets.append(np.array(targets))

    def add(self, target, pred):
        """
        Add instance to calculate metrics
        """
        self.preds.append(np.array(pred))
        self.targets.append(np.array(target))

    def get_metrics(self, beta=1):
        return self.eval(beta=beta)[0]

    def __str__(self):
        """Return matrics plot in markdown language"""
        last_metrics = self.eval()
        repr_ = "| Threshold | Precision | Recall    | F1        |\n" + \
                "| --------- | --------- | --------- | --------- |\n"

        for idx, threshold in enumerate(self.thresholds):
            metrics = last_metrics[idx]
            repr_ += f"| {threshold:<9g} | {metrics[MetricTypes.PRECISION] * 100:<9g} | {metrics[MetricTypes.RECALL] * 100:<9g} | {metrics[MetricTypes.F1] * 100:<9g} |\n"

        return repr_

    def __getitem__(self, idx):
        """Make metrics accessiable by index"""
        return self.eval()[idx]


def _prf_divide(numerator, denominator, metric,
                modifier, average, warn_for, zero_division="warn"):
    """Performs division and handles divide-by-zero.
    On zero-division, sets the corresponding result elements equal to
    0 or 1 (according to ``zero_division``). Plus, if
    ``zero_division != "warn"`` raises a warning.
    The metric, modifier and average arguments are used only for determining
    an appropriate warning.
    """
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator

    if not np.any(mask):
        return result

    # if ``zero_division=1``, set those with denominator == 0 equal to 1
    result[mask] = 0.0 if zero_division in ["warn", 0] else 1.0

    # the user will be removing warnings if zero_division is set to something
    # different than its default value. If we are computing only f-score
    # the warning will be raised only if precision and recall are ill-defined
    if zero_division != "warn" or metric not in warn_for:
        return result

    return result

def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

# ##############
# # AT-K
# ##############

def recall_at_k(yhat_raw, y, k):
    #num true labels in top k predictions / num true labels
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    #get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i,tk].sum()
        denom = y[i,:].sum()
        vals.append(num_true_in_top_k / float(denom))

    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.

    return np.mean(vals)

def precision_at_k(yhat_raw, y, k):
    #num true labels in top k predictions / k
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    #get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i,tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))

    return np.mean(vals)

# ##########################################################################
# #MICRO METRICS: treat every prediction as an individual binary prediction
# ##########################################################################

# def micro_accuracy(yhatmic, ymic):
#     return intersect_size(yhatmic, ymic, 0) / union_size(yhatmic, ymic, 0)

def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)

def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)

def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

def intersect_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)
