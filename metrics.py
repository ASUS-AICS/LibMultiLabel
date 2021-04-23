"""Metrics different to sklearn are placed here.
Some of the functions are from CAML-MIMIC:
(https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py)."""


import numpy as np


def intersect_size(y_true, y_pred, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(y_pred, y_true).sum(axis=axis).astype(float)


def macro_precision(y_true, y_pred):
    num = intersect_size(y_true, y_pred, 0) / (y_pred.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_recall(y_true, y_pred):
    num = intersect_size(y_true, y_pred, 0) / (y_true.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_f1(y_true, y_pred):
    prec = macro_precision(y_true, y_pred)
    rec = macro_recall(y_true, y_pred)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-10)
    return f1


def precision_at_k(y_true, y_pred, k):
    # num true labels in top k predictions / k
    sortd = np.argsort(y_pred)[:,::-1]
    topk = sortd[:,:k]

    # get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y_true[i,tk].sum()
        vals.append(num_true_in_top_k / float(k))

    return np.mean(vals)


def recall_at_k(y_true, y_pred, k):
    # num true labels in top k predictions / num true labels
    sortd = np.argsort(y_pred)[:,::-1]
    topk = sortd[:,:k]

    # get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y_true[i,tk].sum()
        denom = y_true[i,:].sum() + 1e-10
        vals.append(num_true_in_top_k / float(denom))

    return np.mean(vals)
