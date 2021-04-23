"""Metrics modified from CAML-MIMIC are tentatively placed here.
(https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py)
They are used for the internal need to compare with CAML."""

import numpy as np

def intersection(y_true, y_pred, axis):
    # y_true and y_pred are #data * #labels matrices
    return np.logical_and(y_pred, y_true).sum(axis=axis).astype(float)

def macro_precision(y_true, y_pred):
    precision_per_label = intersection(y_true, y_pred, 0) / (y_pred.sum(axis=0) + 1e-10)
    return np.mean(precision_per_label)

def macro_recall(y_true, y_pred):
    recall_per_label = intersection(y_true, y_pred, 0) / (y_true.sum(axis=0) + 1e-10)
    return np.mean(recall_per_label)

def macro_f1(y_true, y_pred):
    prec = macro_precision(y_true, y_pred)
    rec = macro_recall(y_true, y_pred)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-10)
    return f1

def precision_at_k(y_true, y_pred_vals, k):
    # num true labels in top k predictions / k
    sortd = np.argsort(y_pred_vals)[:,::-1]
    topk = sortd[:,:k]

    # get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y_true[i,tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))

    return np.mean(vals)

def recall_at_k(y_true, y_pred_vals, k):
    # num true labels in top k predictions / num true labels
    sortd = np.argsort(y_pred_vals)[:,::-1]
    topk = sortd[:,:k]

    # get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y_true[i,tk].sum()
        denom = y_true[i,:].sum() + 1e-10
        vals.append(num_true_in_top_k / float(denom))

    return np.mean(vals)
