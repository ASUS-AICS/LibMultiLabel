"""Metrics modified from CAML-MIMIC are tentatively placed here.
(https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py)
They are used for the internal need to compare with CAML."""


import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def macro_f1(y_true, y_pred):
    prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    f1 = 2 * (prec * rec) / (prec + rec + 1e-10)
    return f1


def precision_at_k(y_true, y_pred_vals, k):
    # num true labels in top k predictions / k
    sortd = np.argsort(y_pred_vals)[:,::-1]
    topk = sortd[:,:k]

    # get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y_true[i,tk].sum()
        vals.append(num_true_in_top_k / k)

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
