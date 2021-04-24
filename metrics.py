"""Metrics different to sklearn are placed here.
Some of the functions are from CAML-MIMIC:
(https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py)."""


import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def macro_f1(y_true, y_pred):
    macro_prec, macro_rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    f1 = 2 * (macro_prec * macro_rec) / (macro_prec + macro_rec + 1e-10)
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
