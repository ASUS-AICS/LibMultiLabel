"""Metrics modified from CAML-MIMIC are tentatively placed here.
(https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py)
They are used for the internal need to compare with CAML."""


import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def macro_f1(y_true, y_pred):
    macro_prec, macro_rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    f1 = 2 * (macro_prec * macro_rec) / (macro_prec + macro_rec + 1e-10)
    return f1


def precision_at_k(y_true, y_pred_vals, top_ks):
    rank_mat = np.argsort(-y_pred_vals)
    scores = list()
    for k in top_ks:
        y_pred = np.take_along_axis(y_true, rank_mat[:,:k], axis=1)
        score = np.mean(np.sum(y_pred, axis=1) / k).item()
        scores.append(score)
    return scores


def recall_at_k(y_true, y_pred_vals, top_ks):
    # num true labels in top k predictions / num true labels
    rank_mat = np.argsort(-y_pred_vals)
    scores = list()
    for k in top_ks:
        fp = np.take_along_axis(y_true, rank_mat[:, :k], axis=1).sum(axis=1)
        denom = y_true.sum(axis=1) + 1e-10
        score = np.mean(fp / denom).item()
        scores.append(score)
    return scores
