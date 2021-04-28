"""Metrics modified from CAML-MIMIC are tentatively placed here.
(https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py)
They are used for the internal need to compare with CAML."""


import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def macro_f1(y_true, y_pred):
    macro_prec, macro_rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    f1 = 2 * (macro_prec * macro_rec) / (macro_prec + macro_rec + 1e-10)
    return f1


def precision_recall_at_ks(y_true, y_pred_vals, top_ks):
    y_pred_ranked_idx = np.argsort(-y_pred_vals)
    n_pos = y_true.sum(axis=1)
    scores = {}
    for k in top_ks:
        n_pos_in_top_k = np.take_along_axis(y_true, y_pred_ranked_idx[:,:k], axis=1).sum(axis=1)
        scores[f'P@{k}'] = np.mean(n_pos_in_top_k / k).item()  # precision at k
        scores[f'R@{k}'] = np.mean(n_pos_in_top_k / (n_pos + 1e-10)).item()  # recall at k
    return scores
