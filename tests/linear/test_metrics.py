import sys
from math import ceil

import numpy as np
# import torch
# from torchmetrics.functional.retrieval.ndcg import _dcg
from pytest import approx
from sklearn.metrics import ndcg_score
from sklearn.metrics import dcg_score
from numpy.random import default_rng
from pathlib import Path
sys.path.append(Path('...').resolve())

from libmultilabel import linear
from libmultilabel.linear.metrics import NDCG
from libmultilabel.linear.metrics import _DCG

seed = 42
num_instances = 1000
num_labels = 1000
size = (num_instances, num_labels)
rng = default_rng(seed)
target = rng.integers(2, size=size)
preds = rng.random(size)
step = ceil(num_labels / 10)

def test_dcg():
    for k in list(range(1, num_labels, step)):
        assert dcg_score(target, preds, k=k) == approx(np.mean(_DCG(preds, target, k=k))), 1111

        # torch
        # sorted_target = torch.gather(torch.tensor(target), dim=-1,
        #                              index=torch.argsort(torch.tensor(preds), dim=-1, descending=True))[:, :k]

        # assert dcg_score(true_relevance, scores) == _dcg(sorted_target).tolist()[0]


def test_ndcg():
    for k in list(range(1, num_labels, step)):
        ndcg = NDCG(k)
        ndcg.update(preds, target)
        assert ndcg_score(target, preds, k=k) == approx(ndcg.compute())


def test_bactched_ndcg():
    batch_size = 10
    num_batches = ceil(num_instances / batch_size)
    batched_target = np.vsplit(target, num_batches)
    batched_preds = np.vsplit(preds, num_batches)

    for k in list(range(1, num_labels, step)):
        scores = 0
        num_sklearn_instances = 0
        ndcg = NDCG(k)
        for p, t in zip(batched_preds, batched_target):
            ndcg.update(p, t)
            scores += len(p) * ndcg_score(t, p, k=k)
            num_sklearn_instances += len(p)
        assert num_sklearn_instances == num_instances
        assert approx(ndcg.compute()) == scores / num_sklearn_instances
