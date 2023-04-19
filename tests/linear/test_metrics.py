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

from libmultilabel.linear.metrics import NDCG
from libmultilabel.linear.metrics import _DCG

seed = 42
num_instances = 1000
num_labels = 1000
size = (num_instances, num_labels)
rng = default_rng(seed)
target = rng.integers(2, size=size)
preds = rng.random(size)
step = ceil(num_labels / 4)


# compare the results to sklearn.dcg_score
def test_dcg():
    # test the same assertion with increasing k
    for k in list(range(1, num_labels, step)):
        assert dcg_score(target, preds, k=k) == approx(np.mean(_DCG(preds, target, k=k)))

        # test the results with the implementation in torchmetrics
        # Warning: The results of ndcg in torchmetrics didn't match thoses in sklearn within a relative tolerance 1e-6
        # sorted_target = torch.gather(torch.tensor(target), dim=-1,
        #                              index=torch.argsort(torch.tensor(preds), dim=-1, descending=True))[:, :k]


# compare the results to sklearn.ndcg_score
def test_ndcg():
    # test the same assertion with increasing k
    for k in list(range(1, num_labels, step)):
        ndcg = NDCG(k)
        ndcg.update(preds, target)
        assert ndcg_score(target, preds, k=k) == approx(ndcg.compute())


# compare the results to sklearn.ndcg_score calculated through batches
def test_bactched_ndcg():
    batch_size = 10
    num_batches = ceil(num_instances / batch_size)
    # create batches
    batched_target = np.vsplit(target, num_batches)
    batched_preds = np.vsplit(preds, num_batches)

    # test the same assertion with increasing k
    for k in list(range(1, num_labels, step)):
        ndcg = NDCG(k)
        for p, t in zip(batched_preds, batched_target):
            ndcg.update(p, t)
        # batch shouldn't change the original result
        assert ndcg_score(target, preds, k=k) == approx(ndcg.compute())
