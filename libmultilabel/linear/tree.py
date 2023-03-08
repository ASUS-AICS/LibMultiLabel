from typing import Optional

import numpy as np
import scipy.sparse as sparse
import sklearn.cluster
import sklearn.preprocessing

from . import linear

__all__ = ['train_tree']


class Node:
    def __init__(self,
                 labelmap: np.ndarray,
                 children: 'list[Node]',
                 metalabels: np.ndarray,
                 ) -> None:
        self.labelmap = labelmap
        self.children = children
        self.metalabels = metalabels

    def isLeaf(self) -> bool:
        return len(self.children) == 0

    def dfs(self, visit):
        visit(self)
        for child in self.children:
            child.dfs(visit)


class TreeModel:
    def __init__(self,
                 root: Node,
                 flatModel: linear.FlatModel,
                 weightmap: np.ndarray,
                 default_beam_width: int,
                 ) -> None:
        self.root = root
        self.flatModel = flatModel
        self.weightmap = weightmap
        self.default_beam_width = default_beam_width

    def predict_values(self, x: sparse.csr_matrix,
                       beam_width: 'Optional[int]' = None
                       ) -> np.ndarray:
        if beam_width is None:
            beam_width = self.default_beam_width
        allpreds = linear.predict_values(self.flatModel, x)
        return np.vstack([self._beam_search(allpreds[i], beam_width)
                          for i in range(allpreds.shape[0])])

    def _beam_search(self, allpreds: np.ndarray, beam_width: int) -> np.ndarray:
        cur_level = [(self.root, 0.)]   # pairs of (node, score)
        next_level = []
        while len(list(filter(lambda pair: not pair[0].isLeaf(), cur_level))) > 0:
            for node, score in cur_level:
                if node.isLeaf():
                    next_level.append((node, score))
                    continue
                slice = np.s_[self.weightmap[node.index]:
                              self.weightmap[node.index+1]]
                pred = allpreds[slice].ravel()
                child_score = score - np.maximum(0, 1 - pred)**2
                next_level.extend(zip(node.children, child_score.tolist()))

            cur_level = sorted(next_level, key=lambda pair: -
                               pair[1])[:beam_width]
            next_level = []

        scores = np.empty_like(self.root.labelmap, dtype='d')
        scores[:] = -np.inf
        for node, score in cur_level:
            slice = np.s_[self.weightmap[node.index]:
                          self.weightmap[node.index+1]]
            pred = allpreds[slice].ravel()
            scores[node.labelmap] = np.exp(score - np.maximum(0, 1 - pred)**2)
        return scores


def train_tree(y: sparse.csr_matrix,
               x: sparse.csr_matrix,
               options: str,
               K=100, dmax=10,
               default_beam_width=10,
               ) -> TreeModel:
    rep = (y.T * x).tocsr()
    rep = sklearn.preprocessing.normalize(rep, norm='l2', axis=1)

    root = _build_tree(rep, np.arange(y.shape[1]), 0, K, dmax)

    def visit(node):
        idx = y[:, node.labelmap].getnnz(axis=1) > 0
        return _train_node(y[idx], x[idx], options, node)

    root.dfs(visit)
    flatModel, weightmap = _flatten_model(root)
    return TreeModel(root, flatModel, weightmap, default_beam_width)


def _build_tree(rep: sparse.csr_matrix,
                labelmap: np.ndarray,
                d: int, K: int, dmax: int
                ) -> Node:
    if d >= dmax or rep.shape[0] <= K:
        return Node(labelmap, [], np.arange(len(labelmap)))

    metalabels = sklearn.cluster.KMeans(
        K,
        random_state=np.random.randint(2**32),
        n_init=1,
        max_iter=300,
        tol=0.0001,
        algorithm='elkan',
    ).fit(rep).labels_
    maps = [labelmap[metalabels == i] for i in range(K)]
    reps = [rep[metalabels == i] for i in range(K)]
    children = [_build_tree(reps[i], maps[i], d+1, K, dmax)
                for i in range(K)]
    return Node(labelmap, children, metalabels)


def _train_node(y: sparse.csr_matrix,
                x: sparse.csr_matrix,
                options: str,
                node: Node
                ):
    if node.isLeaf():
        node.model = linear.train_1vsrest(
            y[:, node.labelmap], x, options, False
        )
    else:
        childy = [y[:, child.labelmap].getnnz(axis=1).reshape(-1, 1) > 0
                  for child in node.children]
        childy = sparse.csr_matrix(np.hstack(childy))
        node.model = linear.train_1vsrest(
            childy, x, options, False
        )


def _flatten_model(root) -> 'tuple[linear.FlatModel, np.ndarray]':
    index = 0
    weights = []
    bias = root.model['-B']

    def visit(node):
        assert bias == node.model['-B']
        nonlocal index
        node.index = index
        index += 1
        weights.append(node.model.pop('weights'))

    root.dfs(visit)
    model = linear.FlatModel({
        'weights': np.hstack(weights),
        '-B': bias,
        'threshold': 0
    })
    weightmap = np.cumsum(
        [0] + list(map(lambda w: w.shape[1], weights)))

    return model, weightmap
