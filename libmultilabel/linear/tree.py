from collections import deque
from itertools import chain

import numpy as np
import scipy.sparse as sparse
from sklearn.cluster import KMeans, MiniBatchKMeans

from .linear import predict_values, train_1vsrest


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


class Tree:
    def __init__(self, K=100, dmax=10, beam_width=10) -> None:
        self.K = K
        self.dmax = dmax   # no mention of value?
        self.beam_width = beam_width    # no mention of value?

    def train(self,
              y: sparse.csr_matrix,
              x: sparse.csr_matrix,
              options: str
              ) -> None:
        rep = (y.T * x).tocsr()

        import time
        start = time.time()
        self.root = self._build(rep, np.arange(y.shape[1]), 0)
        print(f'building in {time.time() - start:.2f}s')

        def visit(node):
            idx = y[:, node.labelmap].getnnz(axis=1) > 0
            assert(np.all(y[idx].getnnz(axis=1) > 0))
            return self._train_node(y[idx], x[idx], options, node)

        self.root.dfs(visit)

    def _build(self,
               rep: sparse.csr_matrix,
               labelmap: np.ndarray,
               d: int
               ) -> Node:
        if d >= self.dmax or rep.shape[0] <= self.K:
            return Node(labelmap, [], np.arange(len(labelmap)))

        # metalabels = KMeans(self.K).fit(rep).labels_
        metalabels = MiniBatchKMeans(
            self.K, random_state=np.random.randint(2**32)).fit(rep).labels_
        maps = [labelmap[metalabels == i] for i in range(self.K)]
        reps = [rep[metalabels == i] for i in range(self.K)]
        children = [self._build(reps[i], maps[i], d+1)
                    for i in range(self.K)]
        return Node(labelmap, children, metalabels)

    @staticmethod
    def _train_node(y: sparse.csr_matrix,
                    x: sparse.csr_matrix,
                    options: str,
                    node: Node
                    ):
        if node.isLeaf():
            node.model = train_1vsrest(
                y[:, node.labelmap], x, options,
            )
        else:
            childy = [y[:, child.labelmap].getnnz(axis=1).reshape(-1, 1) > 0
                      for child in node.children]
            childy = sparse.csr_matrix(np.hstack(childy))
            node.model = train_1vsrest(
                childy, x, options,
            )

    def predict_values(self, x: sparse.csr_matrix) -> np.ndarray:
        return np.vstack([self._beam_search(x[i]) for i in range(x.shape[0])])

    def _beam_search(self, x: sparse.csr_matrix) -> np.ndarray:
        cur_level = [(self.root, 0.)]
        next_level = []
        while len(list(filter(lambda pair: not pair[0].isLeaf(), cur_level))) > 0:
            for node, score in cur_level:
                if node.isLeaf():
                    next_level.append((node, score))
                    continue
                pred = predict_values(node.model, x).ravel()
                child_score = score - np.log(1 + np.exp(-pred))
                next_level.extend(zip(node.children, child_score.tolist()))

            cur_level = sorted(next_level, key=lambda pair: -
                               pair[1])[:self.beam_width]
            next_level = []

        scores = np.empty_like(self.root.labelmap, dtype='d')
        scores[:] = -np.inf
        for node, score in cur_level:
            scores[node.labelmap] = predict_values(node.model, x) + score
        return scores
