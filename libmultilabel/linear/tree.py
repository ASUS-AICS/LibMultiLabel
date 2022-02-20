import numpy as np
import scipy.sparse as sparse

<<<<<<< HEAD
import time

=======
>>>>>>> initial tree implementation
from sklearn.cluster import KMeans

from .linear import train_1vsrest, predict_values


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

class Tree:
    def __init__(self) -> None:
        self.K = 100
        self.dmax = 10   # no mention of value?
        self.beam_width = 10    # no mention of value?

    def train(self,
              y: sparse.csr_matrix,
              x: sparse.csr_matrix,
              options: str
              ) -> None:
        rep = y.T * sparse.hstack([x, y])
        self.root = self._build(rep, np.arange(y.shape[1]), 0)

        def visit(node): return self._train_node(y, x, options, node)

        def dfs(node):
            visit(node)
            for child in node.children:
                dfs(child)

        dfs(self.root)

    def _build(self,
               rep: sparse.csr_matrix,
               labelmap: np.ndarray,
               d: int
               ) -> Node:
        if d >= self.dmax or rep.shape[0] < self.K:
            return Node(labelmap, [], np.arange(len(labelmap)))

        # metalabels = KMeans(self.K).fit(rep).labels_
        metalabels = KMeans(self.K, n_init=1, tol=1e-3).fit(rep).labels_
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
            childy = [np.sum(y[:, child.labelmap], axis=1).reshape(-1, 1) > 0
                      for child in node.children]
            childy = sparse.csr_matrix(np.hstack(childy))
            node.model = train_1vsrest(
                childy, x, options,
            )

    def predict_values(self, x: sparse.csr_matrix) -> np.ndarray:
        num_class = self.root.labelmap.shape[0]
        totalprob = np.ones((x.shape[0], num_class))
        self._beam_search(totalprob, x, self.root)
        return totalprob

    def _beam_search(self, totalprob: np.ndarray, x: sparse.csr_matrix, node: Node):
        pred = predict_values(node.model, x)
        prob = 1 / (1 + np.exp(-pred))
        if node.isLeaf():
            totalprob[:, node.labelmap] *= prob
        else:
            totalprob[:, node.labelmap] *= prob[:, node.metalabels]
            order = np.argsort(pred, axis=1)
            top = order[:, -self.beam_width:]
            for i, child in enumerate(node.children):
                possible = np.sum(top == i, axis=1) > 0
                self._beam_search(totalprob[possible], x[possible], child)
                impossible = np.where(~possible)[0].reshape(-1, 1)
                totalprob[impossible, child.labelmap] = 0
