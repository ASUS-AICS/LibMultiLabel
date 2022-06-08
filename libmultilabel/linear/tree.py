import numpy as np
import scipy.sparse as sparse

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

        self.root = self._build(rep, np.arange(y.shape[1]), 0)

        def visit(node):
            idx = y[:, node.labelmap].getnnz(axis=1) > 0
            return self._train_node(y[idx], x[idx], options, node)

        self.root.dfs(visit)

    def _build(self,
               rep: sparse.csr_matrix,
               labelmap: np.ndarray,
               d: int
               ) -> Node:
        if d >= self.dmax or rep.shape[0] <= self.K:
            return Node(labelmap, [], np.arange(len(labelmap)))

        metalabels = KMeans(self.K).fit(rep).labels_
        # metalabels = KMeans(self.K, n_init=1, tol=1e-3).fit(rep).labels_
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
        num_class = self.root.labelmap.shape[0]
        totalprob = np.ones((x.shape[0], num_class))
        self._beam_search(totalprob, x, np.arange(x.shape[0]), self.root)
        return totalprob

    def _beam_search(self,
                     totalprob: np.ndarray,
                     x: sparse.csr_matrix,
                     instances: np.ndarray,
                     node: Node):
        pred = predict_values(node.model, x[instances])
        prob = 1 / (1 + np.exp(-pred))
        if node.isLeaf():
            totalprob[np.ix_(instances, node.labelmap)] *= prob
        else:
            totalprob[np.ix_(instances, node.labelmap)] *= prob[:, node.metalabels]
            top = np.argpartition(pred, -self.beam_width,
                                  axis=1)[:, -self.beam_width:]
            for i, child in enumerate(node.children):
                possible = np.sum(top == i, axis=1) > 0
                self._beam_search(totalprob, x, instances[possible], child)
                totalprob[np.ix_(instances[~possible], child.labelmap)] = 0
