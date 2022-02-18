import numpy as np
import scipy.sparse as sparse

from sklearn.cluster import KMeans

from .linear import train_1vsrest, predict_values


class Node:
    def __init__(self,
                 labelmap: np.ndarray,
                 children: 'list[Node]'
                 ) -> None:
        self.labelmap = labelmap
        self.children = children


def dfs(node, visit):
    visit(node)
    for child in node.children:
        dfs(child, visit)


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
        dfs(
            self.root,
            lambda node: self._train_node(y, x, options, node),
        )

    def _build(self,
               rep: sparse.csr_matrix,
               labelmap: np.ndarray,
               d: int
               ) -> Node:
        if d >= self.dmax or rep.shape[0] < self.K:
            return Node(labelmap, [])

        clusters = KMeans(self.K).fit(rep).labels_
        labelmaps = [labelmap[clusters == i] for i in range(self.K)]
        reps = [rep[clusters == i] for i in range(self.K)]
        children = [self._build(reps[i], labelmaps[i], d+1)
                    for i in range(self.K)]
        return Node(labelmap, children)

    @staticmethod
    def _train_node(y: sparse.csr_matrix,
                    x: sparse.csr_matrix,
                    options: str,
                    node: Node
                    ):
        if node.children == []:
            node.model = train_1vsrest(
                y[:, node.labelmap], x, options,
            )
        else:
            childy = [np.sum(y[:, child.labelmap], axis=0).reshape(-1, 1) > 0
                      for child in node.children]
            childy = sparse.hstack(childy).tocsr()
            node.model = train_1vsrest(
                childy, x, options,
            )

    def predict_values(self, x: sparse.csr_matrix) -> np.ndarray:
        return self._beam_search(x, self.root)

    def _beam_search(self, x: sparse.csr_matrix, node: Node) -> np.ndarray:
        num_class = self.root.labelmap.shape[0]
        ret = np.zeros((x.shape[0], num_class))
        if x.shape[0] == 0:
            return ret

        pred = predict_values(node.model, x)
        ret[:, node.labelmap] = pred

        if node.children != []:
            ord = np.argsort(pred, axis=1)
            top = ord[:, -self.beam_width:]
            childidx = [np.where(top == i)[0]
                        for i in range(len(node.children))]
            childpred = [self._beam_search(x[childidx], child)
                         for child in node.children]
            for ci, cp in zip(childidx, childpred):
                ret[ci] = ret[ci] * cp

        return ret
