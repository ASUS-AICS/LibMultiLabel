import numpy as np
import scipy.sparse as sparse
import sklearn.cluster
import sklearn.preprocessing

from . import linear


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
        self.dmax = dmax
        self.beam_width = beam_width

    def train(self,
              y: sparse.csr_matrix,
              x: sparse.csr_matrix,
              options: str,
              ) -> None:
        rep = (y.T * x).tocsr()
        rep = sklearn.preprocessing.normalize(rep, norm='l2', axis=1)

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

        metalabels = sklearn.cluster.KMeans(
            self.K,
            random_state=np.random.randint(2**32),
            n_init=1,
            max_iter=300,
            tol=0.0001,
            algorithm='lloyd',
        ).fit(rep).labels_
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

    def predict_values(self, x: sparse.csr_matrix) -> np.ndarray:
        return np.vstack([self._beam_search(x[i]) for i in range(x.shape[0])])

    def _beam_search(self, x: sparse.csr_matrix) -> np.ndarray:
        """
        A path is a sequence of nodes where the first is the root and every
        node except the root is a child of the previous node.
        A partial path is a path where the final node is an internal node.
        A complete path is a path where the final node is a leaf node.

        Beam search is a greedy algorithm looking for paths with the highest scores.
        At each level, beam search keeps track of beam_width paths
        with the highest scores in cur_level.
        For each partial path in cur_level, concatenate the children of the
        final node to create a set of paths and add them to next_level.
        Their scores are the score of the partial path + the score of the child.
        For each complete path in cur_level, add them to next_level.
        Set cur_level to the beam_width highest scored paths in next_level and repeat
        until all paths in cur_level is complete.
        """
        cur_level = [(self.root, 0.)]   # pairs of (node, score)
        next_level = []
        while len(list(filter(lambda pair: not pair[0].isLeaf(), cur_level))) > 0:
            for node, score in cur_level:
                if node.isLeaf():
                    next_level.append((node, score))
                    continue
                pred = linear.predict_values(node.model, x).ravel()
                child_score = score - np.log(1 + np.exp(-pred))
                next_level.extend(zip(node.children, child_score.tolist()))

            cur_level = sorted(next_level, key=lambda pair: -
                               pair[1])[:self.beam_width]
            next_level = []

        scores = np.empty_like(self.root.labelmap, dtype='d')
        scores[:] = -np.inf
        for node, score in cur_level:
            pred = linear.predict_values(node.model, x).ravel()
            scores[node.labelmap] = score - np.log(1 + np.exp(-pred))
        return -np.log(np.exp(-scores) - 1)


class CachedTree(Tree):
    def predict_values(self, x: sparse.csr_matrix) -> np.ndarray:
        if not hasattr(self, 'model'):
            self._build_model()
        allpreds = linear.predict_values(self.model, x)
        return np.vstack([self._beam_search(allpreds[i]) for i in range(allpreds.shape[0])])

    def _build_model(self) -> None:
        index = 0
        weights = []
        bias = self.root.model['-B']

        def visit(node):
            assert bias == node.model['-B']
            nonlocal index
            node.index = index
            index += 1
            weights.append(node.model.pop('weights'))

        self.root.dfs(visit)
        self.model = {
            'weights': np.hstack(weights),
            '-B': bias,
            'threshold': 0
        }
        self.weightmap = np.cumsum(
            [0] + list(map(lambda w: w.shape[1], weights)))

    def _beam_search(self, allpreds: np.ndarray) -> np.ndarray:
        """ Same as Tree._beam_search except this uses cached decision
        values instead of computing on the fly.
        """
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
                child_score = score - np.log(1 + np.exp(-pred))
                next_level.extend(zip(node.children, child_score.tolist()))

            cur_level = sorted(next_level, key=lambda pair: -
                               pair[1])[:self.beam_width]
            next_level = []

        scores = np.empty_like(self.root.labelmap, dtype='d')
        scores[:] = -np.inf
        for node, score in cur_level:
            slice = np.s_[self.weightmap[node.index]:
                          self.weightmap[node.index+1]]
            pred = allpreds[slice].ravel()
            scores[node.labelmap] = score - np.log(1 + np.exp(-pred))
        return -np.log(np.exp(-scores) - 1)


class GlobalTree(Tree):
    def predict_values(self, x: sparse.csr_matrix) -> np.ndarray:
        if not hasattr(self, 'model'):
            self._build_model()
        return linear.predict_values(self.model, x)

    def _build_model(self) -> None:
        num_features = self.root.model['weights'].shape[0]
        num_labels = self.root.labelmap.shape[0]
        weights = np.zeros((num_features, num_labels))
        bias = self.root.model['-B']

        def visit(node):
            assert bias == node.model['-B']
            nodeweights = node.model.pop('weights')
            if node.isLeaf():
                weights[:, node.labelmap] += nodeweights
            else:
                for i in range(len(node.children)):
                    childlabels = node.labelmap[node.metalabels == i]
                    weights[:, childlabels] += nodeweights[:, i]

        self.root.dfs(visit)
        self.model = {
            'weights': np.asmatrix(weights),
            '-B': bias,
            'threshold': 0
        }


class WeirdLossTree(CachedTree):
    def _beam_search(self, allpreds: np.ndarray) -> np.ndarray:
        """ Same as Tree._beam_search except this uses cached decision
        values instead of computing on the fly.
        """
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
                               pair[1])[:self.beam_width]
            next_level = []

        scores = np.empty_like(self.root.labelmap, dtype='d')
        scores[:] = -np.inf
        for node, score in cur_level:
            slice = np.s_[self.weightmap[node.index]:
                          self.weightmap[node.index+1]]
            pred = allpreds[slice].ravel()
            scores[node.labelmap] = np.exp(score - np.maximum(0, 1 - pred)**2)
        return scores
