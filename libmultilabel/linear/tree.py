from typing import Callable, Optional

import numpy as np
import scipy.sparse as sparse
import sklearn.cluster
import sklearn.preprocessing
from tqdm import tqdm

from . import linear

__all__ = ['train_tree']


class Node:
    def __init__(self,
                 label_map: np.ndarray,
                 children: 'list[Node]',
                 metalabels: np.ndarray,
                 ):
        """
        Args:
            label_map (np.ndarray): The labels under this node.
            children (list[Node]): Children of this node. Must be an empty list if this is a leaf node.
            metalabels (np.ndarray): The metalabels corresponding to the labels under this node.
        """
        self.label_map = label_map
        self.children = children
        self.metalabels = metalabels

    def isLeaf(self) -> bool:
        return len(self.children) == 0

    def dfs(self, visit: 'Callable[[Node], None]'):
        visit(self)
        # Stops if self.children is empty, i.e. self is a leaf node
        for child in self.children:
            child.dfs(visit)


class TreeModel:
    def __init__(self,
                 root: Node,
                 flat_model: linear.FlatModel,
                 weight_map: np.ndarray,
                 default_beam_width: int,
                 ):
        self.root = root
        self.flat_model = flat_model
        self.weight_map = weight_map
        self.default_beam_width = default_beam_width

    def predict_values(self, x: sparse.csr_matrix,
                       beam_width: 'Optional[int]' = None
                       ) -> np.ndarray:
        """Calculates the decision values associated with x.

        Args:
            model: A model returned from a training function.
            x (sparse.csr_matrix): A matrix with dimension number of instances * number of features.
            beam_width (int, optional): Number of candidates considered during beam search. If None, uses the default beam width set during training. Defaults to None.

        Returns:
            np.ndarray: A matrix with dimension number of instances * number of classes.
        """
        if beam_width is None:
            beam_width = self.default_beam_width
        allpreds = linear.predict_values(self.flat_model, x)
        return np.vstack([self._beam_search(allpreds[i], beam_width)
                          for i in range(allpreds.shape[0])])

    def _beam_search(self, allpreds: np.ndarray, beam_width: int) -> np.ndarray:
        """Predict with beam search using cached decision values.

        Args:
            allpreds (np.ndarray): Cached decision values of each node.
            beam_width (int): Number of candidates considered.

        Returns:
            np.ndarray: A matrix with dimension 1 * number of classes.
        """
        cur_level = [(self.root, 0.)]   # pairs of (node, score)
        next_level = []
        while True:
            num_internal = sum(
                map(lambda pair: not pair[0].isLeaf(), cur_level))
            if num_internal == 0:
                break

            for node, score in cur_level:
                if node.isLeaf():
                    next_level.append((node, score))
                    continue
                slice = np.s_[self.weight_map[node.index]:
                              self.weight_map[node.index+1]]
                pred = allpreds[slice].ravel()
                child_score = score - np.maximum(0, 1 - pred)**2
                next_level.extend(zip(node.children, child_score.tolist()))

            cur_level = sorted(next_level, key=lambda pair: -
                               pair[1])[:beam_width]
            next_level = []

        scores = np.empty_like(self.root.label_map, dtype='d')
        scores[:] = -np.inf
        for node, score in cur_level:
            slice = np.s_[self.weight_map[node.index]:
                          self.weight_map[node.index+1]]
            pred = allpreds[slice].ravel()
            scores[node.label_map] = np.exp(score - np.maximum(0, 1 - pred)**2)
        return scores


def train_tree(y: sparse.csr_matrix,
               x: sparse.csr_matrix,
               options: str,
               K=100, dmax=10,
               default_beam_width=10,
               verbose: bool = True,
               ) -> TreeModel:
    """Trains a linear model for multiabel data using a divide-and-conquer strategy.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        K (int, optional): Maximum degree of nodes in the tree. Defaults to 100.
        dmax (int, optional): Maximum depth of the tree. Defaults to 10.
        default_beam_width (int, optional): Beam width used in predict_values when none is specified. Defaults to 10.
        verbose (bool, optional): Output extra progress information. Defaults to True.

    Returns:
        A model which can be used in predict_values.
    """
    label_representation = (y.T * x).tocsr()
    label_representation = sklearn.preprocessing.normalize(
        label_representation, norm='l2', axis=1)
    root = _build_tree(label_representation, np.arange(y.shape[1]), 0, K, dmax)

    total = 0

    def count(node):
        nonlocal total
        total += 1
    root.dfs(count)

    pbar = tqdm(total=total, disable=not verbose)

    def visit(node):
        relevant_instances = y[:, node.label_map].getnnz(axis=1) > 0
        _train_node(y[relevant_instances],
                    x[relevant_instances], options, node)
        pbar.update()

    root.dfs(visit)
    pbar.close()

    flat_model, weight_map = _flatten_model(root)
    return TreeModel(root, flat_model, weight_map, default_beam_width)


def _build_tree(label_representation: sparse.csr_matrix,
                label_map: np.ndarray,
                d: int, K: int, dmax: int
                ) -> Node:
    """Builds the tree recursively by kmeans clustering.

    Args:
        label_representation (sparse.csr_matrix): A matrix with dimensions number of classes under this node * number of features.
        label_map (np.ndarray): Maps 0..label_representation.shape[0] to the index in the complete data.
        d (int): Current depth.
        K (int): Maximum degree of nodes in the tree.
        dmax (int): Maximum depth of the tree.

    Returns:
        Node: root of the (sub)tree built from rep.
    """
    if d >= dmax or label_representation.shape[0] <= K:
        return Node(label_map=label_map,
                    children=[],
                    metalabels=np.arange(len(label_map)))

    metalabels = sklearn.cluster.KMeans(
        K,
        random_state=np.random.randint(2**32),
        n_init=1,
        max_iter=300,
        tol=0.0001,
        algorithm='elkan',
    ).fit(label_representation).labels_

    children = []
    for i in range(K):
        child_map = label_map[metalabels == i]
        child_representation = label_representation[metalabels == i]
        child = _build_tree(child_representation,
                            child_map,
                            d + 1, K, dmax)
        children.append(child)

    return Node(label_map=label_map,
                children=children,
                metalabels=np.arange(len(label_map)))


def _train_node(y: sparse.csr_matrix,
                x: sparse.csr_matrix,
                options: str,
                node: Node
                ):
    """If node is internal, creates the metalabels representing each child and trains
    on the metalabels. Otherwise, train on y.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        node (Node): Node to be trained.
    """
    if node.isLeaf():
        node.model = linear.train_1vsrest(
            y[:, node.label_map], x, options, False
        )
    else:
        child_y = [y[:, child.label_map].getnnz(axis=1).reshape(-1, 1) > 0
                   for child in node.children]
        child_y = sparse.csr_matrix(np.hstack(child_y))
        node.model = linear.train_1vsrest(
            child_y, x, options, False
        )


def _flatten_model(root: Node) -> 'tuple[linear.FlatModel, np.ndarray]':
    """Flattens tree weight matrices into a single weight matrix. The flattened weight
    matrix is used to predict all possible values, which is cached for beam search.
    This pessimizes complexity but is faster in practice.

    Args:
        root (Node): Root of the tree.

    Returns:
        tuple[linear.FlatModel, np.ndarray]: The flattened model and the flattened ranges of each node.
    """
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
    weight_map = np.cumsum(
        [0] + list(map(lambda w: w.shape[1], weights)))

    return model, weight_map
