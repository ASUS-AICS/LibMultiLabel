import numpy as np
import scipy.sparse as sparse

from liblinear.liblinearutil import train

__all__ = ['train_1vsrest', 'evaluate']

def train_1vsrest(y: sparse.csr_matrix, x: sparse.csr_matrix, options: str):
    """
    Trains a linear model for multiabel data using a one-vs-all strategy.

    Returns the model.

    y is a 0/1 matrix with dimensions number of instances * number of classes
    x is a matrix with dimensions number of instances * number of features
    options is the option string passed to liblinear
    """
    if options.find('-R') != -1:
        raise ValueError('-R is not supported')

    bias = -1.
    if options.find('-B') != -1:
        options_split = options.split()
        i = options_split.index('-B')
        bias = float(options_split[i+1])
        options = ' '.join(options_split[:i] + options_split[i+2:])
        x = sparse.hstack([
            x,
            np.full((x.shape[0], 1), bias),
        ], 'csr')

    y = y.tocsc()
    nr_class = y.shape[1]
    n = x.shape[1]
    weights = np.zeros((n, nr_class), order='F')
    for i in range(nr_class):
        yi = y[:, i].toarray().reshape(-1)
        modeli = train(yi, x, options)
        w = np.ctypeslib.as_array(modeli.w, (n,))
        # liblinear label mapping depends on data, we ensure
        # it is the same for all labels
        if modeli.get_labels()[0] == 0:
            w = -w
        weights[:, i] = w

    return {'weights': np.asmatrix(weights), '-B': bias}

def evaluate(model, x: sparse.csr_matrix) -> np.ndarray:
    """
    Calculates the decision values associated with x.

    Returns a 0/1 matrix with dimension number of instances * number of classes

    x is a matrix with dimension number of instances * number of features
    """
    bias = model['-B']
    bias_col = np.full((x.shape[0], 1 if bias > 0 else 0), bias)
    n = model['weights'].shape[0]
    n -= 1 if bias > 0 else 0
    if x.shape[1] < n:
        x = sparse.hstack([
            x,
            np.zeros((x.shape[0], n - x.shape[1])),
            bias_col,
        ], 'csr')
    else:
        x = sparse.hstack([
            x[:, :n],
            bias_col,
        ], 'csr')

    return (x * model['weights']).A
