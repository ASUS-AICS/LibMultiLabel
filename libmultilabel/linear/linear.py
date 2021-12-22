import os
import numpy as np
import scipy.sparse as sparse

from liblinear.liblinearutil import train

__all__ = ['train_1vsrest',
           'train_thresholding',
           'train_cost_sensitive',
           'predict_values']


def train_1vsrest(y: sparse.csr_matrix, x: sparse.csr_matrix, options: str):
    """Trains a linear model for multiabel data using a one-vs-rest strategy.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.

    Returns:
        A model which can be used in predict_values.
    """
    # Follows the MATLAB implementation at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/multilabel/
    if any(o in options for o in ['-R', '-C', '-v']):
        raise ValueError('-R, -C and -v are not supported')

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

    if not '-q' in options:
        options += ' -q'

    y = y.tocsc()
    num_class = y.shape[1]
    num_feature = x.shape[1]
    weights = np.zeros((num_feature, num_class), order='F')
    for i in range(num_class):
        yi = y[:, i].toarray().reshape(-1)
        modeli = train(2*yi - 1, x, options)
        w = np.ctypeslib.as_array(modeli.w, (num_feature,))
        weights[:, i] = w

    return {'weights': np.asmatrix(weights), '-B': bias, 'threshold': 0}


def train_thresholding(y: sparse.csr_matrix, x: sparse.csr_matrix, options: str):
    """Trains a linear model for multilabel data using a one-vs-rest strategy
    and cross-validation to pick an optimal decision threshold for Macro-F1.
    Outperforms train_1vsrest in most aspects at the cost of higher
    time complexity.
    See user guide for more details.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.

    Returns:
        A model which can be used in predict_values.
    """
    # Follows the MATLAB implementation at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/multilabel/
    if any(o in options for o in ['-R', '-C', '-v']):
        raise ValueError('-R, -C and -v are not supported')

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
    num_class = y.shape[1]
    num_feature = x.shape[1]
    weights = np.zeros((num_feature, num_class), order='F')
    thresholds = np.zeros(num_class)
    for i in range(num_class):
        yi = y[:, i].toarray().reshape(-1)
        w, t = thresholding_one_label(2*yi - 1, x, options)
        weights[:, i] = w.ravel()
        thresholds[i] = t

    return {'weights': np.asmatrix(weights), '-B': bias, 'threshold': thresholds}


def thresholding_one_label(y: np.ndarray,
                           x: sparse.csr_matrix,
                           options: str
                           ) -> 'tuple[np.ndarray, float]':
    """Outer cross-validation for thresholding on a single label.

    Args:
        y (np.ndarray): A +1/-1 array with dimensions number of instances * 1.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.

    Returns:
        tuple[np.ndarray, float]: tuple of the weights and threshold.
    """
    fbr_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    nr_fold = 3

    l = y.shape[0]

    perm = np.random.permutation(l)

    f_list = np.zeros_like(fbr_list)

    for fold in range(nr_fold):
        mask = np.zeros_like(perm, dtype='?')
        mask[np.arange(int(fold*l/nr_fold), int((fold+1)*l/nr_fold))] = 1
        val_idx = perm[mask]
        train_idx = perm[mask != True]

        scutfbr_w, scutfbr_b_list = scutfbr(
            y[train_idx], x[train_idx], fbr_list, options)
        wTx = (x[val_idx] * scutfbr_w).A1

        for i in range(fbr_list.size):
            F = fmeasure(y[val_idx], 2*(wTx > -scutfbr_b_list[i]) - 1)
            f_list[i] += F

    best_fbr = fbr_list[::-1][np.argmax(f_list[::-1])]  # last largest
    if np.max(f_list) == 0:
        best_fbr = np.min(fbr_list)

    # final model
    w, b_list = scutfbr(y, x, np.array([best_fbr]), options)

    return w, b_list[0]


def scutfbr(y: np.ndarray,
            x: sparse.csr_matrix,
            fbr_list: 'list[float]',
            options: str
            ) -> 'tuple[np.matrix, np.ndarray]':
    """Inner cross-validation for SCutfbr heuristic.

    Args:
        y (np.ndarray): A +1/-1 array with dimensions number of instances * 1.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        fbr_list (list[float]): list of fbr values.
        options (str): The option string passed to liblinear.

    Returns:
        tuple[np.matrix, np.ndarray]: tuple of weights and threshold candidates.
    """

    b_list = np.zeros_like(fbr_list)

    nr_fold = 3

    l = y.shape[0]

    perm = np.random.permutation(l)

    for fold in range(nr_fold):
        mask = np.zeros_like(perm, dtype='?')
        mask[np.arange(int(fold*l/nr_fold), int((fold+1)*l/nr_fold))] = 1
        val_idx = perm[mask]
        train_idx = perm[mask != True]

        w = do_train(y[train_idx], x[train_idx], options)

        wTx = (x[val_idx] * w).A1
        scut_b = 0.
        start_F = fmeasure(y[val_idx], 2*(wTx > -scut_b) - 1)

        # stableness to match the MATLAB implementation
        sorted_wTx_index = np.argsort(wTx, kind='stable')
        sorted_wTx = wTx[sorted_wTx_index]

        tp = np.sum(y[val_idx] == 1)
        fp = val_idx.size - tp
        fn = 0
        cut = -1
        best_F = 2*tp / (2*tp + fp + fn)
        y_val = y[val_idx]

        # following MATLAB implementation to suppress NaNs
        prev_settings = np.seterr('ignore')
        for i in range(val_idx.size):
            if y_val[sorted_wTx_index[i]] == -1:
                fp -= 1
            else:
                tp -= 1
                fn += 1

            # There will be NaNs, but the behaviour is correct
            F = 2*tp / (2*tp + fp + fn)

            if F >= best_F:
                best_F = F
                cut = i
        np.seterr(**prev_settings)

        if best_F > start_F:
            if cut == -1:  # i.e. all 1 in scut
                scut_b = np.nextafter(-sorted_wTx[0], np.inf)  # predict all 1
            elif cut == val_idx.size - 1:
                scut_b = np.nextafter(-sorted_wTx[-1], np.inf)
            else:
                scut_b = -(sorted_wTx[cut] + sorted_wTx[cut + 1]) / 2

        F = fmeasure(y_val, 2*(wTx > -scut_b) - 1)

        for i in range(fbr_list.size):
            if F > fbr_list[i]:
                b_list[i] += scut_b
            else:
                b_list[i] -= np.max(wTx)

    b_list = b_list / nr_fold
    return do_train(y, x, options), b_list


def do_train(y: np.ndarray, x: sparse.csr_matrix, options: str) -> np.matrix:
    """Wrapper around liblinear.liblinearutil.train.
    Forcibly suppresses all IO regardless of options.

    Args:
        y (np.ndarray): A +1/-1 array with dimensions number of instances * 1.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.

    Returns:
        np.matrix: the weights.
    """
    if not '-q' in options:
        options += ' -q'
    with silent_stderr():
        model = train(y, x, options)

    w = np.ctypeslib.as_array(model.w, (x.shape[1], 1))
    w = np.asmatrix(w)
    # The memory is freed on model deletion so we make a copy.
    return w.copy()


class silent_stderr:
    """Context manager that suppresses stderr.
    Liblinear emits warnings on missing classes with
    specified weight, which may happen during cross-validation.
    Since this information is useless to the user, we suppress it.
    """

    def __init__(self):
        self.stderr = os.dup(2)
        self.devnull = os.open('/dev/null', os.O_WRONLY)

    def __enter__(self):
        os.dup2(self.devnull, 2)

    def __exit__(self, type, value, traceback):
        os.dup2(self.stderr, 2)
        os.close(self.devnull)
        os.close(self.stderr)


def fmeasure(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate F1 score.

    Args:
        y_true (np.ndarray): array of +1/-1.
        y_pred (np.ndarray): array of +1/-1.

    Returns:
        float: the F1 score.
    """
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    fn = np.sum(np.logical_and(y_true == 1, y_pred == -1))
    fp = np.sum(np.logical_and(y_true == -1, y_pred == 1))

    F = 0
    if tp != 0 or fp != 0 or fn != 0:
        F = 2*tp / (2*tp + fp + fn)
    return F


def train_cost_sensitive(y: sparse.csr_matrix, x: sparse.csr_matrix, options: str):
    """Trains a linear model for multilabel data using a one-vs-rest strategy
    and cross-validation to pick an optimal asymmetric misclassification cost
    for Macro-F1.
    Outperforms train_1vsrest in most aspects at the cost of higher
    time complexity.
    See user guide for more details.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.

    Returns:
        A model which can be used in predict_values.
    """
    # Follows the MATLAB implementation at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/multilabel/
    if any(o in options for o in ['-R', '-c', '-C', '-v']):
        raise ValueError('-R, -c, -C and -v are not supported')

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
    num_class = y.shape[1]
    num_feature = x.shape[1]
    weights = np.zeros((num_feature, num_class), order='F')
    for i in range(num_class):
        yi = y[:, i].toarray().reshape(-1)
        w = cost_sensitive_one_label(2*yi - 1, x, options)
        weights[:, i] = w.ravel()

    return {'weights': np.asmatrix(weights), '-B': bias, 'threshold': 0}


def cost_sensitive_one_label(y: np.ndarray,
                             x: sparse.csr_matrix,
                             options: str
                             ) -> np.ndarray:
    """Loop over parameter space for cost-sensitive on a single label.

    Args:
        y (np.ndarray): A +1/-1 array with dimensions number of instances * 1.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.

    Returns:
        np.ndarray: the weights.
    """

    l = y.shape[0]
    perm = np.random.permutation(l)

    param_space = [(a, c)
                   for a in [1, 1.33, 1.8, 2.5, 3.67, 6, 13]
                   for c in [1, 10, 100]]

    bestScore = -np.Inf
    for a, c in param_space:
        cv_options = f'{options} -c {c} -w1 {a}'
        score = cross_validate(y, x, cv_options, perm)
        if bestScore < score:
            bestScore = score
            bestA = a
            bestC = c

    final_options = f'{options} -c {bestC} -w1 {bestA}'
    return do_train(y, x, final_options)


def cross_validate(y: np.ndarray,
                   x: sparse.csr_matrix,
                   options: str,
                   perm: np.ndarray
                   ) -> float:
    """Cross-validation for cost-sensitive.

    Args:
        y (np.ndarray): A +1/-1 array with dimensions number of instances * 1.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.

    Returns:
        float: cross-validation F1 score.
    """
    l = y.shape[0]
    nr_fold = 3
    pred = np.zeros_like(y)
    for fold in range(nr_fold):
        mask = np.zeros_like(perm, dtype='?')
        mask[np.arange(int(fold*l/nr_fold), int((fold+1)*l/nr_fold))] = 1
        val_idx = perm[mask]
        train_idx = perm[mask != True]

        w = do_train(y[train_idx], x[train_idx], options)
        pred[val_idx] = (x[val_idx] * w).A1 > 0

    return fmeasure(y, 2*pred - 1)


def predict_values(model, x: sparse.csr_matrix) -> np.ndarray:
    """Calculates the decision values associated with x.

    Args:
        model: A model returned from a training function.
        x (sparse.csr_matrix): A matrix with dimension number of instances * number of features.

    Returns:
        np.ndarray: A matrix with dimension number of instances * number of classes.
    """
    bias = model['-B']
    bias_col = np.full((x.shape[0], 1 if bias > 0 else 0), bias)
    num_feature = model['weights'].shape[0]
    num_feature -= 1 if bias > 0 else 0
    if x.shape[1] < num_feature:
        x = sparse.hstack([
            x,
            np.zeros((x.shape[0], num_feature - x.shape[1])),
            bias_col,
        ], 'csr')
    else:
        x = sparse.hstack([
            x[:, :num_feature],
            bias_col,
        ], 'csr')

    return (x * model['weights']).A + model['threshold']
