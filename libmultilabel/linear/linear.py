import numpy as np
import scipy.sparse as sparse

from liblinear.liblinearutil import train

__all__ = ['train_1vsrest', 'train_thresholding', 'predict_values']


def train_1vsrest(y: sparse.csr_matrix, x: sparse.csr_matrix, options: str):
    """Trains a linear model for multiabel data using a one-vs-all strategy.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.

    Returns:
        A model which can be used in predict_values.
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
    num_class = y.shape[1]
    num_feature = x.shape[1]
    weights = np.zeros((num_feature, num_class), order='F')
    for i in range(num_class):
        yi = y[:, i].toarray().reshape(-1)
        modeli = train(yi, x, options)
        w = np.ctypeslib.as_array(modeli.w, (num_feature,))
        # liblinear label mapping depends on data, we ensure
        # it is the same for all labels
        if modeli.get_labels()[0] == 0:
            w = -w
        weights[:, i] = w

    return {'weights': np.asmatrix(weights), '-B': bias, 'threshold': 0}


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


def train_thresholding(y: sparse.csr_matrix, x: sparse.csr_matrix, options: str):
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
    num_class = y.shape[1]
    num_feature = x.shape[1]
    weights = np.zeros((num_feature, num_class), order='F')
    thresholds = np.zeros(num_class)
    for i in range(num_class):
        yi = y[:, i].toarray().reshape(-1)
        w, b = thresholding_one_label(yi, x, options)
        weights[:, i] = w.ravel()
        thresholds[i] = b

    return {'weights': np.asmatrix(weights), '-B': bias, 'threshold': thresholds}


def thresholding_one_label(y: np.ndarray,
                           x: sparse.csr_matrix,
                           options: str
                           ) -> 'tuple[np.ndarray, float]':

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
        print(f'INFO: train_one_label: F all 0')

    # final model
    w, b_list = scutfbr(y, x, np.array([best_fbr]), options)
    print(f'INFO: train_one_label: best_fbr {best_fbr:.1f}')

    return w, b_list[0]


def scutfbr(y: np.ndarray,
            x: sparse.csr_matrix,
            fbr_list: 'list[float]',
            options: str
            ) -> 'tuple[np.ndarray, np.ndarray]':

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

        # stableness to match previous implementations
        sorted_wTx_index = np.argsort(wTx, kind='stable')
        sorted_wTx = wTx[sorted_wTx_index]

        tp = np.sum(y[val_idx] == 1)
        fp = val_idx.size - tp
        fn = 0
        cut = -1
        best_F = 2*tp / (2*tp + fp + fn)
        y_val = y[val_idx]

        prev_settings = np.seterr('ignore')  # supress NaNs
        for i in range(val_idx.size):
            if y_val[sorted_wTx_index[i]] == 0:
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
    # python doesn't flush by default, incompatible with I/O from C
    print('', end='', flush=True)
    model = train(y, x, options)
    w = np.ctypeslib.as_array(model.w, (x.shape[1], 1))
    w = np.asmatrix(w)
    # The order of labels is data dependent, we flip them to be consistent.
    # The memory is freed on model deletion so we make a copy.
    if model.get_labels()[0] == 0:
        w = -w
    else:
        w = w.copy()
    return w


def fmeasure(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    fn = np.sum(np.logical_and(y_true == 1, y_pred == -1))
    fp = np.sum(np.logical_and(y_true == -1, y_pred == 1))

    if tp != 0 or fp != 0 or fn != 0:
        return 2*tp / (2*tp + fp + fn)
    return 0
