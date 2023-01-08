import re

import scipy.sparse as sparse
from sklearn.base import BaseEstimator
from sklearn.model_selection._search import GridSearchCV
from sklearn.utils.validation import check_is_fitted, check_X_y

import libmultilabel.linear as linear
from libmultilabel.linear.utils import LINEAR_TECHNIQUES


class MultiLabelMixin:
    """Mixin class for multi-label classifiers.

    Args:
        metric_threshold (int, optional): The decision value threshold over which a label
            is predicted as positive. Defaults to 0.
        val_metric (str, optional): The evaluation metric for cross validation. Defaults to 'P@1'.
    """
    def __init__(self, metric_threshold=0, val_metric='P@1'):
        self.metrics = None
        self.metric_threshold = metric_threshold
        self.val_metric = val_metric

    def score(self, X: sparse.csr_matrix, y: sparse.csr_matrix):
        """Sklearn estimator score method."""
        if self.metrics is None:
            self.metrics = linear.get_metrics(
                self.metric_threshold,
                [self.val_metric],
                y.shape[1],
            )
        preds = self.predict(X)
        target = y.toarray()
        self.metrics.update(preds, target)
        metric_dict = self.metrics.compute()
        return metric_dict[self.val_metric]

    def _more_tags(self):
        return {'requires_y': True}


class MultiLabelClassifier(MultiLabelMixin, BaseEstimator):
    """Customized sklearn estimator for the multi-label classifier.

    Args:
        options (str, optional): The option string passed to liblinear. Defaults to '-s 2'.
        metric_threshold (int, optional): The decision value threshold over which a label
            is predicted as positive. Defaults to 0.
        val_metric (str, optional): The evaluation metric for cross validation. Defaults to 'P@1'.
        linear_technique (str, optional): Multi-label technique defined in `utils.LINEAR_TECHNIQUES`.
            Defaults to '1vsrest'.
    """
    def __init__(self, options='-s 2', metric_threshold=0, val_metric='P@1', linear_technique='1vsrest'):
        super().__init__(metric_threshold, val_metric)
        self.options = options
        self.linear_technique = linear_technique

    def fit(self, X: sparse.csr_matrix, y: sparse.csr_matrix):
        X, y = check_X_y(X, y, accept_sparse=True, multi_output=True)
        self.is_fitted_ = True
        self.model = LINEAR_TECHNIQUES[self.linear_technique](
            y, X, self.options)
        return self

    def predict(self, X: sparse.csr_matrix):
        # Sklearn required code for checking if fit has been called
        check_is_fitted(self)
        preds = linear.predict_values(self.model, X)
        return preds


class MultiLabelGridSearchCV(GridSearchCV):
    _required_parameters = ['estimator', 'param_grid']

    def __init__(self, estimator, param_grid, n_jobs=None, **kwargs):
        key = None
        for name, transform in estimator.steps:
            if isinstance(transform, MultiLabelClassifier):
                key = f'{name}__options'
        assert key is not None

        # Set to single-core liblinear while running with `n_jobs` > 1
        if n_jobs is not None and n_jobs > 1:
            regex = r'-m \d+'
            param_grid[key] = [f"{re.sub(regex, '', v)} -m 1" for v in param_grid[key]]

        super().__init__(
            estimator=estimator,
            n_jobs=n_jobs,
            param_grid=param_grid,
            **kwargs
        )
        self.param_grid = param_grid
