import re

import scipy.sparse as sparse
import sklearn.base
import sklearn.model_selection
import sklearn.pipeline
import sklearn.utils

import libmultilabel.linear as linear
from libmultilabel.linear.utils import LINEAR_TECHNIQUES


class MultiLabelEstimator(sklearn.base.BaseEstimator):
    """Customized sklearn estimator for the multi-label classifier.

    Args:
        options (str, optional): The option string passed to liblinear. Defaults to '-s 2'.
        linear_technique (str, optional): Multi-label technique defined in `utils.LINEAR_TECHNIQUES`.
            Defaults to '1vsrest'.
        metric_threshold (int, optional): The decision value threshold over which a label
            is predicted as positive. Defaults to 0.
        scoring_metric (str, optional): The scoring metric. Defaults to 'P@1'.
    """

    def __init__(self, options='-s 2', linear_technique='1vsrest', metric_threshold=0, scoring_metric='P@1'):
        super().__init__()
        self.options = options
        self.linear_technique = linear_technique
        self.metric_threshold = metric_threshold
        self.scoring_metric = scoring_metric
        self._is_fitted = False

    def fit(self, X: sparse.csr_matrix, y: sparse.csr_matrix):
        X, y = sklearn.utils.validation.check_X_y(
            X, y, accept_sparse=True, multi_output=True)
        self._is_fitted = True
        self.model = LINEAR_TECHNIQUES[self.linear_technique](
            y, X, self.options)
        return self

    def predict(self, X: sparse.csr_matrix):
        sklearn.utils.validation.check_is_fitted(self, attributes=['_is_fitted'])
        preds = linear.predict_values(self.model, X)
        return preds

    def score(self, X: sparse.csr_matrix, y: sparse.csr_matrix):
        metrics = linear.get_metrics(
            self.metric_threshold,
            [self.scoring_metric],
            y.shape[1],
        )
        preds = self.predict(X)
        metrics.update(preds, y.toarray())
        metric_dict = metrics.compute()
        return metric_dict[self.scoring_metric]


class GridSearchCV(sklearn.model_selection.GridSearchCV):
    _required_parameters = ['pipeline', 'param_grid']

    def __init__(self, pipeline: sklearn.pipeline.Pipeline, param_grid: dict, n_jobs=None, **kwargs):
        assert isinstance(pipeline, sklearn.pipeline.Pipeline)
        if n_jobs is not None and n_jobs > 1:
            param_grid = self._set_singlecore_options(pipeline, param_grid)

        super().__init__(
            estimator=pipeline,
            n_jobs=n_jobs,
            param_grid=param_grid,
            **kwargs
        )

    def _set_singlecore_options(self, pipeline: sklearn.pipeline.Pipeline, param_grid: dict):
        """Set liblinear options to `-m 1`. The grid search option `n_jobs`
        runs multiple processes in parallel. Using multithreaded liblinear
        in conjunction with grid search oversubscribes the CPU and deteriorates
        the performance significantly.
        """
        params = pipeline.get_params()
        for name, transform in params.items():
            if isinstance(transform, MultiLabelEstimator):
                regex = r'-m \d+'
                key = f'{name}__options'
                param_grid[key] = [
                    f"{re.sub(regex, '', v)} -m 1" for v in param_grid[key]]
        return param_grid
