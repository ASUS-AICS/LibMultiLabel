import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from libmultilabel.linear import predict_values, train_binary_and_multiclass


class MulticlassClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, options):
        self.options = options

    def fit(self, X, y):
        # We need to validate the data because we do a safe_indexing later.
        # X, y = self._validate_data(
        #     X, y, accept_sparse=["csr", "csc"], force_all_finite=False
        # )
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Return the classifier
        self.model = train_binary_and_multiclass(y, X, self.options)
        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        preds = predict_values(self.model, X)
        return np.argmax(preds, axis=1)
