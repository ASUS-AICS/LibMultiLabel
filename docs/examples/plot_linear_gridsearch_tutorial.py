"""
Hyperparameter Search for Linear Methods
=============================================================
This guide helps users to tune the hyperparameters of the feature generation step and the linear model.

Here we show an example of tuning a linear text classifier with the `rcv1 dataset <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html#rcv1v2%20(topics;%20full%20sets)>`_.
Starting with loading and preprocessing of the data without using ``Preprocessor``:
"""

from sklearn.preprocessing import MultiLabelBinarizer
from libmultilabel import linear

datasets = linear.load_dataset("txt", "data/rcv1/train.txt", "data/rcv1/test.txt")
binarizer = MultiLabelBinarizer(sparse_output=True)
y = binarizer.fit_transform(datasets["train"]["y"]).astype("d")

######################################################################
# we format labels into a 0/1 sparse matrix with ``MultiLabelBinarizer``.
#
# Next, we construct a ``Pipeline`` object that will be used for hyperparameter search later.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(max_features=20000, min_df=3)),
        ("clf", linear.MultiLabelEstimator(options="-s 2 -m 4", linear_technique="1vsrest", scoring_metric="P@1")),
    ]
)

######################################################################
# The vectorizor ``TfidfVectorizer`` is used in ``Pipeline`` to generate TF-IDF features from raw texts.
# As for the estimator ``MultiLabelEstimator``, argument ``options`` is a LIBLINEAR option
# (see *train Usage* in `liblinear <https://github.com/cjlin1/liblinear>`__ README), and
# ``linear_technique`` is one of the linear techniques, including ``1vsrest``, ``thresholding``, ``cost_sensitive``,
# ``cost_sensitive_micro``, and ``binary_and_mulitclass``.
#
# We can specify the aliases of the components used by the pipeline.
# For example, ``tfidf`` is the alias of ``TfidfVectorizer`` and ``clf`` is the alias of the estimator.
#
# To search for the best setting, we employ ``GridSearchCV``.
# The usage is similar to sklearn's except that the parameter ``scoring`` is not available.  Please specify
# ``scoring_metric`` in ``linear.MultiLabelEstimator`` instead.

liblinear_options = ["-s 2 -c 0.5", "-s 2 -c 1", "-s 2 -c 2", "-s 1 -c 0.5", "-s 1 -c 1", "-s 1 -c 2"]
parameters = {"clf__options": liblinear_options, "tfidf__max_features": [10000, 20000, 40000], "tfidf__min_df": [3, 5]}
clf = linear.GridSearchCV(pipeline, parameters, cv=5, n_jobs=4, verbose=1)
clf = clf.fit(datasets["train"]["x"], y)

######################################################################
# Here we check the combinations of six feature generation options and six liblinear options
# in the linear classifier. The key in ``parameters`` should follow the sklearn's coding rule
# starting with the estimator's alias and two underscores (i.e., ``clf__``).
# We specify ``n_jobs=4`` to run four tasks in parallel.
# After finishing the grid search, we can get the best parameters by the following code:

for param_name in sorted(parameters.keys()):
    print(f"{param_name}: {clf.best_params_[param_name]}")

######################################################################
# The best parameters are::
#
#   clf__options: -s 2 -c 0.5 -m 1                      
#   tfidf__max_features: 10000                          
#   tfidf__min_df: 5
#
# Note that in the above code, the ``refit`` argument of ``GridSearchCV`` is enabled by default, meaning that the best configuration will be trained on the whole dataset after hyperparameter search.
# We refer to this as the retrain strategy.
# After fitting ``GridSearchCV``, the retrained model is stored in ``clf``. 
#
# We can apply the ``predict`` function of ``GridSearchCV`` object to use the estimator trained under the best hyperparameters for prediction.
# Then use ``linear.compute_metrics`` to calculate the test performance.

# For testing, we also need to read in data first and format test labels into a 0/1 sparse matrix.
y = binarizer.transform(datasets["test"]["y"]).astype("d").toarray()
preds = clf.predict(datasets["test"]["x"])
metrics = linear.compute_metrics(
    preds,
    y,
    monitor_metrics=["Macro-F1", "Micro-F1", "P@1", "P@3", "P@5"],
)
print(metrics)

######################################################################
# The result of the best parameters will look similar to::
#
#   {'Macro-F1': 0.5296621774388927, 'Micro-F1': 0.8021279986938116, 'P@1': 0.9561621216872636, 'P@3': 0.7983185389507189, 'P@5': 0.5570921518306848}
