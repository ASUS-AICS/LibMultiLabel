"""
Feature Generation and Parameter Selection for Linear Methods
=============================================================

This tutorial demonstrates feature generation and parameter selection for linear methods.

Here we show an example of training a linear text classifier with the rcv1 dataset.
If you haven't downloaded it yet, see `Data Preparation  <../cli/linear.html#step-1-data-preparation>`_.
Then you can read and preprocess the data as follows
"""

from sklearn.preprocessing import MultiLabelBinarizer
from libmultilabel import linear

datasets = linear.load_dataset("txt", "data/rcv1/train.txt", "data/rcv1/test.txt")
binarizer = MultiLabelBinarizer(sparse_output=True)
y = binarizer.fit_transform(datasets["train"]["y"]).astype("d")

######################################################################
# We format labels into a 0/1 sparse matrix with ``MultiLabelBinarizer``.
#
# Feature Generation
# ------------------
# Before training a linear classifier, we must convert each text to a vector of numerical features.
# To use the default setting (TF-IDF features), check
# `Linear Model for MultiLabel Classification <../auto_examples/plot_linear_quickstart.html#linear-model-for-multi-label-classification>`_
# for easily conducting training and testing.
#
# If you want to tweak the generation of TF-IDF features, consider

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=20000, min_df=3)
x = vectorizer.fit_transform(datasets["train"]["x"])
model = linear.train_1vsrest(y, x, "-s 2 -m 4")

#######################################################################
# We use the generated numerical features ``x`` as the input of
# the linear method ``linear.train_1vsrest``.
#
# An Alternative Way for Using a Linear Method
# --------------------------------------------
# Besides the default way shown in `Feature Generation <#feature-generation>`_,
# we can construct a sklearn estimator for training and prediction.
# This way is used namely for parameter selection described later,
# as the estimator makes LibMultiLabel methods in a sklearn Pipeline for a grid search.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(max_features=20000, min_df=3)),
        ("clf", linear.MultiLabelEstimator(options="-s 2 -m 4", linear_technique="1vsrest", scoring_metric="P@1")),
    ]
)

######################################################################
# For the estimator ``MultiLabelEstimator``, arguments ``options`` is a LIBLINEAR option
# (see *train Usage* in `liblinear <https://github.com/cjlin1/liblinear>`__ README), and
# ``linear_technique`` is one of linear techniques: ``1vsrest``, ``thresholding``, ``cost_sensitive``,
# ``cost_sensitive_micro``, and ``binary_and_mulitclass``.
# In ``pipeline``, we specify settings used by the estimator.
# For example, ``tfidf`` is the alias of ``TfidfVectorizer`` and ``clf`` is the alias of the estimator.
#
# We can then use the following code for training.
pipeline.fit(datasets["train"]["x"], y)

######################################################################
# Grid Search over Feature Generations and LIBLINEAR Options
# -----------------------------------------------------------
# To search for the best setting, we can employ ``GridSearchCV``.
# The usage is similar to sklearn's except that the parameter ``scoring`` is not available.  Please specify
# ``scoring_metric`` in ``linear.MultiLabelEstimator`` instead.
liblinear_options = ["-s 2 -c 0.5", "-s 2 -c 1", "-s 2 -c 2"]
parameters = {"clf__options": liblinear_options, "tfidf__max_features": [10000, 20000, 40000], "tfidf__min_df": [3, 5]}
clf = linear.GridSearchCV(pipeline, parameters, cv=5, n_jobs=4, verbose=1)
clf = clf.fit(datasets["train"]["x"], y)

######################################################################
# Here we check the combinations of six feature generations and three regularization parameters
# in the linear classifier. The key in ``parameters`` should follow the sklearn's coding rule
# starting with the estimator's alias and two underscores (i.e., ``clf__``).
# We specify ``n_jobs=4`` to run four tasks in parallel.
# After finishing gridsearch, we can get the best parameters by the following code:

for param_name in sorted(parameters.keys()):
    print(f"{param_name}: {clf.best_params_[param_name]}")

######################################################################
# The best parameters are::
#
#   clf__options: '-s 2 -c 0.5 -m 1'
#   tfidf__max_features: 20000
#   tfidf__min_df: 3
#
# For testing, we also need to read in data first and format test labels into a 0/1 sparse matrix.

y = binarizer.transform(datasets["test"]["y"]).astype("d").toarray()

######################################################################
# Applying the ``predict`` function of ``GridSearchCV`` object to use the
# estimator trained under the best hyper-parameters for prediction.
# Then use ``linear.compute_metrics`` to calculate the test performance.

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
#   {'Macro-F1': 0.4965720851051106, 'Micro-F1': 0.8004678830627301, 'P@1': 0.9587412721675744, 'P@3': 0.8021469454453142, 'P@5': 0.5605401496291271}
