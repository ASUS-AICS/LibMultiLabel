"""
Feature Generation and Parameter Selection for Linear Methods
=============================================================

This tutorial demonstrates feature generation and parameter selection for linear methods.
Our workflow is similiar to the one in sklearn.

Here we show an example of training a linear text classifier with the rcv1 dataset.
If you haven't downloaded it yet, see `Data Preparation  <../cli/linear.html#step-1-data-preparation>`_.

Feature Generation and Setting Up a Linear Method
-------------------------------------------------

To use a linear method, we must convert ecah text to a vector of numerical features.
If you decide to use the default setting (TF-IDF features) and specify the LIBLINEAR options
used by linear methods by yourself, check `our quickstart <../api/api.html#quickstart>`_
for easily conducting training and testing.

To choose among different feature generations and LIBLINEAR options, we conduct a customized estimator
for making LibMultiLabel methods in a sklearn Pipeline for a grid search.
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from libmultilabel.linear import MultiLabelEstimator

estimator = MultiLabelEstimator(
    options='-s 2 -m 1', linear_technique='1vsrest')
pipeline = Pipeline([
    ('vect', CountVectorizer(max_features=20000, min_df=3)),
    ('tfidf', TfidfTransformer()),
    ('clf', estimator)
])

######################################################################
# For the estimator ``MultiLabelEstimator``, arguments ``options`` is a LIBLINEAR option, and
# ``linear_technique`` is one of linear techniques: ``1vsrest``, ``thresholding``, ``cost_sensitive``,
# ``cost_sensitive_micro``, and ``binary_and_mulitclass``.
# In ``pipeline``, we specify ``tfidf`` as the way to convert texts to vectors.

######################################################################
# Grid Search over Feature Generations and LIBLINEAR Options
# -----------------------------------------------------------
# From the above construction of an estimator, because options are set, we can directly train a model with ``pipeline.fit``.
# We format labels into a 0/1 sparse matrix with ``MultiLabelBinarizer``.

from sklearn.preprocessing import MultiLabelBinarizer
from libmultilabel.linear import MultiLabelEstimator, read_libmultilabel_format

training_file = 'data/rcv1/train.txt'
train_data = read_libmultilabel_format(training_file)

binarizer = MultiLabelBinarizer(sparse_output=True)
binarizer.fit(train_data['label'])
train_labels = binarizer.transform(train_data['label']).astype('d')
pipeline.fit(train_data['text'], train_labels)

######################################################################
# However, to search for the best setting, we can employ ``GridSearchCV``.

from libmultilabel.linear.sklearn_helper import GridSearchCV

liblinear_options = ['-s 2 -c 0.5', '-s 2 -c 1', '-s 2 -c 2']
parameters = {
    'clf__options': liblinear_options,
    'vect__max_features': [10000, 20000, 40000],
    'vect__min_df': [3, 5]
}
clf = GridSearchCV(pipeline, parameters, cv=5, n_jobs=4, verbose=1)
clf = clf.fit(train_data['text'], train_labels)

######################################################################
# Here we check the combinations of two feature generations and three regularization parameters
# in the linear classifier. The key in ``parameters`` should follow the sklearn's coding rule
# starting with the estimator's alias and two underscores (i.e., ``clf__``).
# We specify ``n_jobs=4`` to run four tasks in parallel.
