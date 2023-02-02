"""
Linear Grid Search
==================

This tutorial demonstrates how to train a linear text classifier and grid search over LIBLINEAR options using a customized sklearn estimator.



Train a linear text classifier
------------------------------

Here we show an example of training a linear text classifier with the rcv1 dataset.
If you haven't downloaded it yet, see
`Getting the Dataset <../api/linear_tutorial.html#getting-the-dataset>`_.

A typical workflow for training a linear text classifier includes binarizing labels,
transforming text to vectors with TF-IDF vectorizer, and classifying text with linear techniques.
In the classifying step, a customized sklearn estimator, ``MultiLabelEstimator``, helps users to run LibMultiLabel classifiers in a sklearn Pipeline.
The sample code shows you how it works.
"""

from libmultilabel.linear.preprocessor import read_libmultilabel_format
from libmultilabel.linear.sklearn_helper import MultiLabelEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline


training_file = 'data/rcv1/train.txt'
train_data = read_libmultilabel_format(training_file)

binarizer = MultiLabelBinarizer(sparse_output=True)
binarizer.fit(train_data['label'])
train_labels = binarizer.transform(train_data['label']).astype('d')

estimator = MultiLabelEstimator(
    options='-s 2 -m 1', linear_technique='1vsrest')
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', estimator)
])
pipeline.fit(train_data['text'], train_labels)

######################################################################
# For the estimator ``MultiLabelEstimator``, arguments ``options`` is a LIBLINEAR option, and ``linear_technique`` is one of linear techniques:
# ``1vsrest``, ``thresholding``, ``cost_sensitive``, ``cost_sensitive_micro``, and ``binary_and_mulitclass``.

######################################################################
# Grid Search over LIBLINEAR options
# ----------------------------------
# To run grid search with multiple LIBLINEAR options, directly pass in the defined pipeline and parameters to search to GridSearchCV as below.

from libmultilabel.linear.sklearn_helper import GridSearchCV


liblinear_options = ['-s 2 -c 0.5', '-s 2 -c 1', '-s 2 -c 2']
parameters = {'clf__options': liblinear_options}
clf = GridSearchCV(pipeline, parameters, cv=5, n_jobs=4, verbose=1)
clf = clf.fit(train_data['text'], train_labels)

######################################################################
# Class ``libmultilabel.linear.sklearn_helper.GridSearchCV`` append ``-m 1`` to LIBLINEAR options while ``n_jobs>1``
# to avoid oversubscribing CPU. The key in ``parameters`` should follow the sklearn's coding rule starting
# with the estimator's alias and two underscores __(i.e., clf__).
