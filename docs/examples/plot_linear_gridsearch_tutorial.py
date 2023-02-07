"""
Feature Generation and Parameter Selection for Linear Methods
=============================================================

This tutorial demonstrates feature generation and parameter selection for linear methods.

Here we show an example of training a linear text classifier with the rcv1 dataset.
If you haven't downloaded it yet, see `Data Preparation  <../cli/linear.html#step-1-data-preparation>`_.
Otherwise, you can read and preprocess the data as follow.
"""

from sklearn.preprocessing import MultiLabelBinarizer
import libmultilabel.linear as linear
from libmultilabel.linear.preprocessor import read_libmultilabel_format # code?

train_data = read_libmultilabel_format('data/rcv1/train.txt')
binarizer = MultiLabelBinarizer(sparse_output=True)
binarizer.fit(train_data['label'])
y = binarizer.transform(train_data['label']).astype('d')

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
# If you want to generate TF-IDF features with your own settings, consider

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=20000, min_df=3)
vectorizer.fit(train_data['text'])
x = vectorizer.transform(train_data['text'])
model = linear.train_1vsrest(y, x, '-s 2 -m 4')

#######################################################################
# We can then use the generated numerical features ``x`` as the input of
# the linear method ``linear.train_1vsrest``.
#
# An Alternative Way for Using a Linear Method
# --------------------------------------------
# Besides the default way shown in `Feature Generation <#feature-generation>`_,
# we can construct a sklearn estimator for training (do we need to demo prediction?).
# This way is used namely for parameter selection described later,
# as the estimator makes LibMultiLabel methods in a sklearn Pipeline for a grid search.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, min_df=3)),
    ('clf', linear.MultiLabelEstimator(
        options='-s 2 -m 4', linear_technique='1vsrest'))
])

######################################################################
# For the estimator ``MultiLabelEstimator``, arguments ``options`` is a LIBLINEAR option, and
# ``linear_technique`` is one of linear techniques: ``1vsrest``, ``thresholding``, ``cost_sensitive``,
# ``cost_sensitive_micro``, and ``binary_and_mulitclass``.
# In ``pipeline``, we specify settings used by the estimator.
# For example, ``tfidf`` indicates the vectorizer for generating TF-IDF features,
# and ``clf`` is the estimator.
#
# We can then use the following code for training.
pipeline.fit(train_data['text'], y)

######################################################################
# Grid Search over Feature Generations and LIBLINEAR Options
# -----------------------------------------------------------
# To search for the best setting, we can employ ``GridSearchCV``.
liblinear_options = ['-s 2 -c 0.5', '-s 2 -c 1', '-s 2 -c 2']
parameters = {
    'clf__options': liblinear_options,
    'tfidf__max_features': [10000, 20000, 40000],
    'tfidf__min_df': [3, 5]
}
clf = linear.GridSearchCV(pipeline, parameters, cv=5, n_jobs=4, verbose=1)
clf = clf.fit(train_data['text'], y)

######################################################################
# Here we check the combinations of two feature generations and three regularization parameters
# in the linear classifier. The key in ``parameters`` should follow the sklearn's coding rule
# starting with the estimator's alias and two underscores (i.e., ``clf__``).
# We specify ``n_jobs=4`` to run four tasks in parallel.
