"""
Tweaking Feature Generation for Linear Methods
=============================================================

In both `API  <../auto_examples/plot_linear_quickstart.html>`_ and `CLI  <../cli/linear.html>`_ usage of linear methods, LibMultiLabel handles the feature generation step by default.
Unless necessary, you do not need to generate features in different ways as described in this tutorial.

This tutorial demonstrates how to customize the way to generate features for linear methods through an API example.
Here we use the `rcv1 <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html#rcv1v2%20(topics;%20full%20sets)>`_ dataset as an example.
"""

from sklearn.preprocessing import MultiLabelBinarizer
from libmultilabel import linear

datasets = linear.load_dataset("txt", "data/rcv1/train.txt", "data/rcv1/test.txt")
tfidf_params = {
    "max_features": 20000,
    "min_df": 3,
    "ngram_range": (1, 3)
}
preprocessor = linear.Preprocessor(tfidf_params=tfidf_params)
preprocessor.fit(datasets)
datasets = preprocessor.transform(datasets)

############################################
# The argument ``tfidf_params`` of the ``Preprocessor`` can specify how to generate the TF-IDF features.
# In this example, we adjust the ``max_features``, ``min_df``, and ``ngram_range`` of the preprocessor.
# For explanation of these three and other options, refer to the `sklearn page <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>`_.
# Users can also try other methods to generalize features, like word embedding.
#
# Finally, we use the generated numerical features to train and evaluate the model.
# The rest of the steps is the same in the quickstarts.
# Please refer to them for details.