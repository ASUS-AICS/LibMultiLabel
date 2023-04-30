"""
Linear Model for Multi-label Classification.
============================================

This guide will take you through how LibMultiLabel can
be used to train a linear classifier in python scripts.

For this guide, we will only need the linear module:
"""

import libmultilabel.linear as linear

######################################################################
# To start, we need to read and preprocess the input data:

preprocessor = linear.Preprocessor(data_format='txt')

datasets = preprocessor.load_data('data/rcv1/train.txt',
                                  'data/rcv1/test.txt')

######################################################################
# The preprocessor handles many issues such as: mapping
# the labels into indices and transforming textual data to
# numerical data. The loaded dataset has the structure::
#
#     {
#         'train': {
#             'x': # training features
#             'y': # training labels
#         },
#         'test': {
#             'x': # test features
#             'y': # test labels
#         },
#     }
#
# Next we train the model:

model = linear.train_1vsrest(datasets['train']['y'],
                             datasets['train']['x'],
                             '')

######################################################################
# The third argument is the options string for
# `LibLinear <https://www.csie.ntu.edu.tw/~cjlin/liblinear/>`__.
# We may leave it as the default for now.
#
# Once we have the model, we may predict with it:

preds = linear.predict_values(model, datasets['test']['x'])

######################################################################
# ``preds`` holds the decision values, i.e. the raw values
# outputted by the model. To transform it into predictions,
# you can apply the API ``get_positive_labels`` to get predicted labels and their corresponding scores
# by using ``label_mapping`` in ``preprocessor`` and ``preds`` from the last step.

pred_labels, pred_scores = linear.get_positive_labels(preds, preprocessor.label_mapping)

######################################################################
# We now have the labels (``pred_labels``) and scores (``pred_scores``).
# You can use the following code to save the prediction to a list.

prediction = []
for label, score in zip(pred_labels, pred_scores):
    prediction.append(
        [f"{i}:{s:.4}" for i, s in zip(label, score)])

######################################################################
# The first instance looks like:
#
#   >>> print(prediction[0])
#   ...
#       ['GCAT:1.345', 'GSPO:1.519']
#
# To see how well we performed, we may want to check various
# metrics with the test set.
# Since the dataset we loaded are stored as ``scipy.sparse.csr_matrix``,
# we will first transform the dataset to ``np.array``.

target = datasets['test']['y'].toarray()

##############################################################################
# Then we will compute the metrics with ``compute_metrics``.

metrics = linear.compute_metrics(
    preds,
    target,
    monitor_metrics=['Macro-F1', 'Micro-F1', 'P@1', 'P@3', 'P@5'],
)

print(metrics)

######################################################################
# The results will look similar to::
#
#     {'Macro-F1': 0.5171960144875225, 'Micro-F1': 0.8008124243391698, 'P@1': 0.9573153795447128, 'P@3': 0.799074151109632, 'P@5': 0.5579924865442584}
