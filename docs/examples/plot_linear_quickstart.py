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
# the simplest way is to take the positive values as the labels predicted
# to be associated with the sample, i.e. ``preds > 0``.

label_mask = preds > 0

######################################################################
# We now have the label mask. Next,
# we use ``label_mapping`` in ``Preprocessor`` to get the original labels.

label_mapping = preprocessor.label_mapping
prediction = [label_mapping[row].tolist() for row in label_mask]

######################################################################
# The result of first instance looks like:
#
#   >>> print(prediction[0])
#   ...
#       ['GCAT', 'GSPO']
#
# To see how well we performed, we may want to check various
# metrics with the test set.
# For that we may use:

metrics = linear.get_metrics(metric_threshold=0,
                             monitor_metrics=['Macro-F1', 'Micro-F1', 'P@1', 'P@3', 'P@5'],
                             num_classes=datasets['test']['y'].shape[1])

######################################################################
# This creates the set of metrics we wish to see.
# Since the dataset we loaded are stored as ``scipy.sparse.csr_matrix``,
# we need to transform them to ``np.array`` before we can compute the metrics:

target = datasets['test']['y'].toarray()

######################################################################
# Finally, we compute and print the metrics:

metrics.update(preds, target)
print(metrics.compute())

######################################################################
# The results will look similar to::
#
#     {'Macro-F1': 0.5171960144875225, 'Micro-F1': 0.8008124243391698, 'P@1': 0.9573153795447128, 'P@3': 0.799074151109632, 'P@5': 0.5579924865442584}
