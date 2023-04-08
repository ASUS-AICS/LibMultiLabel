"""
Handling Data with Many Labels
==============================

Training a multi-label classification problem with numerous labels can be time-consuming. 
We show how a solver in LibMultiLabel can reduce the training time on such datasets.

Consider the EUR-Lex dataset, which contains 3,956 labels 
and can be downloaded from `LIBSVM Data Sets <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html>`_. 
We show the use of the ``train_tree`` method and compare it with the standard ``train_1vsrest`` method.
To run the following code, you must put data to the designated directory.
"""

import libmultilabel.linear as linear
import time

preprocessor = linear.Preprocessor(data_format='txt')
datasets = preprocessor.load_data('data/eur-lex/train.txt', 'data/eur-lex/test.txt')

training_start = time.time()
# the standard one-vs-rest way to train multi-label problems
model_OVR = linear.train_1vsrest(datasets['train']['y'], datasets['train']['x'])
training_end = time.time()
print('Training time of one-versus-rest: {:10.2f}'.format(training_end-training_start))

training_start = time.time()
# the method for fast training of data with many labels
model_tree = linear.train_tree(datasets['train']['y'], datasets['train']['x'])
training_end = time.time()
print('Training time of tree-based: {:10.2f}'.format(training_end-training_start))

######################################################################
# On a machine with an AMD-7950X CPU, 
# the ``train_1vsrest`` function required `578.30` seconds, 
# while the ``train_tree`` function only required `144.37` seconds. 
# 
# .. note::
#
#   The ``train_tree`` function in this tutorial is based on the work of :cite:t:`SK20a`.
#
# Due to the nature of some approximations, it is unclear if ``train_tree`` trades performance for time.
# We use the following code to check the test performance.

preds_OVR = linear.predict_values(model_OVR, datasets['test']['x'])
preds_tree = linear.predict_values(model_tree, datasets['test']['x'])

metrics = linear.get_metrics(metric_threshold=0,
                             monitor_metrics=['P@1', 'P@3', 'P@5'],
                             num_classes=datasets['test']['y'].shape[1])

target = datasets['test']['y'].toarray()

metrics.update(preds_OVR, target)
print("Evaluation of OVR:", metrics.compute())

metrics.reset()

metrics.update(preds_tree, target)
print("Evaluation of tree:", metrics.compute())

######################################################################
#  :math:`P@K`, a ranking-based criterion, is a suitable metric for data with many labels.
#
#.. code-block::
#   
#   Evaluation of OVR: {'P@1': 0.833117723156533, 'P@3': 0.6988357050452781, 'P@5': 0.585666235446313}
#   Evaluation of tree: {'P@1': 0.8217335058214748, 'P@3': 0.692539887882708, 'P@5': 0.578835705045278}
#
#For this set, ``train_tree`` gives only slightly lower :math:`P@K`, but is much faster.
#
#.. bibliography::
