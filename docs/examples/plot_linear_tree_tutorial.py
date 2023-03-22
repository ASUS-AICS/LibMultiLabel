"""
Accelerated Training in Linear Model: Extreme Label Problems
============================================================

Training a multi-label classification problem on a large-scale dataset with numerous labels can be time-consuming. 
In this tutorial, we will introduce an accelerated training algorithm for multi-label classification that can reduce training time on such datasets.

To demonstrate the effectiveness of our approach, we will use the EUR-Lex dataset, 
which contains a large number of labels (3,956 labels) 
and can be downloaded from `LIBSVM Data Sets <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html>`_. 
We will provide code that is similar to `our quickstart of linear model <plot_linear_quickstart.html>`_, but with the added benefit of faster training time.
"""

import libmultilabel.linear as linear
import time

preprocessor = linear.Preprocessor(data_format='txt')
datasets = preprocessor.load_data('data/eur-lex/train.txt',
                                  'data/eur-lex/test.txt')

training_start = time.time()
# Original API from quickstart of linear model
model_OVR = linear.train_1vsrest(datasets['train']['y'],
                                 datasets['train']['x'],
                                 '-s 2')
training_end = time.time()
print('Training time of one-versus-rest: {:10.2f}'.format(training_end-training_start))

training_start = time.time()
# New API for accelerated training a linear model
model_tree = linear.train_tree(datasets['train']['y'],
                               datasets['train']['x'],
                               '-s 2')
training_end = time.time()
print('Training time of tree-based: {:10.2f}'.format(training_end-training_start))

######################################################################
# On our machine, equipped with an AMD-7950X CPU, 
# we observed that the ``train_1vsrest`` function required `578.30` seconds to complete the training on the EUR-Lex dataset, 
# while the ``train_tree`` function only required `144.37` seconds. 
# This demonstrates the significant reduction in training time that can be achieved by using the ``train_tree`` algorithm.
# 
# .. note::
#
#   The ``train_tree`` function in this tutorial is based on the work of :cite:t:`SK20a`, 
#   who introduced label approximation techniques to improve the training speed of their model, known as Bonsai. 
#   Although the details of these techniques are beyond the scope of this tutorial,
#   we have incorporated them into our implementation to provide a faster training algorithm for multi-label classification problems.
#
# While approximation methods may lead to slightly worse performance compared to the original algorithm, 
# we found that the difference in performance between ``train_tree`` and ``train_1vsrest`` was minimal in our experiments. 
# To evaluate the performance of our models on the EUR-Lex dataset, we provide the following code for calculating evaluation metrics.

preds_OVR = linear.predict_values(model_OVR, datasets['test']['x'])
preds_tree = linear.predict_values(model_tree, datasets['test']['x'])

metrics_OVR = linear.get_metrics(metric_threshold=0,
                                 monitor_metrics=['P@1', 'P@3', 'P@5'],
                                 num_classes=datasets['test']['y'].shape[1])
metrics_tree = linear.get_metrics(metric_threshold=0,
                                  monitor_metrics=['P@1', 'P@3', 'P@5'],
                                  num_classes=datasets['test']['y'].shape[1])

target = datasets['test']['y'].toarray()

metrics_OVR.update(preds_OVR, target)
print("Evaluation of OVR:", metrics_OVR.compute())
metrics_tree.update(preds_tree, target)
print("Evaluation of tree:", metrics_tree.compute())

######################################################################
# Using the evaluation code provided, we can calculate the following metrics for our trained models:
#
#.. code-block::
#
#   Evaluation of OVR: {'P@1': 0.833117723156533, 'P@3': 0.6989219491159984, 'P@5': 0.585666235446313}
#   Evaluation of tree: {'P@1': 0.825614489003881, 'P@3': 0.6947822337214317, 'P@5': 0.5803622250970245}
#
#As we can see, while there is a slight difference in performance between the two methods, it is not significant. 
#Therefore, the ``train_tree`` algorithm can be considered a viable option 
#for training linear models on large-scale multi-label classification problems with a reduced training time.
#
#.. bibliography::
