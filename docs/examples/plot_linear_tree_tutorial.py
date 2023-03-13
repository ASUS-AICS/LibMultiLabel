"""
Tree-Based Linear Model for Extreme Multi-Label Classification.
===============================================================

In this tutorial, we show how to use our tree-based linear solver for a multi-label classification problem, 
and lightly explain why a tree-based setting can reduce the training time, especially the number of labels :math:`m` is extremely large. 

Traditionally, we can treat each label as an independent binary classification problem in multi-label classification problem, 
i.e., we have to solve :math:`m` binary classification problems, and this naive method is called `one-versus-rest`.
`Our quickstart of linear model <plot_linear_quickstart.html>`_ is an example for using one-versus-rest linear solver,
and the training complexity on a data set 

.. math ::

  (X, Y) \in (\mathbb{R}^{l \\times n}, \mathbb{R}^{l \\times m})

is roughly :math:`O( m \\times \\textrm{nnz of } X  )`. 
Therefore, when the number of labels :math:`m` is a extremely large number,
the training time grows up to a painful long time.  
Let us show this issue by training a data set `EUR-Lex` with 

.. math::

  l = 15449, n=186104, m=3956 \\textrm{ and nnz of } X = 4160670,

where the code is similar to `our quickstart of linear model <plot_linear_quickstart.html>`_ instead of the data set.
"""

import libmultilabel.linear as linear
import time

preprocessor = linear.Preprocessor(data_format='txt')
datasets = preprocessor.load_data('data/eur-lex/train.txt',
                                  'data/eur-lex/test.txt')

training_start = time.time()
model_OVR = linear.train_1vsrest(datasets['train']['y'],
                                 datasets['train']['x'],
                                 '-s 2')
training_end = time.time()
print("Training time of one-versus-rest: {:10.2f}".format(training_end-training_start))

######################################################################
# In our machine with CPU AMD-7950X, a one-versus-rest method requires :math:`578.30` seconds when the solver applies multi-core version LIBLINEAR.
# 
# To conquer the explosion of training time, we can divid the labels to :math:`k` clusters, 
# where :math:`k` is much smaller than :math:`m` (e.g., taking :math:`k=100`), and train a subproblem with :math:`k` metalabels via one-versus-rest method. 
# That is, the data set on this subproblem becomes 
# 
# .. math::
#
#  (X, \tilde{Y}) \in (\mathbb{R}^{l \times n}, \mathbb{R}^{l \times k})
#  
# and an given instance :math:`\boldsymbol{x}` has :math:`j\text{th}` metalabel means that 
# this data :math:`\boldsymbol{x}` has some labels in the :math:`j\text{th}` cluster.
# If the number of labels in a cluster is still large, it can then be divided again.    
# Therefore, this divid-and-conquer step can construct a tree for label, and we call `tree-based linear method` here. 
# 
# Actually, we have :math:`k` binary classification problems in each node of tree, so the total number of binary problems is approximate to :math:`2m`.
# However, except for the root node, the instances are much less than :math:`l` (idealy :math:`l/k` ), 
# which implies the total training complexity becomes 
# 
# .. math::
# 
#  O( k \sum_{i=0}^{d} \frac{ \text{nnz of } X }{ k^{i} } ) \approx O( \frac{k^{2}}{k-1} \times \text{nnz of } X )
#
# if the label tree is a balanced :math:`d-\text{level}` tree and the nnz of :math:`X` is uniformly distributed.
#
# In our implementation, we mainly consider the method `Bonsai` :cite:t:`SK20a`, and you can use the solver by the following code.

training_start = time.time()
model_tree = linear.train_tree(datasets['train']['y'],
                               datasets['train']['x'],
                               '-s 2')
training_end = time.time()
print("Training time of tree-based: {:10.2f}".format(training_end-training_start))


######################################################################
# The training time is reduced to :math:`144.37` seconds. 
#
# For the performance, a tree-based method usually worse than one-versus-rest method. 
# In this example, the evaluation can be calculated by the following code.

preds_OVR = linear.predict_values(model_OVR, datasets['test']['x'])
preds_tree = linear.predict_values(model_tree, datasets['test']['x'])

metrics = linear.get_metrics(metric_threshold=0,
                             monitor_metrics=['Macro-F1', 'Micro-F1', 'P@1', 'P@3', 'P@5'],
                             num_classes=datasets['test']['y'].shape[1])

target = datasets['test']['y'].toarray()

metrics.update(preds_OVR, target)
print("Evaluation of OVR:", metrics.compute())
metrics.update(preds_tree, target)
print("Evaluation of tree:", metrics.compute())

######################################################################
# The results will look similar to
#
#
#.. bibliography::
