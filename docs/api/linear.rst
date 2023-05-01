Linear Classifier API
=====================


Train and Predict
^^^^^^^^^^^^^^^^^
Linear methods are methods based on
`LibLinear <https://www.csie.ntu.edu.tw/~cjlin/liblinear/>`_.
The simplest usage is::

   model = linear.train_1vsrest(train_y, train_x, options)
   predict = linear.predict_values(model, test_x)


.. currentmodule:: libmultilabel.linear

.. autofunction:: train_1vsrest

.. autofunction:: train_thresholding

.. autofunction:: train_cost_sensitive

.. autofunction:: train_cost_sensitive_micro

.. autofunction:: train_binary_and_multiclass

.. autofunction:: train_tree

.. autofunction:: predict_values

.. autofunction:: get_topk_labels

.. autofunction:: get_positive_labels

Preprocessor
^^^^^^^^^^^^

.. autoclass:: Preprocessor
   :members:

   .. automethod:: __init__


.. autofunction:: read_libmultilabel_format

.. autofunction:: read_libsvm_format

Load and Save Pipeline
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: save_pipeline

.. autofunction:: load_pipeline


Metrics
^^^^^^^
Metrics are specified by their names in ``compute_metrics`` and ``get_metrics``.
The possible metric names are:

* ``'P@K'``, where ``K`` is a positive integer
* ``'RP@K'``, where ``K`` is a positive integer
* ``'NDCG@K'``, where ``K`` is a positive integer
* ``'Macro-F1'``
* ``'Micro-F1'``

.. Their definitions are given in the `user guide <https://www.csie.ntu.edu.tw/~cjlin/papers/libmultilabel/userguide.pdf>`_.

.. autofunction:: compute_metrics

.. autofunction:: get_metrics

.. autoclass:: MetricCollection
   :members:

.. autofunction:: tabulate_metrics

Grid Search with Sklearn Estimators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: MultiLabelEstimator
   :members:

   .. automethod:: __init__

   .. automethod:: fit

   .. automethod:: predict

   .. automethod:: score

.. autoclass:: GridSearchCV
   :members:

   .. automethod:: __init__
