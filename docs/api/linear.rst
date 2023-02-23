Linear Classifier API
=====================


Train and Predict
^^^^^^^^^^^^^^^^^
Linear methods are methods based on
`LibLinear <https://www.csie.ntu.edu.tw/~cjlin/liblinear/>`_.
The simplest usage is::

   model = linear.train_1vsrest(train_y, train_x, options)
   predict = linear.predict_values(model, test_x)

.. See `the user guide <../guides/linear_guides.html>`_ for more details.

.. currentmodule:: libmultilabel.linear

.. autofunction:: train_1vsrest

.. autofunction:: train_thresholding

.. autofunction:: train_cost_sensitive

.. autofunction:: train_cost_sensitive_micro

.. autofunction:: train_binary_and_multiclass

.. autofunction:: predict_values


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
