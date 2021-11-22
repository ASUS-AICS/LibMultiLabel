Linear Classifier API
=====================

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

.. autofunction:: predict_values

.. autoclass:: Preprocessor
   :members:

   .. automethod:: __init__

.. autofunction:: save_pipeline

.. autofunction:: load_pipeline