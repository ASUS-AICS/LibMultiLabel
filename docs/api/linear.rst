Linear Methods
===============

Linear methods are methods based on
`LibLinear <https://www.csie.ntu.edu.tw/~cjlin/liblinear/>`_.
The simplest usage is::

   model = linear.train_1vsrest(train_y, train_x, options)
   predict = linear.predict_values(model, test_x)

See `the user guide <../guides/linear.rst>`_ for more details.

.. currentmodule:: libmultilabel.linear

.. autofunction:: train_1vsrest

.. autofunction:: predict_values

.. autoclass:: Preprocessor
   :members:

.. autofunction:: save_pipeline

.. autofunction:: load_pipeline