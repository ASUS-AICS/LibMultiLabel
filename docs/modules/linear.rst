Linear Methods
===============

Linear methods are methods based on
`LibLinear <https://www.csie.ntu.edu.tw/~cjlin/liblinear/>`_.
The typical usage is::

   model = linear.train_1vsrest(train_y, train_x, options)
   predict = linear.predict_values(model, test_x)

.. currentmodule:: libmultilabel.linear

.. autofunction:: train_1vsrest

.. autofunction:: predict_values

