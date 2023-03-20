Installation
===========================

The library API is used in your own scripts if you want
more fine grained control over the training/prediction process.
For an out-of-the-box usage of an application, see the
`command line interface <../cli/linear.html>`_ instead.

The library API is composed of a `linear classifier module <linear.html>`_ and a `neural network classifier module <nn.html>`_::

    import libmultilabel.nn
    import libmultilabel.linear

The two types of APIs can be used independently.
The modules can be installed with:

* Install both neural network module and linear classifier module ::

    pip3 install libmultilabel[all]

* Install neural network module only ::

    pip3 install libmultilabel[nn]

* Install linear classifier module only ::

    pip3 install libmultilabel[linear]

* Additionally install parameter search functionality for neural networks ::

    pip3 install libmultilabel[nn-param-search]