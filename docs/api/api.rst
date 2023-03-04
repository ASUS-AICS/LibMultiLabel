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
We provide two installation types:

* Install both neural network module and linear classifier module. ::

    pip3 install libmultilabel

* Install only linear classifier module without any torch-related requirements. ::

    pip3 install libmultilabel[linear]
