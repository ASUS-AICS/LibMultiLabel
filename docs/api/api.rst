Getting Started
===============

The library API is used in your own scripts if you want
more fine grained control over the training/prediction process.
For an out-of-the-box usage of an application, see the
`command line interface <../cli/linear.html>`_ instead.

The library API is composed of a `linear classifier module <linear.html>`_ and a `neural network classifier module <nn.html>`_::

    import libmultilabel.nn
    import libmultilabel.linear

Both of which can be used independently.

Installation
^^^^^^^^^^^^

We provide two installation types to install LibMultiLabel:

* Install both neural network module and linear classifier module. ::

    pip3 install libmultilabel

* Install only linear classifier module without any torch-related requirements. ::

    pip3 install libmultilabel[linear]

Quickstart
^^^^^^^^^^^^^^^^^^^^

These tutorials show how the library API may be used in
end-to-end examples. After these tutorials, more in-depth
explanations can be found in the `user guides <placeholder>`_.

* `Linear Model for Multi-label Classification <../auto_examples/plot_linear_tutorial.html>`_
* `Bert Model for Multi-label  Classification <../auto_examples/plot_bert_tutorial.html>`_
* `KimCNN Model for Multi-label Classification <../auto_examples/plot_KimCNN_tutorial.html>`_

.. toctree::
    :caption: Library
    :maxdepth: 1
    :hidden:

    ../auto_examples/plot_linear_tutorial
    ../auto_examples/plot_bert_tutorial
    ../auto_examples/plot_KimCNN_tutorial

Before we start, please download and decompress the data ``rcv1`` via the following commands::

    mkdir -p data/rcv1
    wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_train.txt.bz2 -O data/rcv1/train.txt.bz2
    wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_test.txt.bz2 -O data/rcv1/test.txt.bz2
    bzip2 -d data/rcv1/*.bz2




