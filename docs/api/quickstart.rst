Quickstart
^^^^^^^^^^^^^^^^^^^^

These tutorials show how the library API may be used in
end-to-end examples. After these tutorials, more in-depth
explanations can be found in the `user guides <https://www.csie.ntu.edu.tw/~cjlin/papers/libmultilabel/userguide.pdf>`__.

* `Linear Model for Multi-label Classification <../auto_examples/plot_linear_quickstart.html>`_
* `Bert Model for Multi-label  Classification <../auto_examples/plot_bert_quickstart.html>`_
* `KimCNN Model for Multi-label Classification <../auto_examples/plot_KimCNN_quickstart.html>`_

.. toctree::
    :caption: Library
    :maxdepth: 1
    :hidden:

    ../auto_examples/plot_linear_quickstart
    ../auto_examples/plot_bert_quickstart
    ../auto_examples/plot_KimCNN_quickstart

Before we start, please download and decompress the data ``rcv1`` via the following commands::

    mkdir -p data/rcv1
    wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_train.txt.bz2 -O data/rcv1/train.txt.bz2
    wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_test.txt.bz2 -O data/rcv1/test.txt.bz2
    bzip2 -d data/rcv1/*.bz2
