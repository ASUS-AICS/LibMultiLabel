LibMultiLabel - a Library for Multi-label Text Classification
=============================================================

LibMultiLabel is a library for multi-label text classification
and a simple command line tool with the following functionalities

- end-to-end services from raw texts to final evaluation/analysis
- support for common neural network architectures and linear classsifiers
- easy hyper-parameter selection

The tool is used as::

   python3 main.py --config main_config.yml
   python3 search_params.py --config search_config.yml

The library is composed of a neural network module and a linear classifier module::

   import libmultilabel.nn
   import libmultilabel.linear


Quick Start via an Example
--------------------------

1. Data Preparation

   * Create a data sub-directory within ``LibMultiLabel`` and go to this sub-directory::

      mkdir -p data/rcv1
      cd data/rcv1

   * Download the ``rcv1`` dataset from `LIBSVM data sets <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets>`_ by the following commands::

      wget -O train.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_train.txt.bz2
      wget -O test.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_test.txt.bz2

   * Uncompress data files and change the directory back to ``LibMultiLabel``::

      bzip2 -d *.bz2
      cd ../..

2. Training and Prediction

   Train a cnn model and predict the test set by an example config. Use ``--cpu`` to run the program on the cpu::

      python3 main.py --config example_config/rcv1/kim_cnn.yml

   For more details about the usage see the `command line interface <api/cli.html>`_.

User Guide
----------

.. toctree::
   :maxdepth: 2

   guides/linear

API Documentation
-----------------

.. toctree::
   :maxdepth: 2

   api/linear
   api/nn
   api/cli

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
