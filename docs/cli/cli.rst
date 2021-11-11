Using CLI
=========

To work with the command line interface, see the following sections if you're just getting started:

    - :ref:`installation`
    - :ref:`cli-quickstart`

If you want to know more about training, prediction, and hyperparameter search, see:

    - `Training and Prediction for Linear Classifiers <linear.html>`_
    - `Training, Prediction, and Hyperparameter Search for Neural Networks <nn.html>`_

----------

.. _installation:

Install LibMultiLabel from Source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Clone `LibMultiLabel <https://github.com/ASUS-AICS/LibMultiLabel>`_.
* Install the latest development version, run:

.. code-block:: bash

    pip3 install -r requirements.txt

.. _cli-quickstart:

Using CLI via an Example
^^^^^^^^^^^^^^^^^^^^^^^^

Step 1. Data Preparation
------------------------

Create a data sub-directory within LibMultiLabel and go to this sub-directory.

.. code-block:: bash

    mkdir -p data/rcv1
    cd data/rcv1

Download the RCV1 dataset from `LIBSVM datasets <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html>`_ by the following commands.

.. code-block:: bash

    wget -O train.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_train.txt.bz2
    wget -O test.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_test.txt.bz2

Uncompress data files and change the directory back to LibMultiLabel.

.. code-block:: bash

    bzip2 -d *.bz2
    cd ../..

See `Dataset Formats <data_format.html>`_ here if you want to use your own dataset.

Step 2. Training and Prediction
-------------------------------

Train a CNN model and predict the test set by an example config. Use ``--cpu`` to run the program on the cpu.

.. code-block:: bash

    python3 main.py --config example_config/rcv1/kim_cnn.yml
