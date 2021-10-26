Neural Network CLI Tutorial
===========================

Using CLI via an Example
------------------------

Step 1. Data Preparation
^^^^^^^^^^^^^^^^^^^^^^^^

Create a data sub-directory within LibMultiLabel and go to this sub-directory.

.. code-block:: bash

    mkdir -p data/rcv1
    cd data/rcv1

Download the rcv1 dataset from `LIBSVM data sets <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets)>`_ by the following commands.

.. code-block:: bash

    wget -O train.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_train.txt.bz2
    wget -O test.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_test.txt.bz2

Uncompress data files and change the directory back to LibMultiLabel.

.. code-block:: bash

    bzip2 -d *.bz2
    cd ../..


Step 2. Training and Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train a cnn model and predict the test set by an example config. Use ``--cpu`` to run the program on the cpu.

.. code-block:: bash

    python3 main.py --config example_config/rcv1/kim_cnn.yml