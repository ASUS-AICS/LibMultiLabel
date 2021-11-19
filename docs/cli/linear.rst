Training and Prediction for Linear Classifiers
==============================================

For users who are just getting started, see:

    - :ref:`cli-quickstart`

If you have been familiar with the basic operations, see:

    - :ref:`linear_train`
    - :ref:`linear_predict`

-------------------------------------------------------------------

.. _cli-quickstart:

Using CLI via an Example
^^^^^^^^^^^^^^^^^^^^^^^^

Step 1. Data Preparation
------------------------

Create a data sub-directory within LibMultiLabel and go to this sub-directory.

.. code-block:: bash

    mkdir -p data/rcv1
    cd data/rcv1

Download the RCV1 :ref:`libsvm-format` dataset from `LIBSVM Data <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html>`_ by the following commands.

.. code-block:: bash

    wget -O train.svm.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_train.svm.bz2
    wget -O test.svm.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_combined_test.svm.bz2

Uncompress data files and change the directory back to LibMultiLabel.

.. code-block:: bash

    bzip2 -d *.bz2
    cd ../..

See `Dataset Formats <ov_data_format.html#dataset-formats>`_ here if you want to use your own dataset.

Step 2. Training and Prediction via an Example
----------------------------------------------

Train a L2-regularized L2-loss (primal) SVM and predict the test set by an example config.

.. code-block:: bash

    python3 main.py --config example_config/rcv1/l2svm.yml

The most commonly used three options for linear classifier are ``--linear``, ``--liblinear_options``, and ``--linear_technique``.

.. code-block:: bash

    python3 main.py --config example_config/rcv1/l2svm.yml --linear --liblinear_options "-s 2 -B 1 -e 0.0001 -q" --linear_technique 1vsrest

----------------------------------------------

.. _linear_train:

Training and (Optional) Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For training, use

.. code-block:: bash

    python3 main.py --config CONFIG_PATH \
                    --linear \
                    --liblinear_options LIBLINEAR_OPTIONS \
                    --linear_technique MULTILABEL_TECHNIQUE

- **config**: configure parameters in a yaml file.  A validation set is not needed because the program may split the training set for internal validation. If specified, it will be ignored.

The linear classifiers are based on `LIBLINEAR <https://www.csie.ntu.edu.tw/~cjlin/liblinear/>`_, and its options may be specified.

- **linear**: If this option exists, it is set to True such that the linear classifiers will be run. Otherwise it is set to False by default such that the neural network module will be executed and the program will terminate if the neural network config is not given.

- **liblinear_options**: An `option string for LIBLINEAR <https://github.com/cjlin1/liblinear>`_.

- **linear_technique**: An option for multi-label techniques. We now support ``1vsrest`` (implementing one-vs-rest technique), ``thresholding`` (implementing thresholding technique), and ``cost_sensitive`` (implementing cost-sensitive technique).

.. _linear_predict:

Prediction
^^^^^^^^^^

To deploy/evaluate a model, you can predict a test set by the following command.

.. code-block:: bash

    python3 main.py --eval --config CONFIG_PATH --linear --checkpoint_path CHECKPOINT_PATH
