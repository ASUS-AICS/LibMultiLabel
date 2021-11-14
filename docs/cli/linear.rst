Training and Prediction for Linear Classifiers
==============================================

For users who are just getting started, see:

    - :ref:`installation`
    - :ref:`cli-quickstart`

If you have been familiar with the basic operations, see:

    - :ref:`linear_train`
    - :ref:`linear_predict`

-------------------------------------------------------------------

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

Train a L2-regularized L2-loss (primal) SVM and predict the test set by an example config. Use ``--cpu`` to run the program on the cpu.

.. code-block:: bash

    python3 main.py --config example_config/rcv1/l2svm.yml --linear --liblinear_options "-s 2 -B 1" --linear_technique 1vsrest

----------------------------------------------

.. _linear_train:

Training and (Optional) Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For training, use

.. code-block:: bash

    python3 main.py --config CONFIG_PATH \
                    --linear [--liblinear_options LIBLINEAR_OPTIONS] \
                             [--linear_technique MULTILABEL_TECHNIQUE]

- **config**: configure parameters in a yaml file. Notice that the validation set is not needed to be specified additionally as the program will ignore it. In addition, if users don't want to specify the test set, the empty string ``""`` needs to be specified in ``test_path``.

The linear classifiers are based on LIBLINEAR, and its options may be specified.

- **liblinear_options**: An `option string for LIBLINEAR <https://github.com/cjlin1/liblinear>`_.

- **linear_technique**: An option for multi-label techniques. We now support ``1vsrest`` (implementing one-vs-rest technique), ``thresholding`` (implementing thresholding technique), and ``cost_sensitive`` (implementing cost-sensitive technique). Notice that the cost-sensitive technique may take long (e.g., over an hour) for RCV1 dataset.

.. code-block:: bash

    python3 main.py --config example_config/rcv1/l2svm.yml --linear --linear_technique 1vsrest
    python3 main.py --config example_config/rcv1/l2svm.yml --linear --linear_technique thresholding

.. _linear_predict:

Prediction
^^^^^^^^^^

To deploy/evaluate a model, you can predict a test set by the following command.

.. code-block:: bash

    python3 main.py --eval --config CONFIG_PATH --linear --checkpoint_path CHECKPOINT_PATH
