Training and Prediction for Linear Classifiers
==============================================

For a step-by-step tutorial, see

    - :ref:`cli-quickstart`

For the documentation on some commonly used command line flags,
see

    - :ref:`linear_train`
    - :ref:`linear_predict`

For the complete set of command line flags see

    - `Command Line Options <flags.html>`_

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

Download and uncompress the RCV1 dataset from
`LIBSVM Data <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html>`_ with

.. code-block:: bash

    wget -O train.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_train.txt.bz2
    wget -O test.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_test.txt.bz2
    bzip2 -d *.bz2

Browse an instance of the data with

.. code-block:: bash

    head -n 1 train.txt
    2286    E11 ECAT M11 M12 MCAT   recov recov recov recov excit excit bring mexic mexic [...]

In this example, the dataset used is in :ref:`libmultilabel-format`, which is a textual
data format. See `Dataset Formats <ov_data_format.html#dataset-formats>`_
for more details on accepted data formats.

Step 2. Training and Prediction via an Example
----------------------------------------------

Next, move back to the root directory and run the main script

.. code-block:: bash

    cd ../..
    python3 main.py --config example_config/rcv1/l2svm.yml

This trains a L2-regularized L2-loss (primal) SVM and predict the test set.
The config file holds all the options used, of which
three commonly used options for linear classifiers are ``--linear``,
``--liblinear_options``, and ``--linear_technique``.
These options may be overriden on the command line

.. code-block:: bash

    python3 main.py --config example_config/rcv1/l2svm.yml --linear --liblinear_options="-s 2 -B 1 -e 0.0001 -q" --linear_technique 1vsrest

----------------------------------------------

.. _linear_train:

Training and (Optional) Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To train a model, use

.. code-block:: bash

    python3 main.py --config CONFIG_PATH \
                    --linear \
                    --liblinear_options=LIBLINEAR_OPTIONS \
                    --linear_technique MULTILABEL_TECHNIQUE \
                    --data_format DATA_FORMAT

- **config**: configure parameters in a yaml file.
  A validation set is not needed because the program may split the training set for
  internal validation. If specified, it will be ignored.

The linear classifiers are based on
`LIBLINEAR <https://www.csie.ntu.edu.tw/~cjlin/liblinear/>`_,
and its options may be specified.

- **linear**: If this option exists, it is set to True such that the linear
  classifiers will be run. Otherwise it is set to False by default such that
  the neural network module will be executed and the program will terminate if
  the neural network config is not given.

- **liblinear_options**: An
  `option string for LIBLINEAR <https://github.com/cjlin1/liblinear>`_.
  For example

    .. code-block:: bash

        --liblinear_options="-s 2 -B 1 -c 1"

- **linear_technique**: An option for multi-label techniques.
  It should be one of:
  ``1vsrest`` (one-vs-rest),
  ``thresholding`` (thresholding),
  and ``cost_sensitive`` (cost-sensitive).

- **data_format**: The data format, it should be one of
  ``txt`` (LibMultiLabel format),
  ``svm`` (LibSVM format).
  See `Dataset Formats <ov_data_format.html#dataset-formats>`_
  for more details on accepted data formats.

.. _linear_predict:

Prediction
^^^^^^^^^^

To predict a test set on a previously trained model, use

.. code-block:: bash

    python3 main.py --eval \
                    --config CONFIG_PATH \
                    --linear \
                    --data_format DATA_FORMAT \
                    --checkpoint_path CHECKPOINT_PATH

where ``CHECKPOINT_PATH`` is a path to a ``linear_pipeline.pickle``.
