Training and Prediction for Linear Classifiers
==============================================

For a step-by-step tutorial, see

    - :ref:`cli-quickstart`

For the documentation on some commonly used command line flags,
see

    - :ref:`linear_train`
    - :ref:`linear_predict`

For the complete set of command line flags, see

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

Linear methods take either textual or bag-of-words numeric data as inputs.
For this example, the data will be in :ref:`libmultilabel-format`, a textual data format.
Download and uncompress the RCV1 dataset with

.. code-block:: bash

    wget -O train.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_train.txt.bz2
    wget -O test.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_test.txt.bz2
    bzip2 -d *.bz2

Browse an instance of the data with

.. code-block:: bash

    head -n 1 train.txt
    # Output: 2286    E11 ECAT M11 M12 MCAT   recov recov recov recov excit excit bring mexic mexic [...]

If you want to use numeric data in :ref:`libsvm-format` instead, you may do so with

.. code-block::

    wget -O train.svm.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_train.svm.bz2
    wget -O test.svm.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_combined_test.svm.bz2
    bzip2 -d *.bz2
    head -n 1 train.svm
    # Output: 34,59,93,94,102  864:0.0497399253756197 1523:0.044664135988103 1681:0.0673871572152868 [...]

See `Dataset Formats <ov_data_format.html#dataset-formats>`_
for more details on the data formats.

Step 2. Training and Prediction via an Example
----------------------------------------------

Next, move back to the root directory and run the main script

.. code-block:: bash

    cd ../..
    python3 main.py --config example_config/rcv1/l2svm.yml

This trains a L2-regularized L2-loss SVM and evaluates the model on the test set.

----------------------------------------------

.. _linear_train:

Training and (Optional) Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To train and evaluate a model, use

.. code-block:: bash

    python3 main.py --config CONFIG_PATH \
                    --training_file TRAINING_DATA_PATH \
                    --test_file TEST_DATA_PATH \
                    --linear \
                    --liblinear_options=LIBLINEAR_OPTIONS \
                    --linear_technique MULTILABEL_TECHNIQUE \
                    --data_format DATA_FORMAT

- **config**: Path to a configuration file. Command line options
  may be specified here instead. See `Command Line Options <flags.html>`_ for more details.

The linear classifiers are based on
`LIBLINEAR <https://www.csie.ntu.edu.tw/~cjlin/liblinear/>`_,
and its options may be specified.

- **training_file**: The path to training data.

- **test_file**: The path to test data.
  If test data is available, also evaluates the trained model on the test data.

- **linear**: This option specifies that linear models should be ran,
  as opposed to running neural network models.

- **liblinear_options**: An
  `option string for LIBLINEAR <https://github.com/cjlin1/liblinear>`_.
  For example

    .. code-block:: bash

        --liblinear_options='-s 2 -B 1 -e 0.0001 -q'

- **linear_technique**: An option for multi-label techniques.
  It should be one of:
  ``1vsrest`` (one-vs-rest),
  ``thresholding`` (thresholding),
  and ``cost_sensitive`` (cost-sensitive).

- **data_format**: The data format. It should be one of
  ``txt`` (LibMultiLabel format),
  ``svm`` (LibSVM format).
  See `Dataset Formats <ov_data_format.html#dataset-formats>`_
  for more details on accepted data formats.

.. _linear_predict:

Prediction
^^^^^^^^^^

To predict a test set by applying a previously trained model, use

.. code-block:: bash

    python3 main.py --config CONFIG_PATH \
                    --test_file TEST_DATA_PATH \
                    --eval \
                    --linear \
                    --data_format DATA_FORMAT \
                    --checkpoint_path CHECKPOINT_PATH

where ``CHECKPOINT_PATH`` is a path to a ``linear_pipeline.pickle``.
