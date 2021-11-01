Training and Prediction for Linear Classifiers
==============================================

Training and (Optional) Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For training, use

.. code-block:: bash

    python3 main.py --config CONFIG_PATH \
                    --linear [--liblinear_options LIBLINEAR_OPTIONS]

- **config**: configure parameters in a yaml file.

The linear classifiers are based on LIBLINEAR, and its options may be specified.

- **liblinear_options**: An `option string for LIBLINEAR <https://github.com/cjlin1/liblinear>`_.

Prediction
^^^^^^^^^^

To deploy/evaluate a model, you can predict a test set by the following command.

.. code-block:: bash

    python3 main.py --eval --config CONFIG_PATH --linear --load_checkpoint CHECKPOINT_PATH
