Specifying Options
==================

The LibMultiLabel toolkit uses a yaml file to configure the dataset and the training process. At a high level, the config file is split into four parts:

- **data**: the data config consists of data paths that tell where to place the datasets, pre-trained word embeddings, and vocabularies.

- **model**: the model config defines the parameters that are related to the network definition (i.e., model name).

- **train**: the train config specifies the hyperparameters (i.e., batch size, learning rate, etc.) used when training a model.

- **eval**: the eval config decides metrics monitored or reported during the evaluation process.

The configuration may also be overridden by passing additional arguments, for example:

.. code-block:: bash

    python3 main.py --config config.yml --cpu --eval

------------

Data Options
^^^^^^^^^^^^

+----------------+----------------------+-----------------------------------------------------------------------------------------------+
| parameters     | default              | description                                                                                   |
+================+======================+===============================================================================================+
| data_dir       | ./data/rcv1          | The directory to load data                                                                    |
+----------------+----------------------+-----------------------------------------------------------------------------------------------+
| data_name      | rcv1                 | Dataset name                                                                                  |
+----------------+----------------------+-----------------------------------------------------------------------------------------------+
| train_path     | [data_dir]/train.txt | Path to training data                                                                         |
+----------------+----------------------+-----------------------------------------------------------------------------------------------+
| val_path       | [data_dir]/valid.txt | Path to validation data                                                                       |
+----------------+----------------------+-----------------------------------------------------------------------------------------------+
| test_path      | [data_dir]/test.txt  | Path to test data                                                                             |
+----------------+----------------------+-----------------------------------------------------------------------------------------------+
| val_size       | 0.2                  | Training-validation split: a ratio in [0, 1] or an integer for the size of the validation set |
+----------------+----------------------+-----------------------------------------------------------------------------------------------+
| min_vocab_freq | 1                    | The minimum frequency needed to include a token in the vocabulary                             |
+----------------+----------------------+-----------------------------------------------------------------------------------------------+
| max_seq_length | 500                  | The maximum number of tokens of a sample                                                      |
+----------------+----------------------+-----------------------------------------------------------------------------------------------+
| shuffle        | 2500                 | Whether to shuffle training data before each epoch                                            |
+----------------+----------------------+-----------------------------------------------------------------------------------------------+
| vocab_file     |                      | Path to a file holding vocabuaries                                                            |
+----------------+----------------------+-----------------------------------------------------------------------------------------------+
| embed_file     |                      | Path to a file holding pre-trained embeddings                                                 |
+----------------+----------------------+-----------------------------------------------------------------------------------------------+
| label_file     |                      | Path to a file holding all labels                                                             |
+----------------+----------------------+-----------------------------------------------------------------------------------------------+

..
    .. argparse::
        :filename: ../main.py
        :func: get_parser
        :nodefault:

Model Options
^^^^^^^^^^^^^

.. add network config after the bug is fixed

+------------------+-----------------+---------------------+
| parameters       | default         | description         |
+==================+=================+=====================+
| model_name       | KimCNN          |                     |
+------------------+-----------------+---------------------+
| init_weight      | kaiming_uniform |                     |
+------------------+-----------------+---------------------+


Train Options
^^^^^^^^^^^^^

+---------------+---------+----------------------------------------------------------------+
| parameters    | default | description                                                    |
+===============+=========+================================================================+
| seed          |         | Random seed                                                    |
+---------------+---------+----------------------------------------------------------------+
| epochs        | 10000   | Number of epochs to train                                      |
+---------------+---------+----------------------------------------------------------------+
| batch_size    | 16      | Size of training batches                                       |
+---------------+---------+----------------------------------------------------------------+
| optimizer     | adam    | Optimizer: SGD or Adam                                         |
+---------------+---------+----------------------------------------------------------------+
| learning_rate | 0.0001  | Learning rate for optimizer                                    |
+---------------+---------+----------------------------------------------------------------+
| weight_decay  | 0       | Weight decay factor                                            |
+---------------+---------+----------------------------------------------------------------+
| momentum      | 0.9     | Momentum factor for SGD only                                   |
+---------------+---------+----------------------------------------------------------------+
| patience      | 5       | Number of epochs to wait for improvement before early stopping |
+---------------+---------+----------------------------------------------------------------+
| shuffle       | 2500    |                                                                |
+---------------+---------+----------------------------------------------------------------+

Evaluation Options
^^^^^^^^^^^^^^^^^^

+------------------+-----------------------+------------------------------------------+
| parameters       | default               | description                              |
+==================+=======================+==========================================+
| eval_batch_size  | 256                   | Size of evaluating batches               |
+------------------+-----------------------+------------------------------------------+
| metric_threshold | 0.5                   | Thresholds to monitor for metrics        |
+------------------+-----------------------+------------------------------------------+
| monitor_metrics  | ['P@1', 'P@3', 'P@5'] | Metrics to monitor while validating      |
+------------------+-----------------------+------------------------------------------+
| val_metric       | P@1                   | The metric to monitor for early stopping |
+------------------+-----------------------+------------------------------------------+

