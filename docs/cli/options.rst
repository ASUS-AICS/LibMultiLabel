Specifying Options
==================

The command line interface uses a `yaml file <https://github.com/ASUS-AICS/LibMultiLabel/tree/master/example_config>`_ to configure the dataset and the training process.
At a high level, the config file is split into four parts:

- :ref:`data_options`
- :ref:`model_options`
- :ref:`train_options`
- :ref:`evaluation_options`
- :ref:`hyperparameter_search_options`

------------

.. _data_options:

Data Options
^^^^^^^^^^^^
The data options consists of data paths that tell where to place the datasets, pre-trained word embeddings, and vocabularies.

+----------------+-----------------------------------------------------------------------------------------------+
| Option         | Description                                                                                   |
+================+===============================================================================================+
| data_dir       | The directory to load data                                                                    |
+----------------+-----------------------------------------------------------------------------------------------+
| data_name      | Dataset name                                                                                  |
+----------------+-----------------------------------------------------------------------------------------------+
| train_path     | Path to training data                                                                         |
+----------------+-----------------------------------------------------------------------------------------------+
| val_path       | Path to validation data                                                                       |
+----------------+-----------------------------------------------------------------------------------------------+
| test_path      | Path to test data                                                                             |
+----------------+-----------------------------------------------------------------------------------------------+
| val_size       | Training-validation split: a ratio in [0, 1] or an integer for the size of the validation set |
+----------------+-----------------------------------------------------------------------------------------------+
| min_vocab_freq | The minimum frequency needed to include a token in the vocabulary                             |
+----------------+-----------------------------------------------------------------------------------------------+
| max_seq_length | The maximum number of tokens of a sample                                                      |
+----------------+-----------------------------------------------------------------------------------------------+
| shuffle        | Whether to shuffle training data before each epoch                                            |
+----------------+-----------------------------------------------------------------------------------------------+
| vocab_file     | Path to a file holding vocabuaries                                                            |
+----------------+-----------------------------------------------------------------------------------------------+
| embed_file     | Path to a file holding pre-trained embeddings                                                 |
+----------------+-----------------------------------------------------------------------------------------------+
| label_file     | Path to a file holding all labels                                                             |
+----------------+-----------------------------------------------------------------------------------------------+

.. _model_options:

Model Options
^^^^^^^^^^^^^
The model options defines the parameters that are related to the network definition (i.e., model name).

+--------------------+--------------------------------------------------+
| Option             | Description                                      |
+====================+==================================================+
| model_name         | Model to be used: BiGRU, CAML, KimCNN, or XMLCNN |
+--------------------+--------------------------------------------------+
| init_weight        | Weight initialization to be used                 |
+--------------------+--------------------------------------------------+
| **network_config** | Configuration for defining the network.          |
+--------------------+--------------------------------------------------+

network_config
--------------
Parameters for initializing a network are defined under a nested configuration ``network_config``.

+---------------------+--------------------------------------------------------------------------------------------------+
| Option              | Description                                                                                      |
+=====================+==================================================================================================+
| activation          | Activation function to be used                                                                   |
+---------------------+--------------------------------------------------------------------------------------------------+
| num_filter_per_size | Number of filters in convolutional layers in each size                                           |
+---------------------+--------------------------------------------------------------------------------------------------+
| filter_sizes        | Size of convolutional filters                                                                    |
+---------------------+--------------------------------------------------------------------------------------------------+
| dropout             | Optional specification of dropout                                                                |
+---------------------+--------------------------------------------------------------------------------------------------+
| dropout2            | Optional specification of the second dropout                                                     |
+---------------------+--------------------------------------------------------------------------------------------------+
| num_pool            | Number of pool for dynamic max-pooling                                                           |
+---------------------+--------------------------------------------------------------------------------------------------+
| hidden_dim          | Dimension of the hidden layer                                                                    |
+---------------------+--------------------------------------------------------------------------------------------------+
| rnn_dim             | The size of bidirectional hidden layers. The hidden size of the GRU network is set to rnn_dim//2 |
+---------------------+--------------------------------------------------------------------------------------------------+
| rnn_layers          | Number of recurrent layers                                                                       |
+---------------------+--------------------------------------------------------------------------------------------------+

.. _train_options:

Train Options
^^^^^^^^^^^^^
The train options specifies the hyperparameters (i.e., batch size, learning rate, etc.) used when training a model.

+---------------+----------------------------------------------------------------+
| Option        | Description                                                    |
+===============+================================================================+
| seed          | Random seed                                                    |
+---------------+----------------------------------------------------------------+
| epochs        | Number of epochs to train                                      |
+---------------+----------------------------------------------------------------+
| batch_size    | Size of training batches                                       |
+---------------+----------------------------------------------------------------+
| optimizer     | Optimizer: SGD or Adam                                         |
+---------------+----------------------------------------------------------------+
| learning_rate | Learning rate for optimizer                                    |
+---------------+----------------------------------------------------------------+
| weight_decay  | Weight decay factor                                            |
+---------------+----------------------------------------------------------------+
| momentum      | Momentum factor for SGD only                                   |
+---------------+----------------------------------------------------------------+
| patience      | Number of epochs to wait for improvement before early stopping |
+---------------+----------------------------------------------------------------+

.. _evaluation_options:

Evaluation Options
^^^^^^^^^^^^^^^^^^
The evalation options decides metrics monitored or reported during the evaluation process.

+------------------+------------------------------------------+
| Option           | Description                              |
+==================+==========================================+
| eval_batch_size  | Size of evaluating batches               |
+------------------+------------------------------------------+
| metric_threshold | Thresholds to monitor for metrics        |
+------------------+------------------------------------------+
| monitor_metrics  | Metrics to monitor while validating      |
+------------------+------------------------------------------+
| val_metric       | The metric to monitor for early stopping |
+------------------+------------------------------------------+


.. _hyperparameter_search_options:

Hyperparameter Search Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The hyperparamter search options decide how to perform hyperparameter selection.
It is worth noting that for ``grid_search``, the ``num_samples`` indicates the repeat time of a combination.
Namely, set ``num_samples`` to one means run one time grid search.
For other search algorithms, ``num_samples`` means the number of running trials.

+-----------------+---------------------------------------------------------------------------------------------------------------------+
| Option          | Description                                                                                                         |
+=================+=====================================================================================================================+
| search_alg      | Search algorithms: basic_variant, bayesopt, optuna                                                                  |
+-----------------+---------------------------------------------------------------------------------------------------------------------+
| embed_cache_dir | Path to a directory for storing embeddings for multiple runs.                                                       |
+-----------------+---------------------------------------------------------------------------------------------------------------------+
| num_samples     | Number of running trials. If the search space is `grid_search`, the same grid will be repeated `num_samples` times. |
+-----------------+---------------------------------------------------------------------------------------------------------------------+
