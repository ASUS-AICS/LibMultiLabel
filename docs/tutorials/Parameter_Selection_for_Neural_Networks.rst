Parameter Selection for Neural Networks
==========================================

The performance of a model depends on the choice of hyper-parameters.
The following example demonstrates how the BiGRU model performs differently on the EUR-Lex data set with two parameter sets.
Datasets can be downloaded from the
`LIBSVM datasets <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html>`_.

Direct Trying Some Parameters
-----------------------------

First, train a BiGRU model with the
`default configuration file <https://github.com/ASUS-AICS/LibMultiLabel/blob/master/example_config/EUR-Lex/bigru_lwan.yml>`_
with a little modification on the learning rate.
Some important parameters are listed as follows.

.. code-block:: bash

    learning_rate: 0.001
    network_config:
      embed_dropout: 0.4
      encoder_dropout: 0.4
      rnn_dim: 512
      rnn_layers: 1

The training command is:

.. code-block:: bash

    python3 main.py --config example_config/EUR-Lex/bigru_lwan.yml

After training for 50 epochs, the checkpoint with the best validation performance is stored for testing. The
average P@1 score on the test data set is 81.40%.

Next, the ``learning_rate`` is changed to 0.003 while other parameters are kept the same.

.. code-block:: bash

    learning_rate: 0.003
    network_config:
      embed_dropout: 0.4
      encoder_dropout: 0.4
      rnn_dim: 512
      rnn_layers: 1

By the same training command, the P@1 score of the second parameter set is about 78.14%, which is
4% lower than the first one. This demonstrates the importance of parameter selection.

For more striking examples on the importance of parameter selection, you can see `this paper <https://www.csie.ntu.edu.tw/~cjlin/papers/parameter_selection/acl2021_parameter_selection.pdf>`_.

.. _Parameter Selection for Neural Networks:

Grid Search over Parameters
---------------------------

In the configuration file, we specify a grid search on the following parameters.


.. code-block:: bash

    learning_rate: ['grid_search', [0.003, 0.001, 0.0003]]
    network_config:
      embed_dropout: ['grid_search', [0, 0.2, 0.4, 0.6, 0.8]]
      encoder_dropout: ['grid_search', [0, 0.2, 0.4]]
      rnn_dim: ['grid_search', [256, 512, 1024]]
      rnn_layers: 1
    embed_cache_dir: .vector_cache

We set the ``embed_cache_dir`` to ``.vector_cache`` to avoid downloading pre-trained embeddings repeatedly for each configuration.


Then the training command is:

.. code-block:: bash

    python3 search_params.py --config example_config/EUR-Lex/bigru_lwan_tune.yml

The process finds the best parameter set of ``learning_rate=0.0003``, ``embed_dropout=0.4``, ``encoder_dropout=0.4``, and ``rnn_dim=512``.

After the search process, the program applies the best parameters to obtain the final model by adding
the validation set for training. The average P@1 score is 83.65% on the test set.

Early Stopping of the Parameter Search
--------------------------------------

It is time consuming to search over the entire parameter space.
To save time, LibMultiLabel has incorporated some early stopping techniques implemented in `Ray <https://arxiv.org/abs/1807.05118>`_.

Here we demonstrate an example of applying an `ASHA (Asynchronous Successive Halving Algorithm) Scheduler <https://arxiv.org/abs/1810.05934>`_.

First, uncomment the following lines in the
`configuration file <https://github.com/ASUS-AICS/LibMultiLabel/blob/master/example_config/EUR-Lex/bigru_lwan_tune.yml>`_:

.. code-block:: bash

    scheduler:
      time_attr: training_iteration
      max_t: 50
      grace_period: 10
      reduction_factor: 3
      brackets: 1

Under the same computing environment and the same command, the best parameter set of ``learning_rate=0.001``,
``embed_dropout=0.4``, ``encoder_dropout=0.2``, and ``rnn_dim=512`` is found in 47% of the time compared to the
grid search, while the average test P@1 score = 82.90% is similar to the result without early stopping.

A summary of results is in the following table. Four Nvidia Tesla V100 GPUs were used in this experiment.


.. list-table::
   :widths: 50 25 25 25 25 50
   :header-rows: 1

   * - Methods
     - Macro-F1
     - Micro-F1
     - P@1
     - P@5
     - Training Time (GPU)

   * - wo/ parameter selection
     - 20.48
     - 51.56
     - 78.13
     - 52.16
     - 27.8 minutes
   * - w/ parameter selection (grid search)
     - 23.65
     - 59.41
     - 83.65
     - 58.72
     - 24.6 hours
   * - w/ parameter selection (ASHA)
     - 22.70
     - 57.42
     - 82.90
     - 56.38
     - 11.6 hours
