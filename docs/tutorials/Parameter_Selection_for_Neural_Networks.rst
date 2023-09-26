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
      post_encoder_dropout: 0.4
      rnn_dim: 512
      rnn_layers: 1

The training command is:

.. code-block:: bash

    python3 main.py --config example_config/EUR-Lex/bigru_lwan.yml

After training for 50 epochs, the checkpoint with the best validation performance is stored for testing. The
average P@1 score on the test data set is 80.36%.

Next, the ``learning_rate`` is changed to 0.003 while other parameters are kept the same.

.. code-block:: bash

    learning_rate: 0.003
    network_config:
      embed_dropout: 0.4
      post_encoder_dropout: 0.4
      rnn_dim: 512
      rnn_layers: 1

By the same training command, the P@1 score of the second parameter set is about 78.65%, which is
2% lower than the first one. This demonstrates the importance of parameter selection.

For more striking examples on the importance of parameter selection, you can see `this paper <https://www.csie.ntu.edu.tw/~cjlin/papers/parameter_selection/acl2021_parameter_selection.pdf>`_.

.. _Parameter Selection for Neural Networks:

Grid Search over Parameters
---------------------------

In the configuration file, we specify a grid search on the following parameters.


.. code-block:: bash

    learning_rate: ['grid_search', [0.003, 0.001, 0.0003]]
    network_config:
      embed_dropout: ['grid_search', [0, 0.2, 0.4, 0.6, 0.8]]
      post_encoder_dropout: ['grid_search', [0, 0.2, 0.4]]
      rnn_dim: ['grid_search', [256, 512, 1024]]
      rnn_layers: 1
    embed_cache_dir: .vector_cache

We set the ``embed_cache_dir`` to ``.vector_cache`` to avoid downloading pre-trained embeddings repeatedly for each configuration.


Then the training command is:

.. code-block:: bash

    python3 search_params.py --config example_config/EUR-Lex/bigru_lwan_tune.yml

The process finds the best parameter set of ``learning_rate=0.0003``, ``embed_dropout=0.6``, ``post_encoder_dropout=0.2``, and ``rnn_dim=256``.

After the search process, the program applies the best parameters to obtain the final model by adding
the validation set for training. The average P@1 score is 81.99% on the test set, better
than the result without a hyper-parameter search. Note that after obtaining the best 
hyper-parameters, we combine training and validation sets to train a final model for testing.
For more details about 're-training', please refer to the `Re-train or not`_ section.

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
``embed_dropout=0.4``, ``post_encoder_dropout=0.2``, and ``rnn_dim=512`` is found in 26% of the time compared to the
grid search, while the average test P@1 score is similar to the result without early stopping.

A summary of results is in the following table. Eight Nvidia Tesla V100 GPUs were used in this experiment.


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
     - 20.79
     - 54.91
     - 80.36 
     - 53.89
     - 42.5 minutes
   * - w/ parameter selection (grid search)
     - 24.43
     - 57.99
     - 81.99
     - 57.57
     - 23.0 hours
   * - w/ parameter selection (ASHA)
     - 23.07
     - 58.03
     - 82.33
     - 57.07
     - 5.89 hours

Re-train or not
--------------------------------------

In the `Grid Search over Parameters`_ section, we split the available data into training 
and validation sets for hyperparameter search. For methods like SVM, they usually train the 
final model with the best hyper-parameters by combining the training and validation sets. 
This approach maximizes the utilization of information for model learning, and we refer to 
it as the "re-train" strategy.

.. However, when applied in deep learning, merging the validation set into the training 
.. set means that the optimization process, which previously relied on the validation set for 
.. termination, no longer works. While there's no definitively proven best termination criterion 
.. , a typical approach is to determine the optimal epoch during 
.. hyper-parameter search based on the number of training steps that led to the best 
.. validation performance. This optimal epoch serves as a stopping criterion 
.. when training the model with all available data. This strategy has been shown 
.. to provide stable improvements while mitigating the risk of overfitting.

Since re-training is usually beneficial, we have incorporated the strategy into ``search_params.py``.
When hyper-parameter search is done, the re-training process will be automatically 
executed by default, like the case in section `Grid Search over Parameters`_.

Though not recommended, you can use the argument ``--no_retrain`` to disable the 
re-training process.

.. code-block:: bash

    python search_params.py --config example_config/EUR-Lex/bigru_lwan.yml --no_retrain

By doing so, the model achieving the best validation performance during parameter search
will be returned.
In this case, the P@1 performance with re-training shows an improvement of approximately 2%
compared to the performance without re-training. The following test results illustrate
the advantages of the re-training.

.. list-table::
   :widths: 50 25 25 25 25
   :header-rows: 1

   * - Methods
     - Macro-F1
     - Micro-F1
     - P@1
     - P@5

   * - wo/ re-training after hyper-parameter search
     - 22.95
     - 56.37
     - 80.08
     - 56.24

   * - w/ re-training after hyper-parameter search
     - 24.43
     - 57.99
     - 81.99
     - 57.57

In a different scenario, if you want to skip the parameter search but still wish 
to re-train the model with your chosen hyper-parameters, we will provide an example 
of how to do this.

Let's train a BiGRU model using the configuration file used in the `Direct Trying Some Parameters`_ 
section, where the learning rate is set to 0.001. Please note that because the validation set 
is not specified in the configuration file, the training dataset is partitioned into 
a training set and a validation subsets to assess the performance at each epoch.


.. code-block:: bash

    python main.py --config example_config/EUR-Lex/bigru_lwan.yml

Using the model obtained at the epoch of the best validation PR@5,
the test performance is:

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Macro-F1
     - Micro-F1
     - P@1
     - P@5

   * - 20.79
     - 54.91
     - 80.36 
     - 53.89

To get the epoch with the best validation performance, the following code snippet reads 
the log, extracts the performance metrics for each epoch, and identifies the optimal epoch:

.. code-block:: python

    import json
    import numpy as np

    with open('your_log_path_for_the_first_step.json', 'r') as r: # the log file which records the configuration and validation performance of each epoch is saved in the 'runs' directory by default.
        log = json.load(r)
    log_metric = np.array([l[log["config"]["val_metric"]] for l in log["val"]])
    optimal_idx = log_metric.argmax() # if your validation metric is loss, use np.argmin() instead.
    best_epoch = optimal_idx.item() + 1
    print(best_epoch)

In this case, the optimal epoch should be 42.
We then specify ``--no_merge_train_val`` to include the validation set for training and 
specify the number of epochs by ``--epochs``. Note that options explicitly defined 
override those in the configuration file. Because of no validation set, only the model
at the last epoch is returned.

.. code-block:: bash

    python main.py --config example_config/EUR-Lex/bigru_lwan.yml --epochs 42 --merge_train_val

Similar with the last case, the test performance improves after re-training:

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Macro-F1
     - Micro-F1
     - P@1
     - P@5

   * - 22.65
     - 57.06
     - 83.10
     - 56.34