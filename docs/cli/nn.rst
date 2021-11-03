Training, Prediction, and Hyperparameter Search for Neural Networks
===================================================================

Training and (Optional) Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the training procedure, you can build a model from scratch or start from some pre-obtained information.

.. code-block:: bash

    python3 main.py --config CONFIG_PATH \
                    [--checkpoint_path CHECKPOINT_PATH] \
                    [--embed_file EMBED_NAME_OR_EMBED_PATH] \
                    [--vocab_file VOCAB_CSV_PATH]

- **config**: configure parameters in a yaml file. See ``python3 main.py --help``.

If a model was trained before by this package, the training procedure can start with it.

- **checkpoint_path**: specify the path to a pre-trained model.

To use your own word embeddings or vocabulary set, specify the following parameters:

- **embed_file**: choose one of the pretrained embeddings defined in `torchtext <https://pytorch.org/text/0.9.0/vocab.html#torchtext.vocab.Vocab.load_vectors>`_ or specify the path to your word embeddings with each line containing a word followed by its vectors. Example:

.. code-block::

    the 0.04656 0.21318 -0.0074364 ...
    a -0.29712 0.094049 -0.096662 ...
    an -0.3206 0.43316 -0.086867 ...

- **vocab_file**: set the file path to a predefined vocabulary set that contains lines of words.

.. code-block::

    the
    a
    an

For validation, you can evaluate the model with a set of evaluation metrics.
Set ``monitor_metrics`` to define what you want to print on the screen.
The argument ``val_metric`` is the metric for selecting the best model.
Namely, the model occurred at the epoch with the best validation metric is returned after training.
If you do not specify a validation set in the configuration file via ``val_path`` or a training-validation split ratio via ``val_size``,
we will split the training data into training and validation set with an 80-20 split.
Example lines in a configuration file:

.. code-block:: yaml

    monitor_metrics: [P@1, P@3, P@5]
    val_metric: P@1


If ``test_path`` is specified or ``DATA_DIR/test.txt`` exists, the model with the highest ``val_metric`` will be used to predict the test set.

Prediction
^^^^^^^^^^

To deploy/evaluate a model (i.e., a pre-obtained checkpoint), you can predict a test set by the following command.

.. code-block:: bash

    python3 main.py --eval \
                    --config CONFIG_PATH \
                    --checkpoint_path CHECKPOINT_PATH \
                    --test_path TEST_DATA_PATH \
                    --save_k_predictions K \
                    --predict_out_path PREDICT_OUT_PATH

- Use ``--save_k_predictions`` to save the top K predictions for each instance in the test set. K=100 if not specified.
- Use ``--predict_out_path`` to specify the file for storing the predicted top-K labels/scores.

Hyperparameter Search
^^^^^^^^^^^^^^^^^^^^^

Parameter selection is known to be extremely important in machine learning practice; see a powerful reminder in "`this paper <https://www.csie.ntu.edu.tw/~cjlin/papers/parameter_selection/acl2021_parameter_selection.pdf>`_". Here we leverage `Ray Tune <https://docs.ray.io/en/master/tune/index.html>`_, which is a python library for hyperparameter tuning, to select parameters. Due to the dependency of Ray Tune, first make sure your python version is not greater than 3.8. Then, install the related packages with::

    pip3 install -Ur requirements_parameter_search.txt

We provide a program ``search_params.py`` to demonstrate how to run LibMultiLabel with Ray Tune. An example is as follows::

    python3 search_params.py --config example_config/rcv1/cnn_tune.yml \
                             --search_alg basic_variant

- **config**: configure *all* parameters in a yaml file. You can define a continuous, a discrete, or other types of search space (see a list `here <https://docs.ray.io/en/master/tune/api_docs/search_space.html#tune-sample-docs>`_). An example of configuring the parameters is presented as follows:

.. code-block:: yaml

    dropout: ['grid_search', [0.2, 0.4, 0.6, 0.8]] # grid search
    num_filter_per_size: ['choice', [350, 450, 550]] # discrete
    learning_rate: ['uniform', 0.2, 0.8] # continuous
    activation: tanh # not for hyperparameter search

- **search_alg**: specify a search algorithm considered in `Ray Tune <https://docs.ray.io/en/master/tune/api_docs/suggestion.html>`_. We support basic_variant (e.g., grid/random), bayesopt, and optuna. You can also define ``search_alg`` in the config file. For example, if you want to run grid search over ``learning_rate``, the config is like this:

.. code-block:: yaml

    search_alg: basic_variant
    learning_rate: ['grid_search', [0.2, 0.4, 0.6, 0.8]]