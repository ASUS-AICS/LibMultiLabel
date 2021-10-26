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

Model Options
^^^^^^^^^^^^^

Train Options
^^^^^^^^^^^^^

Evaluation Options
^^^^^^^^^^^^^^^^^^