Using Neural Networks Methods
=============================


Quick Start
-----------


Step 1. Load datasets and build dictionaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's start by preparing the datasets.
    * Put ``train.txt`` and ``test.txt`` in your data directory.
    * Build the train, test, and validation sets with ``data_utils.load_datasets``.
      The function will split the train and the validation sets for you based on ``val_size``.
      You can also provide your own one by placing a ``valid.txt`` in the data directory.

Text and label dictionaries are

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 14-18

Step 2. Initialize a model
^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we show how to create a model.
    * Configure the network with ``model_name`` and ``network_config``.
    * Define the label and text dictionaries (i.e., ``classes`` and ``word_dict``).
    * Find `more here <../api/nn.html#libmultilabel.nn.nn_utils.init_model>`_ if you are interested in other settings. For example, a ``moniter_metrics`` is used to defined the metrics you'd like to keep track with during the training procedure.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 21-30

Step 3. Initialize trainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To train the model, we will need a trainer to finish all the tasks.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 33-34

Step 4. Create data loaders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create data loaders.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 37-51

Step 5. Train a model from scratch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Everything's ready. Let's start training!

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 54-54
