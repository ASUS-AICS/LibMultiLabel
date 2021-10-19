Using Neural Networks Methods
=============================


Quick Start
-----------

In this quick start guide, we are going to walk through a training procedure via a simple example.
You will learn how to use LibMultiLabel to train a ``KimCNN`` model on the RCV1 dataset.


Step 1. Load datasets and build dictionaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's start by preparing the datasets.

    * Put ``train.txt`` and ``test.txt`` in your data directory.
    * Build the train, test, and validation sets with ``data_utils.load_datasets``.
      The function will split the train and the validation sets for you based on ``val_size``.
      You can also provide your own one by placing a ``valid.txt`` in the data directory.

Create label and word dictionaries with the datasets generated above.
You can either choose one of the pretrained embeddings defined in torchtext or specify the path to your word embeddings with each line containing a word followed by its vectors.


.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 14-18

Step 2. Initialize a model
^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we show how to create a model.

    * Configure the network with `model_name <../guides/nn_guides.html#networks>`_ and ``network_config``.
    * Define the label dictionary and the text dictionary (i.e., ``classes`` and ``word_dict``) of the model.
    * Find `more here <../api/nn.html#libmultilabel.nn.nn_utils.init_model>`_ if you are interested in other settings. For example, a ``moniter_metrics`` is used to define the metrics you'd like to keep track with during the training procedure.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 21-30

Step 3. Initialize trainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To train the model, we need a trainer to finish all the tasks.

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

-----------

Networks
-----------

* BiGRU
* CAML
* KimCNN
* XMLCNN