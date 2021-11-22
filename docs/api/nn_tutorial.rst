Neural Network Quickstart Tutorial
==================================

In this section, we will introduce the neural network methods via a simple example.
You will learn how to:

    * Preprocess datasets.
    * Initialize a model.
    * Train and test a ``KimCNN`` model on the RCV1 dataset.

Before we started, make sure you import methods below:

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 1-3

Step 0. Setup seed and device
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's start by setting up the seed and device.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 7-8

Step 1. Load data from text files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we're going to process the RCV1 data.
First, create the training, test, and validation (optional) datasets from text files.

    * Put ``train.txt`` and ``test.txt`` in your data directory (i.e., ``data/rcv1``).
    * Load and preprocess raw data with ``data_utils.load_datasets``.
      The function performs a training-validation split based on the ``val_size``.
      You can also provide your own validation set by adding ``valid.txt`` in the data directory.

Then, build labels and word dictionary with datasets generated above.
You can either choose one of the pretrained embeddings defined in torchtext or specify the path
to your word embeddings with each line containing a word followed by its vectors.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 11-14

Step 2. Initialize a model
^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we show how to create a model.

    * Configure the network with `model_name <nn_networks.html>`_ and ``network_config``.
    * Define labels and text dictionary (i.e., ``classes`` and ``word_dict``) of this model.
    * Find `more here <../api/nn.html#libmultilabel.nn.nn_utils.init_model>`_ if you are interested in other settings.
      For example, a ``moniter_metrics`` is used to define the metrics you'd like to keep track with during the training procedure.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 17-26

Step 3. Initialize trainer
^^^^^^^^^^^^^^^^^^^^^^^^^^

To train the model, we need a trainer to control processes like training loop or validation.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 29-32

Step 4. Create data loaders
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create pytorch data loaders for datasets we created in
`Step 1 <../guides/nn_guides.html#step-1-load-data-from-text-files>`_.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 35-41

Step 5. Train and test the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Everything's ready. Let's start training with ``trainer.train``.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 44


When training is finished, test the model with ``trainer.test``.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 47


Get the full source code `here <https://github.com/ASUS-AICS/LibMultiLabel/tree/master/docs/examples/nn_quickstart.py>`_.
