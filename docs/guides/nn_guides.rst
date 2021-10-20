Using Neural Networks Methods
=============================


Quick Start
-----------

In this quick start guide, we will introduce neural networks methods via a simple example.
You will learn how to:

    * Preprocess datasets.
    * Initialize a model.
    * Train and test a ``KimCNN`` model on the RCV1 dataset.

Step 0. Setup seed and device
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's start by setting up the seed and device.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 12-13

Step 1. Load data from text files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we're going to process the RCV1 data.
First, create the training, test, and validation datasets from text files.

    * Put ``train.txt`` and ``test.txt`` in your data directory (i.e., ``data/rcv1``).
    * Load and preprocess raw data with ``data_utils.load_datasets``.
      The function performs a training-validation split based on the ``val_size``.
      You can also provide your own validation set by adding ``valid.txt`` in the data directory.

Then, build labels and word dictionary with datasets generated above.
You can either choose one of the pretrained embeddings defined in torchtext or specify the path
to your word embeddings with each line containing a word followed by its vectors.


.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 16-20

Step 2. Initialize a model
^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we show how to create a model.

    * Configure the network with `model_name <../guides/nn_guides.html#networks>`_ and ``network_config``.
    * Define labels and text dictionary (i.e., ``classes`` and ``word_dict``) of this model.
    * Find `more here <../api/nn.html#libmultilabel.nn.nn_utils.init_model>`_ if you are interested in other settings.
      For example, a ``moniter_metrics`` is used to define the metrics you'd like to keep track with during the training procedure.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 23-32

Step 3. Initialize trainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To train the model, we need a trainer to control all the process like training loop or validation.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 35-37

Step 4. Create data loaders
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create pytorch data loaders for datasets we created in
`Step 1 <../guides/nn_guides.html#step-1-load-data-from-text-files>`_.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 40-49

Step 5. Train and test the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Everything's ready. Let's start training with ``trainer.train``.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 52-52


When training is finished, test the model with ``trainer.test``.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 55-55


Get the full source code `here <https://github.com/ASUS-AICS/LibMultiLabel/tree/master/docs/examples/nn_quickstart.py>`_.

-----------

Networks
-----------

* BiGRU
* CAML
* KimCNN
* XMLCNN