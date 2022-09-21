Bert Quickstart Tutorial
==================================

Similar to our neural network quick start, you will learn how to

    * Preprocess datasets
    * Initialize a model
    * Train and test a ``Bert`` model on the EUR-LEX-57k dataset

via a simple example. Before we started, please download and decompress the data ``EUR-Lex-57k`` by utilizing the commands 

.. code-block:: bash

    mkdir -p data/EUR-Lex-57k
    wget -O data/EUR-Lex-57k/train.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/eurlex57k_raw_texts_train.txt.bz2
    wget -O data/EUR-Lex-57k/test.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/eurlex57k_raw_texts_test.txt.bz2
    wget -O data/EUR-Lex-57k/valid.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/eurlex57k_raw_texts_val.txt.bz2
    bzip2 -d *.bz2

on your command line. Moreover, the following libraries must be imported in your python3 script.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 1-4

Step 0. Setup device
^^^^^^^^^^^^^^^^^^^^

If you need to reproduce the results, please use the function ``set_seed``. For example,

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 7

fixes the random seed on ``1337``. You will get the same result as you always use the seed ``1337``.

For initial a hardware device, please use 

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 8

to assign the hardware device that you want to use.


Step 1. Load LibMultiLabel format data set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We assume that the ``EUR-Lex-57k`` data is located at the directory ``./data/EUR-Lex-57k``, 
and there exists the files ``train.txt``, ``valid.txt`` and ``test.txt``.
You can utilize

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 11-16

to load the data set.
Other arguments of ``data_utils.load_datasets`` can be found `here <../api/nn.html#libmultilabel.nn.data_utils.load_datasets>`__.

For the labels of the data, we apply

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 17

to generate the label set.

Step 2. Initialize a model
^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us introduce the variables as the inputs of ``init_model`` function on the following.

    * ``model_name`` leads ``init_model`` function to find a network model. More details are in `here <nn_networks.html>`_.
    * ``network_config`` contains the configurations of a network model. In this example, we set it as

     .. literalinclude:: ../examples/bert_quickstart.py
         :language: python3
         :lines: 20-25

    * ``classes`` is the label set of the data.
    * ``init_weight``, ``word_dict`` and ``embed_vecs`` are not used on a bert-base model, so we can ignore them.
    * You can check more information from `here <../api/nn.html#libmultilabel.nn.nn_utils.init_model>`_ if you are interested in other settings.
      For example, ``moniter_metrics`` is used to define the metrics you would like to track during the training procedure.
    
Overall, we use

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 26-36

to initialize a model.

Step 3. Initialize a trainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To train the model, we need a trainer to control processes like training loop or validation.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 30-32

Step 4. Create data loaders
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create pytorch data loaders for datasets we created in
`Step 1 <nn_tutorial.html#step-1-load-data-from-text-files>`_.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 35-41

Step 5. Train and test the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Everything's ready. Let's start training with ``trainer.train``.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 44

When training is finished, test the model with ``trainer.test``.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 47

The results will look similar to::

    {'Macro-F1': 0.4615944665965527, 'Micro-F1': 0.7823446989059448, 'P@1': 0.9514147043228149, 'P@3': 0.783452033996582, 'P@5': 0.549974799156189}

Get the full source code `here <https://github.com/ASUS-AICS/LibMultiLabel/tree/master/docs/examples/bert_quickstart.py>`_.
