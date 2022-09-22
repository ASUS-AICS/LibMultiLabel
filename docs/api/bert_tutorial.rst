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
Other arguments of ``data_utils.load_datasets`` can be found `here <../api/nn.html#libmultilabel.nn.data_utils.load_datasets>`_.

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

We use the function ``init_trainer`` to initialize a trainer, which controls processes such as the number of training loops. 
The example is shown as the follows.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 39-41

In this example, we set the number of training loops as ``epochs=15``, and further focus the metric ``P@5`` on validation data. 
For the other variables of ``init_trainer``, please check in `here <../api/nn.html#libmultilabel.nn.nn_utils.init_trainer>`_.

Step 4. Create data loaders
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In most cases, we do not load full data set for training/validating/testing a bert model due to the limitation of hardware, 
which usually comes from the insufficient memory storage issue. 
Therefore, data loader can load the data set as many random sampling subsets, which is usually denoted by ``batch``, 
and the hardware can then handle one batch of the data in one time.

Let us show an example that creates pytorch data loaders form the datasets we created in
`Step 1 <bert_tutorial.html#step-1-load-libmultilabel-format-data-set>`_.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 44-54

This example loads three loaders, and the batch size is set by ``batch_size=8``. Other variables can be checked in `here <../api/nn.html#libmultilabel.nn.data_utils.get_dataset_loader>`_.

Step 5. Train and test a model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that we have initialize the ``model`` in `Step 2 <bert_tutorial.html#step-2-initialize-a-model>`_, 
and the ``trainer`` in `Step 3 <bert_tutorial.html#step-3-initialize-a-trainer>`_.
With the ``train`` and ``val`` data loaders which are created in `Step 4 <bert_tutorial.html#step-4-create-data-loaders>`_, 
the bert model training process can be started via 

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 57

When the training process is finished, we can then run the testing process by

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 60

After the testing process, the results are looked similar to::

  {'Macro-F1': 0.24686789646612792, 'Micro-F1': 0.7266660332679749, 'P@1': 0.9016666412353516, 'P@3': 0.8070555925369263, 'P@5': 0.6758000254631042}

Please get the full example code in `here <https://github.com/ASUS-AICS/LibMultiLabel/tree/master/docs/examples/bert_quickstart.py>`_.
