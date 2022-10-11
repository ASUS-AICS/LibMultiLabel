==================================
Neural Network Quickstart Tutorial
==================================

We will go through two popular neural network examples

    * BERT 
    * KimCNN

in this tutorial. Before we started, please download and decompress the data ``EUR-Lex`` via the following commands::

    mkdir -p data/EUR-Lex
    wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/eurlex_raw_texts_train.txt.bz2 -O data/EUR-Lex/train.txt.bz2
    wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/eurlex_raw_texts_test.txt.bz2 -O data/EUR-Lex/test.txt.bz2
    bzip2 -d data/EUR-Lex/*.bz2

See `data formats <../cli/ov_data_format.html#dataset-formats>`_ for a complete explanation of the format.


BERT Example
============

You will learn how to train and predict a BERT model via LibMultiLabel step-by-step in this example. 
Note that this example requires 8.5 GB GPU memory. 
If your GPU device is not satisfied this requirement, please reduce the ``batch_size`` in `Step 6 <nn_tutorial.html#step-6-create-data-loaders>`_.



Step 1. Import the libraries
----------------------------

Please add the following code to your python3 script.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 1-4

Step 2. Setup device
--------------------

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


Step 3. Load LibMultiLabel format data set
------------------------------------------

We assume that the ``EUR-Lex`` data is located at the directory ``./data/Eur-Lex``, 
and there exists the files ``train.txt`` and ``test.txt``.
You can utilize

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 11-15

to load the data set.
Other arguments of ``data_utils.load_datasets`` can be found `here <../api/nn.html#libmultilabel.nn.data_utils.load_datasets>`_.

For the labels of the data, we apply

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 16

to generate the label set.

Step 4. Initialize a model
--------------------------

Let us introduce the variables as the inputs of ``init_model`` function on the following.

    * ``model_name`` leads ``init_model`` function to find a network model. More details are in `here <nn_networks.html>`_.
    * ``network_config`` contains the configurations of a network model. In this example, we set it as

     .. literalinclude:: ../examples/bert_quickstart.py
         :language: python3
         :lines: 19-24

    * ``classes`` is the label set of the data.
    * ``init_weight``, ``word_dict`` and ``embed_vecs`` are not used on a bert-base model, so we can ignore them.
    * You can check more information from `here <../api/nn.html#libmultilabel.nn.nn_utils.init_model>`_ if you are interested in other settings.
      For example, ``moniter_metrics`` is used to define the metrics you would like to track during the training procedure.
    
Overall, we use

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 25-35

to initialize a model.

Step 5. Initialize a trainer
----------------------------

We use the function ``init_trainer`` to initialize a trainer, which controls processes such as the number of training loops. 
The example is shown as the follows.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 38-40

In this example, we set the number of training loops as ``epochs=15``.
For the other variables of ``init_trainer``, please check in `here <../api/nn.html#libmultilabel.nn.nn_utils.init_trainer>`_.

Step 6. Create data loaders
---------------------------

In most cases, we do not load full data set for training/testing a bert model due to the limitation of hardware, 
which usually comes from the insufficient memory storage issue. 
Therefore, data loader can load the data set as many random sampling subsets, which is usually denoted by ``batch``, 
and the hardware can then handle one batch of the data in one time.

Let us show an example that creates pytorch data loaders form the datasets we created in
`Step 3 <nn_tutorial.html#step-3-load-libmultilabel-format-data-set>`_.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 43-53

This example loads two loaders, and the batch size is set by ``batch_size=8``. Other variables can be checked in `here <../api/nn.html#libmultilabel.nn.data_utils.get_dataset_loader>`_.

Step 7. Train and test a model
------------------------------

Note that we have initialize the ``model`` in `Step 4 <nn_tutorial.html#step-4-initialize-a-model>`_, 
and the ``trainer`` in `Step 5 <nn_tutorial.html#step-5-initialize-a-trainer>`_.
With the ``train`` data loader which is created in `Step 6 <nn_tutorial.html#step-6-create-data-loaders>`_, 
the bert model training process can be started via 

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 56

When the training process is finished, we can then run the testing process by

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 59

After the testing process, the results are looked similar to::

  {'Macro-F1': 0.24686789646612792, 'Micro-F1': 0.7266660332679749, 'P@1': 0.9016666412353516, 'P@3': 0.8070555925369263, 'P@5': 0.6758000254631042}

Please get the full example code in `here <https://github.com/ASUS-AICS/LibMultiLabel/tree/master/docs/examples/bert_quickstart.py>`_.


KimCNN Example
==============

In this section, we will introduce the neural network methods via a simple example.
You will learn how to:

    * Preprocess datasets.
    * Initialize a model.
    * Train and test a ``KimCNN`` model on the rcv1 dataset.

Before we started, make sure you import methods below:

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 1-3

Step 0. Setup device
--------------------

Let's start by setting up the device.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 6

Step 1. Load data from text files
---------------------------------

Now we're going to process the rcv1 data.
First, create the training, test, and validation (optional) datasets from text files.

    * Put ``train.txt`` and ``test.txt`` in your data directory (i.e., ``data/rcv1``).
      Please refer to `here <linear_tutorial.html#getting-the-dataset>`__ for more details.
    * Load and preprocess raw data with ``data_utils.load_datasets``.
      Other arguments can be found `here <../api/nn.html#libmultilabel.nn.data_utils.load_datasets>`__.
      For example, the function performs a training-validation split based on the ``val_size``.
      You can also provide your own validation set by adding ``valid.txt`` in the data directory.

Then, build labels and word dictionary with datasets generated above.
You can either choose one of the pretrained embeddings defined in torchtext or specify the path
to your word embeddings with each line containing a word followed by its vectors.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 9-13

Step 2. Initialize a model
--------------------------

Here we show how to create a model.

    * Configure the network with `model_name <nn_networks.html>`_ and ``network_config``.
    * Define labels and text dictionary (i.e., ``classes`` and ``word_dict``) of this model.
    * Find `more here <../api/nn.html#libmultilabel.nn.nn_utils.init_model>`_ if you are interested in other settings.
      For example, a ``moniter_metrics`` is used to define the metrics you'd like to keep track with during the training procedure.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 16-27

Step 3. Initialize a trainer
----------------------------

To train the model, we need a trainer to control processes like training loop or validation.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 30-32

Step 4. Create data loaders
---------------------------

Create pytorch data loaders for datasets we created in
`Step 1 <nn_tutorial.html#step-1-load-data-from-text-files>`_.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 35-41

Step 5. Train and test the model
--------------------------------

Everything's ready. Let's start training with ``trainer.train``.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 44

When training is finished, test the model with ``trainer.test``.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 47

The results will look similar to::

    {'Macro-F1': 0.4615944665965527, 'Micro-F1': 0.7823446989059448, 'P@1': 0.9514147043228149, 'P@3': 0.783452033996582, 'P@5': 0.549974799156189}

Get the full source code `here <https://github.com/ASUS-AICS/LibMultiLabel/tree/master/docs/examples/nn_quickstart.py>`_.

