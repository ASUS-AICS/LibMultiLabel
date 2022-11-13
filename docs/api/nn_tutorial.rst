==================================
Neural Network Quickstart Tutorial
==================================

We go through two popular neural network examples

    * `BERT <nn_tutorial.html#bert-example>`_ 
    * `KimCNN <nn_tutorial.html#kimcnn-example>`_ 

in this tutorial. Before we start, please download and decompress the data ``rcv1`` via the following commands::

    mkdir -p data/rcv1
    wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_train.txt.bz2 -O data/rcv1/train.txt.bz2
    wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_test.txt.bz2 -O data/rcv1/test.txt.bz2
    bzip2 -d data/rcv1/*.bz2


BERT Example
============

This step-by-step example shows how to train and test a BERT model via LibMultiLabel.


Step 1. Import the libraries
----------------------------

Please add the following code to your python3 script.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 2-4

Step 2. Setup device
--------------------

If you need to reproduce the results, please use the function ``set_seed``. 
For example, you will get the same result as you always use the seed ``1337``.

For initial a hardware device, please use ``init_device`` to assign the hardware device that you want to use.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 7-8

Step 3. Load and tokenize data
------------------------------------------

We assume that the ``rcv1`` data is located at the directory ``./data/rcv1``, 
and there exist the files ``train.txt`` and ``test.txt``.
You can utilize the function ``load_datasets()`` to load the data sets. 
By default, LibMultiLabel tokenizes documents, but the BERT model uses its own tokenizer. 
Thus, we must set ``tokenize_text=False``.
Note that ``datasets`` contains three sets: ``datasets['train']``, ``datasets['val']`` and ``datasets['test']``, 
where ``datasets['train']`` and ``datasets['val']`` are randomly splitted from ``train.txt`` with the ratio ``8:2``.

For the labels of the data, we apply the function ``load_or_build_label()`` to generate the label set.

For BERT, we utilize the API ``AutoTokenizer``, which is supported by ``Hugging Face``, for the word preprocessing setting.
Furthermore, BERT applies some special tokens such as ``<CLS>``, so that we take ``add_special_tokens=True``.
Therefore, we set other variables for word preprocessing as ``None``.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 11-15


Step 4. Initialize a model
--------------------------

We use the following code to initialize a model.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 18-32

* ``model_name`` leads ``init_model`` function to find a network model.
* ``network_config`` contains the configurations of a network model.
* ``classes`` is the label set of the data.
* ``init_weight``, ``word_dict`` and ``embed_vecs`` are not used on a bert-base model, so we can ignore them.
* ``moniter_metrics`` includes metrics you would like to track.
    

Step 5. Initialize a trainer
----------------------------

We use the function ``init_trainer`` to initialize a trainer. 

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 35

In this example, ``checkpoint_dir`` is the place we save the best and the last models during the training. Furthermore, we set the number of training loops by ``epochs=15``, and the validation metric by ``val_metric = 'P@5'``.

Step 6. Create data loaders
---------------------------

In most cases, we do not load a full set due to the hardware limitation.
Therefore, a data loader can load a batch of samples each time.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 38-50

This example loads three loaders, and the batch size is set by ``batch_size=8``. Other variables can be checked in `here <../api/nn.html#libmultilabel.nn.data_utils.get_dataset_loader>`_.

Step 7. Train and test a model
------------------------------

The bert model training process can be started via 

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 53

After the training process is finished, we can then run the test process by

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 56

The results should be similar to::

  {
      'Macro-F1': 0.569891024909958, 
      'Micro-F1': 0.8142925500869751, 
      'P@1':      0.9552904367446899, 
      'P@3':      0.7907078266143799, 
      'P@5':      0.5505486726760864
  }

Please get the full example code `here <https://github.com/ASUS-AICS/LibMultiLabel/tree/master/docs/examples/bert_quickstart.py>`_.


KimCNN Example
==============

This example shows how to train and test a KimCNN model via LibMultiLabel. 
We only list the steps that are different from the BERT example.


Step 3. Load and tokenize data
------------------------------------------

To run KimCNN, LibMultiLabel tokenizes documents and uses an embedding vector for each word. 
Thus, ``tokenize_text = True`` is set.

We choose ``glove.6B.300d`` from torchtext as embedding vectors. 

.. literalinclude:: ../examples/kimcnn_quickstart.py
    :language: python3
    :lines: 11-15
 

Step 4. Initialize a model
--------------------------

We consider the following settings for the KimCNN model.

.. literalinclude:: ../examples/kimcnn_quickstart.py
    :language: python3
    :lines: 18-34



The test results should be similar to::

  {
      'Macro-F1': 0.48948464335831743,
      'Micro-F1': 0.7769773602485657,
      'P@1':      0.9471677541732788,
      'P@3':      0.7772253751754761,
      'P@5':      0.5449321269989014,
  }

Please get the full example code `here <https://github.com/ASUS-AICS/LibMultiLabel/tree/master/docs/examples/kimcnn_quickstart.py>`_.

