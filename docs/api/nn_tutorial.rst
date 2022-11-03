==================================
Neural Network Quickstart Tutorial
==================================

We go through two popular neural network examples

    * `BERT <nn_tutorial.html#bert-example>`_ 
    * `KimCNN <nn_tutorial.html#kimcnn-example>`_ 

in this tutorial. Before we start, please download and decompress the data ``EUR-Lex`` via the following commands::

    mkdir -p data/EUR-Lex
    wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/eurlex_raw_texts_train.txt.bz2 -O data/EUR-Lex/train.txt.bz2
    wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/eurlex_raw_texts_test.txt.bz2 -O data/EUR-Lex/test.txt.bz2
    bzip2 -d data/EUR-Lex/*.bz2


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

We assume that the ``EUR-Lex`` data is located at the directory ``./data/EUR-Lex``, 
and there exist the files ``train.txt`` and ``test.txt``.
You can utilize the function ``load_datasets()`` to load the data sets. 
By default, LibMultiLabel tokenizes documents, but the BERT model uses its own tokenizer. 
Thus, we must set ``tokenize_text=False``.
Note that ``datasets`` contains three sets: ``datasets['train']``, ``datasets['val']`` and ``datasets['test']``, 
where ``datasets['train']`` and ``datasets['val']`` are randomly splitted from ``train.txt`` with the ratio ``8:2``.

For the labels of the data, we apply the function ``load_or_build_label()`` to generate the label set.

For BERT, we utilize the API ``AutoTokenizer``, which is supported by ``Hugging Face``, for the word preprocessing setting.
Thus, we set the other variables for word preprocessing as ``None``.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 11-14


Step 4. Initialize a model
--------------------------

We use the following code to initialize a model.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 17-33

* ``model_name`` leads ``init_model`` function to find a network model.
* ``network_config`` contains the configurations of a network model.
* ``classes`` is the label set of the data.
* ``init_weight``, ``word_dict`` and ``embed_vecs`` are not used on a bert-base model, so we can ignore them.
* ``moniter_metrics`` is used to define the metrics you would like to track during the training procedure.
    

Step 5. Initialize a trainer
----------------------------

We use the function ``init_trainer`` to initialize a trainer, which controls processes such as the number of training loops. 

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 36

In this example, we set the number of training loops by ``epochs=15``, and the validation metric by ``val_metric = 'P@5'``.

Step 6. Create data loaders
---------------------------

In most cases, we do not load a full set due to the hardware limitation.
Therefore, a data loader can load a batch of samples each time.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 39-50

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
      'Macro-F1': 0.1828878672314435, 
      'Micro-F1': 0.5573605895042419,
      'P@1':      0.8023285865783691,
      'P@3':      0.6796895265579224,
      'P@5':      0.5613453984260559
  }

Please get the full example code `here <https://github.com/ASUS-AICS/LibMultiLabel/tree/master/docs/examples/bert_quickstart.py>`_.


KimCNN Example
==============

This example shows how to train and test a KimCNN model via LibMultiLabel. 
We only list the steps that are different from the BERT example.


Step 3. Load and tokenize data
------------------------------------------

To run KimCNN, LibMultiLabel tokenizes documents and for each word uses an embedding vector. 
Thus, ``tokenize_text = True`` is set.

We choose ``glove.6B.300d`` from torchtext as embedding vectors. 

.. literalinclude:: ../examples/kimcnn_quickstart.py
    :language: python3
    :lines: 11-14
 

Step 4. Initialize a model
--------------------------

We consider the following settings for the KimCNN model.

.. literalinclude:: ../examples/kimcnn_quickstart.py
    :language: python3
    :lines: 17-33



The test results should be similar to::

  {
      'Macro-F1': 0.14401986047701182,
      'Micro-F1': 0.46785199642181396,
      'P@1':      0.7345407009124756, 
      'P@3':      0.5999137163162231, 
      'P@5':      0.4993014633655548  
  }

Please get the full example code `here <https://github.com/ASUS-AICS/LibMultiLabel/tree/master/docs/examples/kimcnn_quickstart.py>`_.

