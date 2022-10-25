==================================
Neural Network Quickstart Tutorial
==================================

We go through two popular neural network examples

    * `BERT <nn_tutorial.html#bert-example>`_ 
    * `KimCNN <nn_tutorial.html#kimcnn-example>`_ 

in this tutorial. Before started, please download and decompress the data ``EUR-Lex`` via the following commands::

    mkdir -p data/EUR-Lex
    wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/eurlex_raw_texts_train.txt.bz2 -O data/EUR-Lex/train.txt.bz2
    wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/eurlex_raw_texts_test.txt.bz2 -O data/EUR-Lex/test.txt.bz2
    bzip2 -d data/EUR-Lex/*.bz2

See `data formats <../cli/ov_data_format.html#dataset-formats>`_ for a complete explanation of the format.


BERT Example
============

This example shows how to train and test a BERT model via LibMultiLabel step-by-step.
 
Note that this example requires around 9 GB GPU memory. 
If your GPU device is not satisfied this requirement, please reduce the ``batch_size`` in `Step 7 <nn_tutorial.html#step-7-create-data-loaders>`_.



Step 1. Import the libraries
----------------------------

Please add the following code to your python3 script.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 2-4

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

We assume that the ``EUR-Lex`` data is located at the directory ``./data/EUR-Lex``, 
and there exists the files ``train.txt`` and ``test.txt``.
You can utilize

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 11

to load the data sets. Note that ``datasets`` contains three sets: ``datasets[train]``, ``datasets[val]`` and ``datasets[test]``, 
where ``datasets[train]`` and ``datasets[val]`` are randomly splitted from ``train.txt`` with the ratio ``8:2``.
The details can be found in `here <../api/nn.html#libmultilabel.nn.data_utils.load_datasets>`_, and you can also check out other arguments.

For the labels of the data, we apply

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 12

to generate the label set.


Step 4. Setup word preprocessing setting
-----------------------------------------

There are two methods to mapping a word to a vector in LibMultiLabel, and we need to decide which one is required by the model.
For BERT, we utilize the API ``AutoTokenizer``, which is supported by ``Hugging Face``, to get the word preprocessing setting of BERT.
Thus, we set the other variables for word preprocessing as ``None``.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 15-16


Step 5. Initialize a model
--------------------------

Let us introduce the variables as the inputs of ``init_model`` function on the following.

    * ``model_name`` leads ``init_model`` function to find a network model.

     .. literalinclude:: ../examples/bert_quickstart.py
         :language: python3
         :lines: 19

     More details are in `here <nn_networks.html>`_.

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
    :lines: 26-35

to initialize a model with a suitable learning rate setting.

Step 6. Initialize a trainer
----------------------------

We use the function ``init_trainer`` to initialize a trainer, which controls processes such as the number of training loops. 
The example is shown as the follows.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 38

In this example, we set the number of training loops as ``epochs=15``, and focus on the metric ``P@5`` over validation set.
For the other variables of ``init_trainer``, please check in `here <../api/nn.html#libmultilabel.nn.nn_utils.init_trainer>`_.

Step 7. Create data loaders
---------------------------

In most cases, we do not load full data set for training/validating/testing a bert model due to the limitation of hardware, 
which usually comes from the insufficient memory storage issue. 
Therefore, data loader can load the data set as many random sampling subsets, which is usually denoted by ``batch``, 
and the hardware can then handle a batch of the data in one time.

Let us show an example that creates pytorch data loaders form the datasets we created in
`Step 3 <nn_tutorial.html#step-3-load-libmultilabel-format-data-set>`_.

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 41-52

This example loads three loaders, and the batch size is set by ``batch_size=8``. Other variables can be checked in `here <../api/nn.html#libmultilabel.nn.data_utils.get_dataset_loader>`_.

Step 8. Train and test a model
------------------------------

Note that we have initialized the ``model`` in `Step 5 <nn_tutorial.html#step-5-initialize-a-model>`_, 
and the ``trainer`` in `Step 6 <nn_tutorial.html#step-6-initialize-a-trainer>`_.
With the ``train`` and ``val`` data loaders which are created in `Step 7 <nn_tutorial.html#step-7-create-data-loaders>`_, 
the bert model training process can be started via 

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 55

When the training process is finished, we can then run the testing process by

.. literalinclude:: ../examples/bert_quickstart.py
    :language: python3
    :lines: 58

After the testing process, the results are looked similar to::

  {
      'Macro-F1': 0.1828878672314435, 
      'Micro-F1': 0.5573605895042419,
      'P@1':      0.8023285865783691,
      'P@3':      0.6796895265579224,
      'P@5':      0.5613453984260559
  }

Please get the full example code in `here <https://github.com/ASUS-AICS/LibMultiLabel/tree/master/docs/examples/bert_quickstart.py>`_.


KimCNN Example
==============

This example shows how to train and test a KimCNN model via LibMultiLabel step-by-step. 
Since many steps are as same as BERT's steps, those steps are skipped here.
We only list the parts that are different to BERT example.

Note that this example requires around 2.2 GB GPU memory. 
If your GPU device is not satisfied this requirement, please reduce the ``batch_size``.


Step 1. Import the libraries
----------------------------

There is only one different part we need to take care between KimCNN and BERT:

    * The preprocess function that maps a word to a vector.

Therefore, we import the function ``load_or_build_text_dict`` to replace the function ``AutoTokenizer``. Please consider the following code.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 2-4

Step 2. Setup device
--------------------

This step is as same as `BERT example's Step 2 <nn_tutorial.html#step-2-setup-device>`_.


Step 3. Load LibMultiLabel format data set
------------------------------------------

KimCNN's data preprocessing is similar to BERT's, but there still exists one difference. 

* KimCNN applies another preprocess function for mapping a word to a vector, so we use a different setting for loading the data sets.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 11 

The usage of ``load_or_build_label`` in KimCNN is as same as the usage in BERT, so please consider `BERT example's Step 3 <nn_tutorial.html#step-3-load-libmultilabel-format-data-set>`_.

Step 4. Setup word preprocessing setting
-----------------------------------------

KimCNN required another word preprocessing function that is different to the one used by BERT in LibMultiLabel.
In this example, we choose torchtext's ``glove.6B.300d`` as the embedding of the mapping information, 
which requires to download a ``862 MB`` file at the first time usage.
  
.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 15-16
 
Note that you can either

* choose one of the pretrained embeddings defined in torchtext,
* or specify the path of your word embeddings, in which each line contains a word followed by its representational vector.


Step 5. Initialize a model
--------------------------

Of course, we have to determine the model name

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 19

to let LibMultiLabel know which model is used. 
Furthermore, since KimCNN's network structure is different to BERT's, the setting of ``network_config`` needs to be modified as follows. 

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 20-25

The usage of ``init_model`` in KimCNN is as same as the usage in BERT, except the learning setting.

.. literalinclude:: ../examples/nn_quickstart.py
    :language: python3
    :lines: 26

For the other parts, please consider `BERT example's Step 5 <nn_tutorial.html#step-5-initialize-a-model>`_.

Step 6. Initialize a trainer
----------------------------

This step is as same as `BERT example's Step 6 <nn_tutorial.html#step-6-initialize-a-trainer>`_.

Step 7. Create data loaders
---------------------------

This step is as same as `BERT example's Step 7 <nn_tutorial.html#step-7-initialize-a-trainer>`_.

Step 8. Train and test a model
--------------------------------

This step is as same as `BERT example's Step 8 <nn_tutorial.html#step-8-train-and-test-a-model>`_.

After the testing process, the results are looked similar to::

  {
      'Macro-F1': 0.14401986047701182,
      'Micro-F1': 0.46785199642181396,
      'P@1':      0.7345407009124756, 
      'P@3':      0.5999137163162231, 
      'P@5':      0.4993014633655548  
  }

Please get the full example code in `here <https://github.com/ASUS-AICS/LibMultiLabel/tree/master/docs/examples/nn_quickstart.py>`_.

