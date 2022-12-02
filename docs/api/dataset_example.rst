========================================
Using LibMultiLabel with Your Data Sets
========================================

We show you how to use our library with other data set formats.


An Example from Hugging Face Data set
=====================================

Let us use a data set from Hugging Face to train a KimCNN model in this example.
Please install Hugging Face's library ``datasets`` by the following command::

  pip3 install datasets --user


Step 1. Import the libraries
----------------------------

We use pandas' ``DataFrame`` to process the data set from Hugging Face, 
and load the data set by Hugging Face's API ``load_dataset``. 
Hence, please import these two libraries. 

.. literalinclude:: ../examples/dataset_component.py
    :language: python3
    :lines: 2-3

Step 2. Load Data Set from Hugging Face
---------------------------------------

You can pick a multi-label classification data set from `Hugging Face's Data Set Pool <https://huggingface.co/datasets>`_.
Let us choose ``rotten_tomatoes`` in this example. 
The training and test set can be loaded by the following code.

.. literalinclude:: ../examples/dataset_component.py
    :language: python3
    :lines: 6-9


Step 3. Transform to LibMultiLabel data set
-------------------------------------------

The following code shows how to transform Hugging Face data set format to LibMultiLabel data set format 
by the functions of ``DataFrame``.


.. literalinclude:: ../examples/dataset_component.py
    :language: python3
    :lines: 12-16


Step 4. Remove the data with no labels
--------------------------------------

For the training set, if a data has empty label, it will be removed from the set, 
which is implemented by the following code.

.. literalinclude:: ../examples/dataset_component.py
    :language: python3
    :lines: 19-22


Step 5. Training and test a KimCNN model with the data sets we made
-------------------------------------------------------------------

The following code is almost the same as our `KimCNN example <nn_tutorial.html#kimcnn-example>`_.

.. literalinclude:: ../examples/dataset_component.py
    :language: python3
    :lines: 25-26

The test results should be similar to::

  {
      'Macro-F1': 0.740196337755889,
      'Micro-F1': 0.7404974102973938,
      'P@1':      0.7420262694358826,
  }

Please get the full example code `here <https://github.com/ASUS-AICS/LibMultiLabel/tree/master/docs/examples/dataset_example.py>`_.
