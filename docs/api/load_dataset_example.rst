========================================
Using LibMultiLabel with Your Data Sets
========================================

We show you how to use our library with other data set formats.


An Example from Hugging Face Data set
=====================================

bla

Step 1. Import the libraries
----------------------------

We use pandas' ``DataFrame`` to process the data set from Hugging Face, 
and load the data set by Hugging Face's API ``load_dataset``. 
Hence, please import these two libraries. 

.. literalinclude:: ../examples/dataset_example.py
    :language: python3
    :lines: 2-3

Step 2. Load Data Set from Hugging Face
---------------------------------------

You can pick a multi-label classification data set from `Hugging Face's Data Set Pool <https://huggingface.co/datasets>`_.
Let us choose ``rotten_tomatoes`` in this example. 
The training and test set can be loaded by the following code.

.. literalinclude:: ../examples/dataset_example.py
    :language: python3
    :lines: 6-7


Step 3. Transform to LibMultiLabel data set
-------------------------------------------

The following code shows how to transform Hugging Face data set format to LibMultiLabel data set format 
by the functions of ``DataFrame``.


.. literalinclude:: ../examples/dataset_example.py
    :language: python3
    :lines: 10-17


Step 4. Remove the data with no labels
--------------------------------------

For the training set, if a data has empty label, it will be removed from the set, 
which is implemented by the following code.

.. literalinclude:: ../examples/dataset_example.py
    :language: python3
    :lines: 20-22

