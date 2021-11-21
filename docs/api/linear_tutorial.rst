Linear Classification Quickstart Tutorial
=========================================

.. role:: py(code)
   :language: python3

This guide will take you through how LibMultiLabel can
be used to train a linear classifier in python scripts.

Getting the Dataset
^^^^^^^^^^^^^^^^^^^

For this guide, we will use the rcv1 dataset, which is
a collection of news articles.
On the command line::

    mkdir -p data/rcv1
    wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_train.txt.bz2 -O data/rcv1/train.txt.bz2
    wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_test.txt.bz2 -O data/rcv1/test.txt.bz2
    bzip2 -d data/rcv1/*.bz2

Each line of the dataset contains an ID, its labels and the
text, all seperated by tabs. For example::

    2286    E11 ECAT M11 M12 MCAT   recov recov recov recov excit excit bring mexic ...

See `data formats <placeholder>`_ for a complete explanation of the format.

The Python Script
^^^^^^^^^^^^^^^^^

For this guide, we will only need the linear module:

.. literalinclude:: ../examples/linear_quickstart.py
    :language: python3
    :lines: 1

To start, we need to read and preprocess the input data:

.. literalinclude:: ../examples/linear_quickstart.py
    :language: python3
    :lines: 3-6

The preprocessor handles many issues such as: mapping
the labels into indices and transforming textual data to
numerical data. The loaded dataset has the structure::

    {
        'train': {
            'x': # training features
            'y': # training labels
        },
        'test': {
            'x': # test features
            'y': # test labels
        },
    }

Next we train the model:

.. literalinclude:: ../examples/linear_quickstart.py
    :language: python3
    :lines: 8-10

The third argument is the options string for
`LibLinear <https://www.csie.ntu.edu.tw/~cjlin/liblinear/>`_.
We may leave it as the default for now.

Once we have the model, we may predict with it:

.. literalinclude:: ../examples/linear_quickstart.py
    :language: python3
    :lines: 12

:py:`preds` holds the decision values, i.e. the raw values
outputted by the model. To transform it into predictions,
the simplest way is to take the positive values as
the labels predicted to be associated with the sample,
i.e. :py:`preds > 0`.

To see how well we performed, we may want to check various
metrics with the test set.
For that we may use:

.. literalinclude:: ../examples/linear_quickstart.py
    :language: python3
    :lines: 14-16

This creates the set of metrics we wish to see.
Since the dataset we loaded are stored as :py:`scipy.sparse.csr_matrix`,
we need to transform them to :py:`np.array` before we can compute the metrics:

.. literalinclude:: ../examples/linear_quickstart.py
    :language: python3
    :lines: 18

Finally, we compute and print the metrics:

.. literalinclude:: ../examples/linear_quickstart.py
    :language: python3
    :lines: 20-21

The results will look similar to::

    {'Micro-F1': 0.8007651838722984, 'Macro-F1': 0.5171356568703283, 'P@1': 0.9573166595201372, 'P@3': 0.7991526562689997, 'P@5': 0.5580654451434532}

Get the full source code `here <https://github.com/ASUS-AICS/LibMultiLabel/tree/master/docs/examples/linear_quickstart.py>`_.
