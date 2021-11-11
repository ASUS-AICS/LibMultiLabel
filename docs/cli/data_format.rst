Dataset Formats
===============

The input data for building train, test, and validation datasets must have specific formats.
For neural networks, the only accepted format is the
:ref:`libmultilabel-format`. For linear methods,
both :ref:`libmultilabel-format` and
:ref:`libsvm-format` are accepted.
The formatted datasets, RCV1 and EUR-LEX-57K, can be downloaded from the
`LIBSVM datasets <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html>`_ website.

.. _libmultilabel-format:

LibMultiLabel Format
--------------------

The LibMultiLabel format is a format for IDs (optional), labels, and raw texts.
They are combined in a single file, using tabs and line endings as control characters.
It must satisfy the following requirements

- one sample per line
- ID, labels, and texts are separated by ``<TAB>`` (the ID column is optional)
- labels are split by spaces
- each field should not contain any ``<TAB>``

An example with the ID column::

    2286<TAB>E11 ECAT M11 M12 MCAT<TAB>recov recov recov recov excit ...
    2287<TAB>C24 CCAT<TAB>uruguay uruguay compan compan compan ...

An example without the ID column::

    E11 ECAT M11 M12 MCAT<TAB>recov recov recov recov excit ...
    C24 CCAT<TAB>uruguay uruguay compan compan compan ...

.. _libsvm-format:

LibSVM Format
-------------

The LibSVM format is a format for labels and sparse numerical
features. They are combined in a single file,
using commas, spaces, colons and line endings as control characters.
It must meet the criteria below

- one sample per line
- labels and features are separated by a space
- labels are split by commas
- features are split by spaces
- each feature is specified as ``index:value``, with index starting from ``1``

Some sample lines are as follows::

    1,3,5 1:0.1 9:0.2 13:0.3
    2,4,6 2:0.4 10:0.5 14:0.4
