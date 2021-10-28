Dataset Formats
===============

The input data for building train, test, and validation dataset must have specific formats.
For neural networks, the only accepted format is the
:ref:`libmultilabel-format`. For linear methods,
both :ref:`libmultilabel-format` and
:ref:`libsvm-format` is accepted.

.. _libmultilabel-format:

LibMultiLabel Format
--------------------

The LibMultiLabel format is a format for labels,
raw texts and optional IDs. It combines them
in a single file, using tabs and line endings as control characters.
It has

- one sample per line
- seperate ID, labels and texts by ``<TAB>`` (the ID column is optional)
- labels are split by spaces
- should not contain any ``<TAB>`` in each field

With ID column::

    2286<TAB>E11 ECAT M11 M12 MCAT<TAB>recov recov recov recov excit ...
    2287<TAB>C24 CCAT<TAB>uruguay uruguay compan compan compan ...

Without ID column::

    E11 ECAT M11 M12 MCAT<TAB>recov recov recov recov excit ...
    C24 CCAT<TAB>uruguay uruguay compan compan compan ...

.. _libsvm-format:

LibSVM Format
-------------

The LibSVM format is a format for labels and sparse numerical
features. It combines them in a single file,
using commas, spaces, colons and line endings as control characters.
It is as follows

- one sample per line
- seperate labels and features with a space
- labels are split by commas
- features are split by spaces
- features are specified as ``index:value``, with index starting from ``1``

For example::

    1,3,5 1:0.1 9:0.2 13:0.3
    2,4,6 2:0.4 10:0.5 14:0.4
