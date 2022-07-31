Command Line Options
====================

All command line options may be specified in either a config file
or directly passed as flags. If an option exists in both the config
file and flags, flags takes precedent and overrides the config file.

The config file is a yaml file. Each key-value pair ``key: value`` corresponds to
passing the flag ``--key value``. For example, to set the data directory
in the config file

.. code-block:: yaml

    data_dir: path/to/data

For lists of arguments such as ``monitor_metrics``, the value is a list

.. code-block:: yaml

    monitor_metrics: [P@1, P@3, P@3]

The corresponding command line flag is passed as

.. code-block:: bash

    python3 main.py --monitor_metrics P@1 P@3 P@5

For values which contains dashes such as ``liblinear_options``,
the flag has to include an additional ``=``

.. code-block:: bash

    python3 main.py --liblinear_options='-s 2 -B 1 -e 0.0001 -q'

List of Options
^^^^^^^^^^^^^^^

.. include:: flags.include
