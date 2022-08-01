Command Line Options
====================

Command line options may be specified in either a config file
or directly passed as flags. If an option exists in both the config
file and flags, flags take precedent and override the config file.

The config file is a yaml file, examples may be found in
`example_config <https://github.com/ASUS-AICS/LibMultiLabel/tree/master/example_config>`_.
In the config file, each key-value pair ``key: value`` corresponds to
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
use ``=`` to pass the value on the command line

.. code-block:: bash

    python3 main.py --liblinear_options='-s 2 -B 1 -e 0.0001 -q'

Hierarchical parameters such as ``network_config`` must be specified in a config file.

List of Options
^^^^^^^^^^^^^^^

.. include:: flags.include
