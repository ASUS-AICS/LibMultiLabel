LibMultiLabel - a Library for Multi-label Text Classification
=============================================================

LibMultiLabel is a library for multi-label text classification
and a simple command line tool with the following functionalities

- end-to-end services from raw texts to final evaluation/analysis
- support for common neural network architectures and linear classsifiers
- easy hyper-parameter selection


It can be used with `Command Line Interface <cli/cli.html>`_ or as APIs.

**Using CLI**

The command line tool is used as:

.. code-block:: bash

   python3 main.py --config main_config.yml
   python3 search_params.py --config search_config.yml

See `Command Line Tool <cli/cli.html>`_ for more information. You can also take a look at our `quick start guide <../tutorial/nn_cli_tutorial.html>`_ to learn how to build a nerual network with LibMultiLabel.

**Using APIs**

The library is composed of a neural network module and a linear classifier module::

   import libmultilabel.nn
   import libmultilabel.linear

See *API Documentation* for more information or get start with `the API tutorial <../tutorial/nn_api_tutorial.html>`_.

------

.. toctree::
    :caption: Overview
    :maxdepth: 1

    installation
    data_format

.. toctree::
    :caption: Command Line Interface
    :maxdepth: 1
    :glob:

    cli/*

.. toctree::
   :caption: API Documentation
   :maxdepth: 1
   :glob:

   api/*

.. toctree::
   :caption: Tutorials
   :maxdepth: 1
   :glob:

   tutorial/*

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
