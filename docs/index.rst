LibMultiLabel - a Library for Multi-label Text Classification
=============================================================

LibMultiLabel is a library for multi-label text classification
and a simple command line tool with the following functionalities

- end-to-end services from raw texts to final evaluation/analysis
- support for common neural network architectures and linear classsifiers
- easy hyper-parameter selection

The tool is used as::

   python3 main.py --config main_config.yml
   python3 search_params.py --config search_config.yml

The library is composed of a neural network module and a linear classifier module::

   import libmultilabel.nn
   import libmultilabel.linear

------

.. toctree::
    :caption: Environments and Installation
    :maxdepth: 1

    installation

.. toctree::
    :caption: Data Format
    :maxdepth: 1

    data_format

.. toctree::
    :caption: Command Line Interface
    :maxdepth: 1

    cli/cli
    cli/options
    cli/linear
    cli/nn

.. toctree::
   :caption: API Documentation
   :maxdepth: 1

   api/linear
   api/nn

.. toctree::
   :caption: Tutorials
   :maxdepth: 1

   tutorial/nn_api_tutorial
   tutorial/nn_cli_tutorial

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
