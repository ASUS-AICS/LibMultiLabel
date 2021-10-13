LibMultiLabel - a Library for Multi-label Text Classification
=============================

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

See the API documentation for more details.

Quick Start
-----------

*Something here*

API Documentation
-----------------

.. toctree::
   :maxdepth: 2

   modules/linear
   modules/nn

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
