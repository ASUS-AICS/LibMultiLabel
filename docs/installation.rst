Environments and Installation
=============================

Environments
------------

* Python: 3.7+
* CUDA: 10.2 (if GPU used)
* Pytorch 1.8+

If you have a different version of CUDA, go to the `website <https://pytorch.org/>`_ for the detail of PyTorch installation.


Installation
------------

Install via pip (for API use)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip3 install libmultilabel

For parameter search,

.. code-block:: bash

    pip3 install libmultilabel[parameter-search]

Install from Source (for command-line and/or API uses)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Clone `LibMultiLabel <https://github.com/ASUS-AICS/LibMultiLabel>`_.
* Install the latest development version, run:

.. code-block:: bash

    pip3 install -r requirements.txt
