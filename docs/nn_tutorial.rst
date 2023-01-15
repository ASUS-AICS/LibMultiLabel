==========================================================================
Neural Network Quickstart Tutorial (Two different pages)
==========================================================================

We go through two popular neural network examples

    * `BERT <../auto_examples/plot_bert_tutorial.html>`_ 
    * `KimCNN <../auto_examples/plot_KimCNN_tutorial.html>`_ 

in this tutorial. Before we start, please download and decompress the data ``rcv1`` via the following commands::

    mkdir -p data/rcv1
    wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_train.txt.bz2 -O data/rcv1/train.txt.bz2
    wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_test.txt.bz2 -O data/rcv1/test.txt.bz2
    bzip2 -d data/rcv1/*.bz2

.. toctree::
     :maxdepth: 1
     :hidden:
     
     auto_examples/plot_bert_tutorial
     auto_examples/plot_KimCNN_tutorial
