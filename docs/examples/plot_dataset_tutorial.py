"""
An Example of Using Data Stored in Different Forms
===================================================

Different data sets are stored in various structures and formats.
To apply LibMultiLabel with any of them, one must convert the data to a form accepted by the library first.
In this tutorial, we demonstrate an example of converting a hugging face data set.
Before we start, note that LibMultiLabel format consists of IDs (optional), labels, and raw texts.
`Here <https://www.csie.ntu.edu.tw/~cjlin/libmultilabel/cli/ov_data_format.html#libmultilabel-format>`_ are more details.

To begin, let's install ``datasets`` from the hugging face library by the following command::

  pip3 install datasets

We use Pandas' ``DataFrame`` to process the data set from Hugging Face
and load the data set by Hugging Face's API ``load_dataset``.
Please import these libraries.
"""

import pandas
from datasets import load_dataset

######################################################################
# We choose a multi-label set ``emoji`` from ``tweet_eval`` in this example.
# The data set can be loaded by the following code.

data_sets = dict()
data_sets['train'] = load_dataset('tweet_eval', 'emoji', split='train')
data_sets['val'] = load_dataset('tweet_eval', 'emoji', split='validation')
data_sets['test'] = load_dataset('tweet_eval', 'emoji', split='test')

######################################################################
# Convert to LibMultiLabel format
# --------------------------------------------------
# We load Hugging Face's data set to Pandas' structure by the function ``DataFrame``.
# In consistent with our `linear model quickstart <https://www.csie.ntu.edu.tw/~cjlin/libmultilabel/auto_examples/plot_linear_quickstart.html>`_, which does not need a validation set,
# we use ``pandas.concat`` to merge training and validation sets and use ``reset_index`` to add the new indices to rows.

for split in ['train', 'val', 'test']:
    data_sets[split] = pandas.DataFrame(data_sets[split], columns=['label', 'text'])
data_sets['train'] = pandas.concat([data_sets['train'], data_sets['val']], axis=0, ignore_index=True)
data_sets['train'] = data_sets['train'].reset_index()
data_sets['test'] = data_sets['test'].reset_index()

###############################################################################
# The format of the data set after conversion looks like below::
#
#   >>> print(data_sets['train'].loc[[0]]) #print first row
#   ...
#   index  label                                               text
#       0     12  Sunday afternoon walking through Venice in the...
#
# Next, we train and make prediction with the data set using a linear model. The detailed explanation is in our `linear model quickstart <https://www.csie.ntu.edu.tw/~cjlin/libmultilabel/auto_examples/plot_linear_quickstart.html>`_.
# The difference between here and the quickstart is that the ``data_format`` option should be ``dataframe`` because the data set is a dataframe now.

import libmultilabel.linear as linear

preprocessor = linear.Preprocessor(data_format='dataframe')
datasets = preprocessor.load_data(data_sets['train'], data_sets['test'])

###############################################################################
# Also, if you want to use a NN model,
# use ``load_datasets`` from ``libmultilabel.nn.data_utils`` and change the data to the dataframes we created.
# Here is the modification of our `Bert model quickstart <https://www.csie.ntu.edu.tw/~cjlin/libmultilabel/auto_examples/plot_BERT_quickstart.html>`_.

from libmultilabel.nn.data_utils import load_datasets

datasets = load_datasets(data_sets['train'], data_sets['test'], tokenize_text=False)