"""
An Example of Using Data Stored in Different Forms.
===================================================

Different data sets are stored in various structures and formats.
To apply LibMultiLabel with any of them, one must convert the data to a form accepted by the library first.  
In this tutorial, we demonstrate an example of conversion of a hugging face data set.
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
# We begin with loading Hugging Face's data set to Pandas' structure by the function ``DataFrame``. 
# In consistence with our `linear model quickstart <https://www.csie.ntu.edu.tw/~cjlin/libmultilabel/auto_examples/plot_linear_quickstart.html>`_ which don't have validation set,
# we use ``pandas.concat`` to merge training and validation sets and use ``reset_index`` to add the new indices to rows.

for d_set in ['train', 'val', 'test']:
    data_sets[d_set] = pandas.DataFrame(data_sets[d_set], columns=['label', 'text'])
data_sets['train'] = pandas.concat([data_sets['train'], data_sets['val']], axis=0, ignore_index=True)
data_sets['train'] = data_sets['train'].reset_index()
data_sets['test'] = data_sets['test'].reset_index()
data_sets['train'].to_csv('train.txt', header=None, index=None, sep='\t')

###############################################################################
# The format of the data set before conversion looks like below::
#
#   >>> print(data_sets['train'].loc[[0]]) #print first row
#   ...
#   index  label                                               text
#       0     12  Sunday afternoon walking through Venice in the...  
#
# Now we start to convert the data set. 
# The following code first transforms the data into strings which are in LibMultiLabel format.
# The option ``sep='\t'`` seperates ID, label and raw texts fields by ``<TAB>`` and the other options remove the redundant row and column.

data_sets['train'] = data_sets['train'].to_csv(header=None, index=None, sep='\t')
data_sets['test'] = data_sets['test'].to_csv(header=None, index=None, sep='\t')

#######################################################################
# After the last step, the data set is converted to `the required format <https://www.csie.ntu.edu.tw/~cjlin/libmultilabel/cli/ov_data_format.html#libmultilabel-format>`_::
#
#   >>> print(data_sets['train'].partition('\n')[0])
#   ... 
#   0       12      Sunday afternoon walking through Venice in the sun with @user ️ ️ ️ @ Abbot Kinney, Venice
#
# Next, we train and make prediction with the data set. The detailed explanation is in our `linear model quickstart <https://www.csie.ntu.edu.tw/~cjlin/libmultilabel/auto_examples/plot_linear_quickstart.html>`_. 
# The difference from the quickstart is that when using ``load_data``, we transform the string to a file-like object using ``StringIO``, since ``load_data`` only accepts file path or file-like object.

from io import StringIO
import libmultilabel.linear as linear
preprocessor = linear.Preprocessor(data_format='txt')
datasets = preprocessor.load_data(StringIO(data_sets['train']),
                                  StringIO(data_sets['test']))
model = linear.train_1vsrest(datasets['train']['y'], datasets['train']['x'], '')
preds = linear.predict_values(model, datasets['test']['x'])
metrics = linear.get_metrics(metric_threshold=0,
                             monitor_metrics=['Macro-F1', 'Micro-F1', 'P@1', 'P@3', 'P@5'],
                             num_classes=datasets['test']['y'].shape[1])
target = datasets['test']['y'].toarray()
metrics.update(preds, target)
print(metrics.compute())




