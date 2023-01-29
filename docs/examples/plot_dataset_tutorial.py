

"""
========================================
Using LibMultiLabel with Your Data Sets
========================================

We show you how to use our library with other data set formats.


An Example from Hugging Face Data set
=====================================

Let us use a data set from Hugging Face to train a linear model in this example.
Please install Hugging Face's library ``datasets`` by the following command::

  pip3 install datasets

Step 1. Import the libraries
----------------------------

We use pandas' ``DataFrame`` to process the data set from Hugging Face, 
and load the data set by Hugging Face's API ``load_dataset``.
Furthermore, we utilize some functions from ``numpy`` and ``scikit-learn``
to transform the words and labels to vectors.
Hence, please import these libraries. 
"""

import pandas
from datasets import load_dataset
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

######################################################################
# Step 2. Load Data Set from Hugging Face
# ---------------------------------------
# 
# You can pick a multi-label classification data set from `Hugging Face's Data Set Pool <https://huggingface.co/datasets>`_.
# Let us choose ``emoji`` from ``tweet_eval`` in this example. 
# The training and test sets can be loaded by the following code.
# Note that we do not consider validation set in this example.

data_sets = dict()
data_sets['train'] = load_dataset('tweet_eval', 'emoji', split='train')
data_sets['test'] = load_dataset('tweet_eval', 'emoji', split='test')
#data_sets['val'] = load_dataset('tweet_eval', 'emoji', split='validation')

######################################################################
# Step 3. Transform to LibMultiLabel data structure
# --------------------------------------------------
# 
# We show how to load Hugging Face's data to LibMultiLabel's data structure in this step.
#
#   * First of all, we load Hugging Face's data set to pandas' structure by the functions of ``DataFrame``, and give tags to columns.
#   * Second, we use ``reset_index`` to add indices to rows.
#   * Third, the label structure in each row should be tranfered to a list of strings (e.g., ["0", "12"] means this row data contains the labels "0" and "12").
#   * In the final, we export the data set from ``DataFrame`` to python's ``dict`` structure.

#for tag in ['train', 'val', 'test']:
for tag in ['train', 'test']:
    data_sets[tag] = pandas.DataFrame(data_sets[tag], columns=['label', 'text'])
    data_sets[tag] = data_sets[tag].reset_index()
    data_sets[tag]['label'] = data_sets[tag]['label'].map( lambda s: str(s).split() )
    data_sets[tag] = data_sets[tag].to_dict('list')

#data_sets['train']['label'] += data_sets['val']['label']
#data_sets['train']['text'] += data_sets['val']['text']
#data_sets['train']['index'] += list(map( lambda x: x + data_sets['train']['index'][-1] + 1, data_sets['val']['index'] ))

######################################################################
# Now, the data set is transfered to python's ``dict`` structure, but it is still a text-base data set. 
# For example, let us show the first two rows of the training set::
#
#  >>> for row in data_sets['train']['text'][:2]:
#  ...     print(row)
#  ...
#  Sunday afternoon walking through Venice in the sun ... @ Abbot Kinney, Ven
#  Time for some BBQ and whiskey libations. Chomp, ... Smokehouse Bar-B-Que)
#
# Hence, we utilize ``TfidfVectorizer`` to apply TF-IDF transformation on our text data with the default setting from ``scikit-learn``.

vectorizer = TfidfVectorizer()
vectorizer.fit(data_sets['train']['text'])
for tag in ['train', 'test']:
    data_sets[tag]['x'] = vectorizer.transform(data_sets[tag]['text'])

######################################################################
# Moreover, since the data set only contain the positive labels, we use ``MultiLabelBinarizer`` to process the labels to the general multi-label setting.
# Let us use an example to explain more details::
#
#  >>> print( data_sets['test']['label'][0:2] )
#  [['2'], ['10']]
#  >>> print( data_sets['test']['y'].todense()[0:2] )
#  [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
#
# where ``data_sets['test']['y']`` is tranformed from ``data_sets['test']['label']`` by the following code.

binarizer = MultiLabelBinarizer(sparse_output=True, classes=None)   
binarizer.fit(data_sets['train']['label'])
for tag in ['train', 'test']:
    data_sets[tag]['y'] = binarizer.transform(data_sets[tag]['label']).astype('d')

######################################################################
# Step 4. Remove the data with no labels (Optional)
# -------------------------------------------------
# 
# For the training set, if a data has empty label, it will be removed from the set, 
# which is implemented by the following code.

num_labels = data_sets['train']['y'].getnnz(axis=1)
num_no_label_data = np.count_nonzero(num_labels == 0)
if num_no_label_data > 0:
    if remove_no_label_data:
        data_sets['train']['x'] = data_sets['train']['x'][num_labels > 0]
        data_sets['train']['y'] = data_sets['train']['y'][num_labels > 0]

######################################################################
# Step 5. Training and test a linear model with the data sets we made
# -------------------------------------------------------------------
# 
# Let us use the code from `Linear Classification Quickstart Tutorial <../auto_examples/plot_linear_tutorial.html>`_, except the data set part.


import libmultilabel.linear as linear

model = linear.train_1vsrest(data_sets['train']['y'], data_sets['train']['x'], '')

preds = linear.predict_values(model, data_sets['test']['x'])

metrics = linear.get_metrics(metric_threshold=0,
                             monitor_metrics=['Macro-F1', 'Micro-F1', 'P@1', 'P@3', 'P@5'],
                             num_classes=data_sets['test']['y'].shape[1])

target = data_sets['test']['y'].toarray()

metrics.update(preds, target)
print(metrics.compute())

######################################################################
# The test results should be similar to::
#
#   {'Macro-F1': 0.14032183271967594, 'Micro-F1': 0.189649330848721, 'P@1': 0.26744, 'P@3': 0.15701333333333334, 'P@5': 0.11880399999999999} 
# 





