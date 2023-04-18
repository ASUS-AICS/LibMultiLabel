'''
Use LibMultiLabel for single- or multi-label classification
===========================================================

Generally, a classification task can be categorized as single-label (or binary and multi-class) or multi-label
classification depending on if labels are mutually exclusive. Single- and multi-label classification have different
functions to calculate training loss and evaluation metrics. Thus machine learning practitioners should pay attention
to the details both categories. LibMultiLabel can handle both tasks by using different functions in the linear package.
'''

# %%
# 1. Single-label classification
# ------------------------------

'''The 20 newsgroups dataset comprises around 18000 newsgroups posts on 20 topics split in two subsets: one for 
training (or development) and the other one for testing (or for performance evaluation). 
The split between the train and test set is based upon a messages posted before and after a specific date.
'''
import warnings
import sys
from pathlib import Path
sys.path.append('..')

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

import libmultilabel.linear as linear

warnings.filterwarnings('ignore')

# Prepare raw data
remove = ('headers', 'footers', 'quotes')
raw_train = fetch_20newsgroups(
    subset="train",
    remove=remove
)

raw_test = fetch_20newsgroups(
    subset="test",
    remove=remove
)

"""There is no need to do TF-IDF here as it will be implemented under the hood"""
# Prepare dataframe
df_train = pd.DataFrame([raw_train.target, raw_train.data]).rename({0: 'label', 1: 'text'}).T
df_test = pd.DataFrame([raw_test.target, raw_test.data]).rename({0: 'label', 1: 'text'}).T

# Prepare dataset
preprocessor = linear.Preprocessor(data_format='dataframe')

dataset = preprocessor.load_data(df_train, df_test)
single_label_clf = linear.train_binary_and_multiclass(dataset['train']['y'], dataset['train']['x'], options='')

single_label_preds = linear.predict_values(single_label_clf, dataset['test']['x'])
target = dataset['test']['y'].toarray()

'''
As each instance has exactly one label, only naive precision of F1 
'''
single_label_score = linear.compute_metrics(single_label_preds, target, ['P@1', 'RP@1', 'Macro-F1', 'Micro-F1'])
print("Score of single label clf:", single_label_score)

'''
2. Multi-label classification
-------

Precision is the fraction of relevant instances among the retrieved instances

Multi-class classification looks the same as binary classification except that the number of available classes/labels are larger than 2.
Each sample can only be labeled as one class.
The application of multi-class classification is diverse, from identifying
'''
