import json

load_Hugging_Face_data_sets = dict()
load_Hugging_Face_data_sets["HEAD"] = """
import pandas
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

data_sets = dict()
data_sets['train'] = load_dataset('tweet_eval', 'emoji', split='train')
data_sets['val'] = load_dataset('tweet_eval', 'emoji', split='validation')
data_sets['test'] = load_dataset('tweet_eval', 'emoji', split='test')

for tag in ['train', 'val', 'test']:
    data_sets[tag] = pandas.DataFrame(data_sets[tag], columns=['label', 'text'])
    data_sets[tag] = data_sets[tag].reset_index()
"""
load_Hugging_Face_data_sets["NN_part"] = """
for tag in ['train', 'val', 'test']:
    data_sets[tag]['label'] = data_sets[tag]['label'].astype(str).map(lambda s: s.split())
    data_sets[tag] = data_sets[tag].to_dict('records')

for tag in ['train', 'val']:
    num_no_label_data = sum(1 for d in data_sets[tag] if len(d['label']) == 0)
    if num_no_label_data > 0:
        data_sets[tag] = [d for d in data_sets[tag] if len(d['label']) > 0]
"""
load_Hugging_Face_data_sets["Linear_part"] = """
for tag in ['train', 'val', 'test']:
    data_sets[tag]['label'] = data_sets[tag]['label'].map(str)
    data_sets[tag] = data_sets[tag].to_dict('list')

data_sets['train']['label'] += data_sets['val']['label']
data_sets['train']['text'] += data_sets['val']['text']
data_sets['train']['index'] += list(map( lambda x: x + data_sets['train']['index'][-1] + 1, data_sets['val']['index'] ))

classes = set(data_sets['train']['label'] + data_sets['test']['label'])
classes = sorted([cls for cls in classes], key=int)

vectorizer = TfidfVectorizer()
binarizer = MultiLabelBinarizer(sparse_output=True, classes=classes)   
vectorizer.fit(data_sets['train']['text'])
binarizer.fit(data_sets['train']['label'])
for tag in ['train', 'test']:
    data_sets[tag]['x'] = vectorizer.transform(data_sets[tag]['text'])
    data_sets[tag]['y'] = binarizer.transform(data_sets[tag]['label']).astype('d')

num_labels = data_sets['train']['y'].getnnz(axis=1)
num_no_label_data = np.count_nonzero(num_labels == 0)
if num_no_label_data > 0:
    if remove_no_label_data:
        data_sets['train']['x'] = data_sets['train']['x'][num_labels > 0]
        data_sets['train']['y'] = data_sets['train']['y'][num_labels > 0]
"""

with open("dataset_component.json", "w") as F:
    json.dump(load_Hugging_Face_data_sets, F)
