# Step 1. Import libraries 
import pandas
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# Step 2. Load data set from Hugging Face
data = dict()
data['train'] = load_dataset('tweet_eval', 'emoji', split='train')
data['test'] = load_dataset('tweet_eval', 'emoji', split='test')

# Step 3. Transform to LibMultiLabel data set
for tag in ['train', 'test']:
    data[tag] = pandas.DataFrame(data[tag], columns=['label', 'text'])
    data[tag] = data[tag].reset_index()
    data[tag]['label'] = data[tag]['label'].map(str)
    data[tag] = data[tag].to_dict('list')

classes = set(data['train']['label'] + data['test']['label'])
classes = sorted([cls for cls in classes], key=int)

vectorizer = TfidfVectorizer()
binarizer = MultiLabelBinarizer(sparse_output=True, classes=classes)   
vectorizer.fit(data['train']['text'])
binarizer.fit(data['train']['label'])
for tag in ['train', 'test']:
    data[tag]['x'] = vectorizer.transform(data[tag]['text'])
    data[tag]['y'] = binarizer.transform(data[tag]['label']).astype('d')


# Step 4. Remove the data with no labels (This Step is for Training and Validation set)
num_labels = data['train']['y'].getnnz(axis=1)
num_no_label_data = np.count_nonzero(num_labels == 0)
if num_no_label_data > 0:
    if remove_no_label_data:
        data['train']['x'] = data['train']['x'][num_labels > 0]
        data['train']['y'] = data['train']['y'][num_labels > 0]


# Using LibMultiLabel with the datasets
import libmultilabel.linear as linear

model = linear.train_1vsrest(data['train']['y'], data['train']['x'], '')

preds = linear.predict_values(model, data['test']['x'])

metrics = linear.get_metrics(
        metric_threshold=0, 
        monitor_metrics=['Macro-F1', 'Micro-F1', 'P@1', 'P@3', 'P@5'], 
        num_classes=data['test']['y'].shape[1])

target = data['test']['y'].toarray()

metrics.update(preds, target)
print(metrics.compute())


