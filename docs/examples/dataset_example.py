# Step 1. Import libraries 
import pandas
from datasets import load_dataset

# Step 2. Load data set from Hugging Face
data = dict()
data['train'] = load_dataset('rotten_tomatoes', split='train')
data['val'] = load_dataset('rotten_tomatoes', split='validation')
data['test'] = load_dataset('rotten_tomatoes', split='test')

# Step 3. Transform to LibMultiLabel data set
for tag in ['train', 'val', 'test']:
    data[tag] = pandas.DataFrame(data[tag], columns=['label', 'text'])
    data[tag] = data[tag].reset_index()
    data[tag]['label'] = data[tag]['label'].astype(str).map(lambda s: s.split())
    data[tag] = data[tag].to_dict('records')

# Step 4. Remove the data with no labels (This Step is for Training and Validation set)
for tag in ['train', 'val']:
    num_no_label_data = sum(1 for d in data[tag] if len(d['label']) == 0)
    if num_no_label_data > 0:
        data[tag] = [d for d in data[tag] if len(d['label']) > 0]
