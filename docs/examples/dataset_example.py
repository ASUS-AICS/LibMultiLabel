# Step 1. Import libraries 
import pandas
from datasets import load_dataset

# Step 2. Load data set from Hugging Face
tr_data = load_dataset("rotten_tomatoes", split="train")
te_data = load_dataset("rotten_tomatoes", split="test")

# Step 3. Transform to LibMultiLabel data set
tr_data = pandas.DataFrame(tr_data, columns=["label", "text"])
tr_data = tr_data.reset_index()
tr_data['label'] = tr_data['label'].astype(str).map(lambda s: s.split())
tr_data = tr_data.to_dict('records')
te_data = pandas.DataFrame(te_data, columns=["label", "text"])
te_data = te_data.reset_index()
te_data['label'] = te_data['label'].astype(ste).map(lambda s: s.split())
te_data = te_data.to_dict('records')

# Step 4. Remove the data with no labels (This Step is for Training and Validation set)
num_no_label_data = sum(1 for d in tr_data if len(d['label']) == 0)
if num_no_label_data > 0:
    tr_data = [d for d in tr_data if len(d['label']) > 0]
