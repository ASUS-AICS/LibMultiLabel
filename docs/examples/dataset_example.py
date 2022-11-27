# Step 1. Import libraries 
import pandas
from datasets import load_dataset

# Step 2. Load data set from Hugging Face
data = dict()
data['train'] = load_dataset('tweet_eval', 'emoji', split='train')
data['val'] = load_dataset('tweet_eval', 'emoji', split='validation')
data['test'] = load_dataset('tweet_eval', 'emoji', split='test')

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

# Using LibMultiLabel with the datasets
from libmultilabel.nn.data_utils import *
from libmultilabel.nn.nn_utils import *

# Setup device.
set_seed(1337)
device = init_device()  # use gpu by default

# Preprocessing the datasets.
classes = load_or_build_label(data)
word_dict, embed_vecs = load_or_build_text_dict(dataset=data['train'], embed_file='glove.6B.300d')
tokenizer = None
add_special_tokens = False

# Initialize a model.
model_name = 'KimCNN'
network_config = {
    'embed_dropout': 0.2,
    'encoder_dropout': 0.2,
    'filter_sizes': [2, 4, 8],
    'num_filter_per_size': 128
}
learning_rate = 0.0003
model = init_model(
    model_name=model_name,
    network_config=network_config,
    classes=classes,
    word_dict=word_dict,
    embed_vecs=embed_vecs,
    learning_rate=learning_rate,
    monitor_metrics=['Micro-F1', 'Macro-F1', 'P@1', 'P@3', 'P@5']
)

# Initialize a trainer.
trainer = init_trainer(checkpoint_dir='runs/NN-example', epochs=15, val_metric='P@5')

# Create data loaders.
loaders = dict()
for split in ['train', 'val', 'test']:
    loaders[split] = get_dataset_loader(
        data=data[split],
        word_dict=word_dict,
        classes=classes,
        device=device,
        max_seq_length=512,
        batch_size=8,
        shuffle=True if split == 'train' else False,
        tokenizer=tokenizer,
        add_special_tokens=add_special_tokens
    )

# Train a model from scratch.
trainer.fit(model, loaders['train'], loaders['val'])

# Test the model.
trainer.test(model, dataloaders=loaders['test'])
