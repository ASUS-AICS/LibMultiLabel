# Step 1. Import the libraries
from libmultilabel.nn.data_utils import load_datasets, load_or_build_label, get_dataset_loader, \
                                        load_or_build_text_dict
from libmultilabel.nn.nn_utils import init_device, init_model, init_trainer, set_seed

# Step 2. Setup device.
set_seed(1337)
device = init_device()  # use gpu by default

# Step 3. Load data from text files.
datasets = load_datasets('data/EUR-Lex/train.txt', 'data/EUR-Lex/test.txt', tokenize_text=True)
classes = load_or_build_label(datasets, include_test_labels=False)

# Step 4. setup data preprocessing function
word_dict, embed_vecs = load_or_build_text_dict(dataset=datasets['train'], embed_file='glove.6B.300d')
tokenizer = None

# Step 5. Initialize a model.
model_name = 'KimCNN'
network_config = {
    'embed_dropout': 0.2,
    'encoder_dropout': 0.2,
    'filter_sizes': [2, 4, 8],
    'num_filter_per_size': 128
}
model = init_model(
    model_name=model_name,
    network_config=network_config,
    classes=classes,
    word_dict=word_dict,
    embed_vecs=embed_vecs,
    learning_rate=0.00005,
    optimizer='adamw',
    weight_decay=0.001,
    monitor_metrics=['Micro-F1', 'Macro-F1', 'P@1', 'P@3', 'P@5']
)

# Step 6. Initialize a trainer.
trainer = init_trainer(checkpoint_dir='runs/NN-example', epochs=15, val_metric='P@5')

# Step 7. Create data loaders.
loaders = dict()
for split in ['train', 'val', 'test']:
    loaders[split] = get_dataset_loader(
        data=datasets[split],
        word_dict=word_dict,
        classes=classes,
        device=device,
        max_seq_length=512,
        batch_size=8,
        shuffle=True if split == 'train' else False,
        tokenizer=tokenizer
    )

# Step 8-1. Train a model from scratch.
trainer.fit(model, loaders['train'], loaders['val'])

# Step 8-2. Test the model.
trainer.test(model, dataloaders=loaders['test'])
