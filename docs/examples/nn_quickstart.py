from libmultilabel.nn.data_utils import load_datasets, load_or_build_label, \
                                        load_or_build_text_dict, get_dataset_loader
from libmultilabel.nn.nn_utils import init_device, init_model, init_trainer

# Step 0. Setup device.
device = init_device()  # use gpu by default

# Step 1. Load data from text files.
datasets = load_datasets('data/rcv1/train.txt',
                         'data/rcv1/test.txt')
classes = load_or_build_label(datasets)
word_dict, embed_vecs = load_or_build_text_dict(dataset=datasets['train'],
                                                embed_file='glove.6B.300d')

# Step 2. Initialize a model.
network_config = {
    'embed_dropout': 0.2,
    'encoder_dropout': 0.2,
    'filter_sizes': [2, 4, 8],
    'num_filter_per_size': 128
}
model = init_model(model_name='KimCNN',
                   network_config=network_config,
                   classes=classes,
                   word_dict=word_dict,
                   embed_vecs=embed_vecs,
                   monitor_metrics=['Macro-F1', 'Micro-F1', 'P@1', 'P@3', 'P@5'])

# Step 3. Initialize a trainer.
trainer = init_trainer(checkpoint_dir='runs/rcv1-KimCNN-example',
                       epochs=50,
                       val_metric='Macro-F1')

# Step 4. Create data loaders.
loaders = dict()
for split in ['train', 'val', 'test']:
    loaders[split] = get_dataset_loader(data=datasets[split],
                                        word_dict=word_dict,
                                        classes=classes,
                                        device=device,
                                        batch_size=64)

# Step 5-1. Train a model from scratch.
trainer.fit(model, loaders['train'], loaders['val'])

# Step 5-2. Test the model.
trainer.test(model, dataloaders=loaders['test'])
