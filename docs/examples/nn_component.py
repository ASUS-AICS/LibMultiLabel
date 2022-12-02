import json

NN_quickstart = dict()
NN_quickstart["HEAD"] = """
from libmultilabel.nn.data_utils import *
from libmultilabel.nn.nn_utils import *
from transformers import AutoTokenizer

set_seed(1337)
device = init_device()  # use gpu by default
"""
NN_quickstart["load_data_set_KimCNN"] = """
data_sets = load_datasets('data/rcv1/train.txt', 'data/rcv1/test.txt', tokenize_text=True)
"""
NN_quickstart["load_data_set_BERT"] = """
data_sets = load_datasets('data/rcv1/train.txt', 'data/rcv1/test.txt', tokenize_text=False)
"""
NN_quickstart["build_label"] = """
classes = load_or_build_label(data_sets)
"""
NN_quickstart["KimCNN_part"] = """
word_dict, embed_vecs = load_or_build_text_dict(dataset=data_sets['train'], embed_file='glove.6B.300d')
tokenizer = None
add_special_tokens = False

model_name = 'KimCNN'
network_config = {
    'embed_dropout': 0.2,
    'encoder_dropout': 0.2,
    'filter_sizes': [2, 4, 8],
    'num_filter_per_size': 128
}
learning_rate = 0.0003
"""
NN_quickstart["BERT_part"] = """
word_dict, embed_vecs = None, None 
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
add_special_tokens = True

model_name='BERT'
network_config = {
    'dropout': 0.1,
    'lm_weight': 'bert-base-uncased',
}
learning_rate = 0.00003
"""
NN_quickstart["TAIL"] = """
model = init_model(
    model_name=model_name,
    network_config=network_config,
    classes=classes,
    word_dict=word_dict,
    embed_vecs=embed_vecs,
    learning_rate=learning_rate,
    monitor_metrics=['Micro-F1', 'Macro-F1', 'P@1', 'P@3', 'P@5']
)

trainer = init_trainer(checkpoint_dir='runs/NN-example', epochs=15, val_metric='P@5')

loaders = dict()
for split in ['train', 'val', 'test']:
    loaders[split] = get_dataset_loader(
        data=data_sets[split],
        word_dict=word_dict,
        classes=classes,
        device=device,
        max_seq_length=512,
        batch_size=8,
        shuffle=True if split == 'train' else False,
        tokenizer=tokenizer,
        add_special_tokens=add_special_tokens
    )

trainer.fit(model, loaders['train'], loaders['val'])
trainer.test(model, dataloaders=loaders['test'])
"""

with open("nn_component.json", "w") as F:
    json.dump(NN_quickstart, F)
