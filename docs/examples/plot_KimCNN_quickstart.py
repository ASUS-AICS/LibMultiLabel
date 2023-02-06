"""
KimCNN Model for Multi-label Classification
===========================================

This step-by-step example shows how to train and test a KimCNN model via LibMultiLabel.


Import the libraries
----------------------------

Please add the following code to your python3 script.
"""

from libmultilabel.nn.data_utils import *
from libmultilabel.nn.nn_utils import *

######################################################################
# Setup device
# --------------------
# If you need to reproduce the results, please use the function ``set_seed``.
# For example, you will get the same result as you always use the seed ``1337``.
#
# For initial a hardware device, please use ``init_device`` to assign the hardware device that you want to use.

set_seed(1337)
device = init_device()  # use gpu by default

######################################################################
# Load and tokenize data
# ------------------------------------------
#
# To run KimCNN, LibMultiLabel tokenizes documents and uses an embedding vector for each word.
# Thus, ``tokenize_text=True`` is set.
#
# We choose ``glove.6B.300d`` from torchtext as embedding vectors.

datasets = load_datasets('data/rcv1/train.txt', 'data/rcv1/test.txt', tokenize_text=True)
classes = load_or_build_label(datasets)
word_dict, embed_vecs = load_or_build_text_dict(dataset=datasets['train'], embed_file='glove.6B.300d')
tokenizer = None

######################################################################
# Initialize a model
# --------------------------
#
# We consider the following settings for the KimCNN model.

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

######################################################################
# * ``model_name`` leads ``init_model`` function to find a network model.
# * ``network_config`` contains the configurations of a network model.
# * ``classes`` is the label set of the data.
# * ``init_weight``, ``word_dict`` and ``embed_vecs`` are not used on a bert-base model, so we can ignore them.
# * ``moniter_metrics`` includes metrics you would like to track.
#
#
# Initialize a trainer
# ----------------------------
#
# We use the function ``init_trainer`` to initialize a trainer.

trainer = init_trainer(checkpoint_dir='runs/NN-example', epochs=15, val_metric='P@5')

######################################################################
# In this example, ``checkpoint_dir`` is the place we save the best and the last models during the training. Furthermore, we set the number of training loops by ``epochs=15``, and the validation metric by ``val_metric='P@5'``.
#
# Create data loaders
# ---------------------------
#
# In most cases, we do not load a full set due to the hardware limitation.
# Therefore, a data loader can load a batch of samples each time.

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

######################################################################
# This example loads three loaders, and the batch size is set by ``batch_size=8``. Other variables can be checked in `here <../api/nn.html#libmultilabel.nn.data_utils.get_dataset_loader>`_.
#
# Train and test a model
# ------------------------------
#
# The bert model training process can be started via

trainer.fit(model, loaders['train'], loaders['val'])

######################################################################
# After the training process is finished, we can then run the test process by

trainer.test(model, dataloaders=loaders['test'])

######################################################################
# The test results should be similar to::
#
#  {
#      'Macro-F1': 0.48948464335831743,
#      'Micro-F1': 0.7769773602485657,
#      'P@1':      0.9471677541732788,
#      'P@3':      0.7772253751754761,
#      'P@5':      0.5449321269989014,
#  }

