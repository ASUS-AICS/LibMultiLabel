import collections
import logging
import os

import torch
import tqdm
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer

UNK = Vocab.UNK
PAD = '<pad>'

UNK = Vocab.UNK
PAD = '<pad>'

class TextDataset(Dataset):
    """Class for text dataset"""

    def __init__(self, data, word_dict, classes, max_seq_length):
        self.data = data
        self.word_dict = word_dict
        self.classes = classes
        self.max_seq_length = max_seq_length
        self.num_classes = len(self.classes)
        self.label_binarizer = MultiLabelBinarizer().fit([classes])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        return {
            'text': torch.LongTensor([self.word_dict[word] for word in data['text']][:self.max_seq_length]),
            'label': torch.FloatTensor(self.label_binarizer.transform([data['label']])[0]),
            'index': data.get('index', 0)
        }


def generate_batch(data_batch):
    text_list = [data['text'] for data in data_batch]
    label_list = [data['label'] for data in data_batch]
    return {
        'index': [data['index'] for data in data_batch],
        'text': pad_sequence(text_list, batch_first=True),
        'label': torch.stack(label_list)
    }


def get_dataset_loader(config, data, word_dict, classes, shuffle=False, train=True):
    dataset = TextDataset(data, word_dict, classes, config.max_seq_length)
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size if train else config.eval_batch_size,
        shuffle=shuffle,
        num_workers=config.data_workers,
        collate_fn=generate_batch,
        pin_memory='cuda' in config.device.type,
    )
    return dataset_loader


def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')
    return [t.lower() for t in tokenizer.tokenize(text) if not t.isnumeric()]


def _load_raw_data(config, path, is_test=False):
    logging.info(f'Load data from {path}.')
    data = pd.read_csv(path, sep='\t', names=['label', 'text'],
                       converters={'label': lambda s: s.split(),
                                   'text': tokenize})
    data = data.reset_index().to_dict('records')
    if not is_test:
        data = [d for d in data if len(d['label']) > 0]
    if config.fixed_length:
        pad_seq = [PAD] * config.max_seq_length
        for i in range(len(data)):
            pad_len = config.max_seq_length - len(data[i]['text'])
            data[i]['text'] += pad_seq[:pad_len]
    return data


def load_datasets(config):
    datasets = {}
    test_path = config.test_path or os.path.join(config.data_dir, 'test.txt')
    if config.eval:
        datasets['test'] = _load_raw_data(config, test_path, is_test=True)
    else:
        if os.path.exists(test_path):
            datasets['test'] = _load_raw_data(config, test_path, is_test=True)
        train_path = config.train_path or os.path.join(config.data_dir, 'train.txt')
        datasets['train'] = _load_raw_data(config, train_path)
        val_path = config.val_path or os.path.join(config.data_dir, 'valid.txt')
        if os.path.exists(val_path):
            datasets['val'] = _load_raw_data(config, val_path)
        else:
            datasets['train'], datasets['val'] = train_test_split(
                datasets['train'], test_size=config.val_size, random_state=42)

    msg = ' / '.join(f'{k}: {len(v)}' for k, v in datasets.items())
    logging.info(f'Finish loading dataset ({msg})')
    return datasets


def load_or_build_text_dict(config, dataset):
    if config.vocab_file:
        logging.info(f'Load vocab from {config.vocab_file}')
        with open(config.vocab_file, 'r') as fp:
            vocab_list = [PAD] + [vocab.strip() for vocab in fp.readlines()]
        vocabs = Vocab(collections.Counter(vocab_list), specials=[UNK],
                       min_freq=1, specials_first=False)
    else:
        counter = collections.Counter()
        for data in dataset:
            unique_tokens = set(data['text'])
            counter.update(unique_tokens)
        vocabs = Vocab(counter, specials=[PAD, UNK],
                       min_freq=config.min_vocab_freq)
    logging.info(f'Read {len(vocabs)} vocabularies.')
    return vocabs


def load_or_build_label(config, datasets):
    if config.label_file:
        logging.info('Load labels from {config.label_file}')
        with open(config.label_file, 'r') as fp:
            classes = sorted([s.strip() for s in fp.readlines()])
    else:
        classes = set()
        for dataset in datasets.values():
            for d in tqdm.tqdm(dataset):
                classes.update(d['label'])
        classes = sorted(classes)
    return classes


def get_embedding_weights_from_file(word_dict, embed_file):
    """If there is an embedding file, load pretrained word embedding.
    Otherwise, assign a zero vector to that word.
    """

    with open(embed_file) as f:
        word_vectors = f.readlines()

    embed_size = len(word_vectors[0].split())-1
    embedding_weights = [np.zeros(embed_size) for i in range(len(word_dict))]

    vec_counts = 0
    for word_vector in tqdm.tqdm(word_vectors):
        word, vector = word_vector.rstrip().split(' ', 1)
        vector = np.array(vector.split()).astype(np.float)
        vector = vector / float(np.linalg.norm(vector) + 1e-6)
        embedding_weights[word_dict[word]] = vector
        vec_counts += 1

    logging.info(f'loaded {vec_counts}/{len(word_dict)} word embeddings')

    """ Add UNK embedding.
    Attention xml: np.random.uniform(-1.0, 1.0, emb_size)
    CAML: np.random.randn(embed_size)
    TODO. callback
    """
    unk_vector = np.random.randn(embed_size)
    unk_vector = unk_vector / float(np.linalg.norm(unk_vector) + 1e-6)
    embedding_weights[word_dict[word_dict.UNK]] = unk_vector

    return torch.Tensor(embedding_weights)
