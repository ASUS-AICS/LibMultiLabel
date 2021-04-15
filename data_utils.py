import collections
import copy
import itertools
import os
import pickle
import random

import torch
import tqdm
import numpy as np
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer

from utils import log


class TextDataset(Dataset):
    """Class for text dataset"""

    def __init__(self, data, word_dict, classes, max_seq_length):
        self.data = data
        self.word_dict = word_dict
        self.classes = classes
        self.max_seq_length = max_seq_length
        self.num_class = len(self.classes)
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


def get_dataset_loader(config, dataset, shuffle=False, train=True):
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size if train else config.eval_batch_size,
        shuffle=shuffle,
        num_workers=config.data_workers,
        collate_fn=generate_batch,
        pin_memory='cuda' in config.device.type,
    )
    return dataset_loader


def _load_raw_data(config, data_dir):
    raw_data_cache = config.get('raw_data_cache')
    if raw_data_cache and os.path.exists(raw_data_cache):
        log.info(f'Load existing raw data cache from {raw_data_cache}.')
        with open(raw_data_cache, 'rb') as fp:
            dataset = pickle.load(fp)
        if 'labels' in next(iter(dataset.values())):
            for split in dataset:
                dataset[split]['label'] = dataset[split].pop('labels')
        return dataset

    log.info(f'Load data from train_texts.txt, train_labels.txt, test_texts.txt, and text_labels.txt.')
    dataset = {k: {} for k in ['train', 'test']}
    for split in ['train', 'test']:
        with open(os.path.join(data_dir, f'{split}_texts.txt')) as f:
            texts = f.readlines()
            dataset[split]['text'] = [text for text in texts]
        with open(os.path.join(data_dir, f'{split}_labels.txt')) as f:
            labels = f.readlines()
            dataset[split]['label'] = [label.split() for label in labels]
        dataset[split]['index'] = list(range(len(dataset[split]['text'])))
    return dataset


@log.enter('load_dataset')
def load_dataset(config):
    """Preparing TextDataset from raw data."""
    data_dir = os.path.join(config['data_dir'], config['data_name'])
    os.makedirs(data_dir, exist_ok=True)

    # Load from cache if exists, otherwise load raw data and tokenize
    cache_path = os.path.join(data_dir, 'cache.pkl')
    if os.path.exists(cache_path):
        log.info(f'Load existing cache from {cache_path}.')
        with open(cache_path, 'rb') as fp:
            datasets = pickle.load(fp)
    else:
        datasets = _load_raw_data(config, data_dir)
        for split in datasets.keys():
            datasets[split] = _preprocess_on_split(datasets[split])
        with open(cache_path, 'wb') as fp:
            pickle.dump(datasets, fp)

    map_path = os.path.join(data_dir, f'vocab_label_map_{config["min_vocab_freq"]}.pkl')
    map_path = config['vocab_label_map'] or map_path
    if map_path and os.path.exists(map_path):
        log.info(f'Load existing Vocab and classes from {map_path}.')
        with open(map_path, 'rb') as fp:
            text_dict, classes = pickle.load(fp)
    else:
        classes = set()
        for dataset in datasets.values():
            for d in tqdm.tqdm(dataset):
                classes.update(d['label'])
        classes = sorted(classes)

        text_dict = build_text_dict(datasets['train'], config['min_vocab_freq'], config['vocab_file'])
        with open(map_path, 'wb') as fp:
            pickle.dump((text_dict, classes), fp)
        log.info(f'Save {map_path}')

    if 'dev' not in datasets:
        dev_size = config['dev_size'] if config['dev_size'].is_integer() else int(config['dev_size'] * len(datasets['train']))
        train_size = len(datasets['train']) - dev_size
        datasets['train'], datasets['dev'] = torch.utils.data.random_split(datasets['train'], [train_size, dev_size], generator=torch.Generator().manual_seed(42))

    for split in datasets.keys():
        datasets[split] = TextDataset(datasets[split], text_dict, classes, config['max_seq_length'])

    log.info(f"Finish loading dataset (train: {len(datasets['train'])} / test: {len(datasets['test'])} / dev: {len(datasets['dev'])})")
    return datasets


def _get_tokenizer():
    def caml_tokenizer(text):
        tokenizer = RegexpTokenizer(r'\w+')
        return [t.lower() for t in tokenizer.tokenize(text) if not t.isnumeric()]

    # attention xml
    # [token.lower() if token != sep else token for token in word_tokenize(sentence)
    #     if len(re.sub(r'[^\w]', '', token)) > 0]
    return caml_tokenizer


def _preprocess_on_split(dataset):
    # split text: https://pytorch.org/text/_modules/torchtext/data/utils.html
    # tokenizer = get_tokenizer(tokenizer=None)
    tokenizer = _get_tokenizer()
    dataset['text'] = [tokenizer(text) for text in dataset['text']]
    dataset = [dict(zip(dataset, v)) for v in zip(*dataset.values())]
    dataset = [d for d in dataset if len(d['label']) > 0]

    return dataset


def build_text_dict(examples, min_vocab_freq, vocab_file=None):
    if vocab_file:
        log.info(f'Load vocab from {vocab_file}')
        with open(vocab_file, 'r') as f:
            vocab_list = ['**PAD**'] + [vocab.strip() for vocab in f.readlines()]
        vocabs = Vocab(collections.Counter(vocab_list), specials=['<unk>'], min_freq=1, specials_first=False)
    else:
        counter = collections.Counter()
        for example in examples:
            unique_tokens = set(example['text'])
            counter.update(unique_tokens)
        vocabs = Vocab(counter, specials=['<pad>', '<unk>'], min_freq=min_vocab_freq)

    log.info(f'Read {len(vocabs)} vocabularies.')
    return vocabs


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

    log.info(f'loaded {vec_counts}/{len(word_dict)} word embeddings')

    """ Add UNK embedding.
    Attention xml: np.random.uniform(-1.0, 1.0, emb_size)
    CAML: np.random.randn(embed_size)
    TODO. callback
    """
    unk_vector = np.random.randn(embed_size)
    unk_vector = unk_vector / float(np.linalg.norm(unk_vector) + 1e-6)
    embedding_weights[word_dict[word_dict.UNK]] = unk_vector

    return torch.Tensor(embedding_weights)
