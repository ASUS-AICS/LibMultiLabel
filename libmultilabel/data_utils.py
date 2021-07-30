import collections
import copy
import logging
import os

import torch
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab
from tqdm import tqdm

UNK = Vocab.UNK
PAD = '**PAD**'


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
            'label': torch.IntTensor(self.label_binarizer.transform([data['label']])[0]),
        }


def generate_batch(data_batch):
    text_list = [data['text'] for data in data_batch]
    label_list = [data['label'] for data in data_batch]
    return {
        'text': pad_sequence(text_list, batch_first=True),
        'label': torch.stack(label_list)
    }


def get_dataset_loader(
    data,
    word_dict,
    classes,
    device,
    max_seq_length=500,
    batch_size=1,
    shuffle=False,
    data_workers=4
):
    dataset = TextDataset(data, word_dict, classes, max_seq_length)
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=data_workers,
        collate_fn=generate_batch,
        pin_memory='cuda' in device.type,
    )
    return dataset_loader


def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')
    return [t.lower() for t in tokenizer.tokenize(text) if not t.isnumeric()]


def _load_raw_data(path, is_test=False):
    logging.info(f'Load data from {path}.')
    data = pd.read_csv(path, sep='\t', names=['label', 'text'],
                       converters={'label': lambda s: s.split(),
                                   'text': tokenize})
    data = data.reset_index().to_dict('records')
    if not is_test:
        data = [d for d in data if len(d['label']) > 0]
    return data


def load_datasets(
    data_dir,
    train_path=None,
    test_path=None,
    val_path=None,
    val_size=0.2,
    is_eval=False
):
    datasets = {}
    test_path = test_path or os.path.join(data_dir, 'test.txt')
    if is_eval:
        datasets['test'] = _load_raw_data(test_path, is_test=True)
    else:
        if os.path.exists(test_path):
            datasets['test'] = _load_raw_data(test_path, is_test=True)
        train_path = train_path or os.path.join(data_dir, 'train.txt')
        datasets['train'] = _load_raw_data(train_path)
        val_path = val_path or os.path.join(data_dir, 'valid.txt')
        if os.path.exists(val_path):
            datasets['val'] = _load_raw_data(val_path)
        else:
            datasets['train'], datasets['val'] = train_test_split(
                datasets['train'], test_size=val_size, random_state=42)

    msg = ' / '.join(f'{k}: {len(v)}' for k, v in datasets.items())
    logging.info(f'Finish loading dataset ({msg})')
    return datasets


def load_or_build_text_dict(
    dataset,
    vocab_file=None,
    min_vocab_freq=1,
    embed_file=None,
    embed_cache_dir=None,
    silent=False,
    normalize_embed=False
):
    """Build or load the vocabulary from the training dataset or the predefined `vocab_file`.
    The pretrained embedding can be either from a self-defined `embed_file` or from one of
    the vectors defined in torchtext `vectors` (https://pytorch.org/text/0.9.0/vocab.html#torchtext.vocab.Vocab.load_vectors).

    Args:
        dataset (list): List of training instances with index, label, and tokenized text.
        vocab_file (str, optional): Path to a file holding vocabuaries. Defaults to None.
        min_vocab_freq (int, optional): The minimum frequency needed to include a token in the vocabulary. Defaults to 1.
        embed_file (str): Path to a file holding pre-trained embeddings.
        embed_cache_dir (str, optional): Path to a directory for storing cached embeddings. Defaults to None.
        silent (bool, optional): Enable silent mode. Defaults to False.
        normalize_embed (bool, optional): Whether the embeddings of each word is normalized to a unit vector.
            Defaults to False. We follow the normalization method here:
            https://github.com/jamesmullenbach/caml-mimic/blob/44a47455070d3d5c6ee69fb5305e32caec104960/dataproc/extract_wvs.py#L60.

    Returns:
        torchtext.vocab.Vocab: A vocab object which maps tokens to indices.
    """
    if vocab_file:
        logging.info(f'Load vocab from {vocab_file}')
        with open(vocab_file, 'r') as fp:
            vocab_list = [PAD] + [vocab.strip() for vocab in fp.readlines()]
        vocabs = Vocab(collections.Counter(vocab_list), specials=[UNK],
                       min_freq=1, specials_first=False)  # specials_first=False to keep PAD index 0
    else:
        counter = collections.Counter()
        for data in dataset:
            unique_tokens = set(data['text'])
            counter.update(unique_tokens)
        vocabs = Vocab(counter, specials=[PAD, UNK],
                       min_freq=min_vocab_freq)
    logging.info(f'Read {len(vocabs)} vocabularies.')

    if os.path.exists(embed_file):
        logging.info(f'Load pretrained embedding from file: {embed_file}.')
        embedding_weights = get_embedding_weights_from_file(vocabs, embed_file, silent)
        vocabs.set_vectors(vocabs.stoi, embedding_weights, dim=embedding_weights.shape[1])
    elif not embed_file.isdigit():
        logging.info(f'Load pretrained embedding from torchtext.')
        vocabs.load_vectors(embed_file, cache=embed_cache_dir)
    else:
        raise NotImplementedError

    if normalize_embed:
        embedding_weights = copy.deepcopy(vocabs.vectors.detach().cpu().numpy())
        for i, vector in enumerate(embedding_weights):
            embedding_weights[i] = vector/float(np.linalg.norm(vector) + 1e-6)
        embedding_weights = torch.Tensor(embedding_weights)
        vocabs.set_vectors(vocabs.stoi, embedding_weights, dim=embedding_weights.shape[1])

    return vocabs


def load_or_build_label(datasets, label_file=None, silent=False):
    if label_file:
        logging.info('Load labels from {label_file}')
        with open(label_file, 'r') as fp:
            classes = sorted([s.strip() for s in fp.readlines()])
    else:
        classes = set()
        for dataset in datasets.values():
            for d in tqdm(dataset, disable=silent):
                classes.update(d['label'])
        classes = sorted(classes)
    return classes


def get_embedding_weights_from_file(word_dict, embed_file, silent=False): # , normalize=False):
    """If there is an embedding file, load pretrained word embedding.
    Otherwise, assign a zero vector to that word.

    Args:
        word_dict (torchtext.vocab.Vocab): A vocab object which maps tokens to indices.
        embed_file (str): Path to a file holding pre-trained embeddings.
        silent (bool, optional): Enable silent mode. Defaults to False.

    Returns:
        torch.Tensor: Embedding weights (vocab_size, embed_size)
    """
    with open(embed_file) as f:
        word_vectors = f.readlines()

    embed_size = len(word_vectors[0].split())-1
    embedding_weights = [np.zeros(embed_size) for i in range(len(word_dict))]

    """ Add UNK embedding.
    Attention xml: np.random.uniform(-1.0, 1.0, embed_size)
    CAML: np.random.randn(embed_size)
    """
    unk_vector = np.random.randn(embed_size)
    embedding_weights[word_dict[word_dict.UNK]] = unk_vector

    # Load pretrained word embedding
    vec_counts = 0
    for word_vector in tqdm(word_vectors, disable=silent):
        word, vector = word_vector.rstrip().split(' ', 1)
        vector = np.array(vector.split()).astype(np.float)
        embedding_weights[word_dict[word]] = vector
        vec_counts += 1

    logging.info(f'loaded {vec_counts}/{len(word_dict)} word embeddings')

    return torch.Tensor(embedding_weights)
