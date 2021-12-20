import collections
import gc
import logging
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
    length_list = [len(data['text']) for data in data_batch]
    return {
        'text': pad_sequence(text_list, batch_first=True),
        'label': torch.stack(label_list),
        'length': length_list
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
    """Create a pytorch DataLoader.

    Args:
        data (list): List of training instances with index, label, and tokenized text.
        word_dict (torchtext.vocab.Vocab): A vocab object which maps tokens to indices.
        classes (list): List of labels.
        device (torch.device): One of cuda or cpu.
        max_seq_length (int, optional): The maximum number of tokens of a sample. Defaults to 500.
        batch_size (int, optional): Size of training batches. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle training data before each epoch. Defaults to False.
        data_workers (int, optional): Use multi-cpu core for data pre-processing. Defaults to 4.

    Returns:
        torch.utils.data.DataLoader: A pytorch DataLoader.
    """
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
    data = pd.read_csv(path, sep='\t', header=None, error_bad_lines=False, warn_bad_lines=True).fillna('')
    if data.shape[1] == 2:
        data.columns = ['label', 'text']
        data = data.reset_index()
    elif data.shape[1] == 3:
        data.columns = ['index', 'label', 'text']
    else:
        raise ValueError(f'Expected 2 or 3 columns, got {data.shape[1]}.')
    data['label'] = data['label'].map(lambda s: s.split())
    data['text'] = data['text'].map(tokenize)
    data = data.to_dict('records')
    if not is_test:
        data = [d for d in data if len(d['label']) > 0]
    return data


def load_datasets(
    train_path=None,
    test_path=None,
    val_path=None,
    val_size=0.2,
    merge_train_val=False
):
    """Load data from the specified data paths (i.e., `train_path`, `test_path`, and `val_path`).
    If `valid.txt` does not exist but `val_size` > 0, the validation set will be split from the training dataset.

    Args:
        train_path (str, optional): Path to training data.
        test_path (str, optional): Path to test data.
        val_path (str, optional): Path to validation data.
        val_size (float, optional): Training-validation split: a ratio in [0, 1] or an integer for the size of the validation set. Defaults to 0.2.
        merge_train_val (bool, optional): Whether to merge the training and validation data.
            Defaults to False.

    Returns:
        dict: A dictionary of datasets.
    """
    assert train_path or test_path, "At least one of `train_path` and `test_path` must be specified."

    datasets = {}
    if train_path is not None and os.path.exists(train_path):
        datasets['train'] = _load_raw_data(train_path)

    if val_path is not None and os.path.exists(val_path):
        datasets['val'] = _load_raw_data(val_path)
    elif val_size > 0:
        datasets['train'], datasets['val'] = train_test_split(
            datasets['train'], test_size=val_size, random_state=42)

    if test_path is not None and os.path.exists(test_path):
        datasets['test'] = _load_raw_data(test_path, is_test=True)

    if merge_train_val:
        datasets['train'] = datasets['train'] + datasets['val']
        for i in range(len(datasets['train'])):
            datasets['train'][i]['index'] = i
        del datasets['val']
        gc.collect()

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
        normalize_embed (bool, optional): Whether the embeddings of each word is normalized to a unit vector. Defaults to False.

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
        dim = torch.as_tensor(embedding_weights).shape[1]
        vocabs.set_vectors(vocabs.stoi, torch.Tensor(embedding_weights), dim=dim)
    elif not embed_file.isdigit():
        logging.info(f'Load pretrained embedding from torchtext.')
        vocabs.load_vectors(embed_file, cache=embed_cache_dir)
    else:
        raise NotImplementedError

    if normalize_embed:
        # vocabs.vectors is a torch.FloatTensor from the result of vocabs.set_vectors earlier.
        # To have better precision for calculating the normalization, we use the original
        # embedding_weights, a torch.DobleTensor, if it is available.
        embedding_weights = embedding_weights if embedding_weights else vocabs.vectors.numpy()
        for i, vector in enumerate(embedding_weights):
            # We use the constant 1e-6 by following https://github.com/jamesmullenbach/caml-mimic/blob/44a47455070d3d5c6ee69fb5305e32caec104960/dataproc/extract_wvs.py#L60
            # for an internal experiment of reproducing their results.
            embedding_weights[i] = vector / float(np.linalg.norm(vector) + 1e-6)
        embedding_weights = torch.as_tensor(embedding_weights)
        vocabs.set_vectors(vocabs.stoi, embedding_weights, dim=embedding_weights.shape[1])

    return vocabs


def load_or_build_label(datasets, label_file=None, include_test_labels=False):
    """Generate label set either by the given datasets or a predefined label file.

    Args:
        datasets (dict): A dictionary of datasets. Each dataset contains list of instances
            with index, label, and tokenized text.
        label_file (str, optional): Path to a file holding all labels.
        include_test_labels (bool, optional): Whether to include labels in the test dataset.
            Defaults to True.

    Returns:
        list: A list of labels sorted in alphabetical order.
    """
    if label_file:
        logging.info(f'Load labels from {label_file}.')
        with open(label_file, 'r') as fp:
            classes = sorted([s.strip() for s in fp.readlines()])
    else:
        classes = set()
        for split, data in datasets.items():
            if split == 'test' and not include_test_labels:
                continue
            for instance in data:
                classes.update(instance['label'])
        classes = sorted(classes)
    logging.info(f'Read {len(classes)} labels.')
    return classes


def get_embedding_weights_from_file(word_dict, embed_file, silent=False):
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

    return embedding_weights
