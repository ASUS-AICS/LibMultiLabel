import warnings
from typing import Any
import pandas as pd

import torch
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator, Vocab, pretrained_aliases


# TODO: why this?
warnings.simplefilter(action='ignore', category=FutureWarning)

UNK = '<unk>'
PAD = '<pad>'


class TokenDataset(Dataset):
    """Amazing docstring about this class"""

    def __init__(self, data: pd.DataFrame,
                 vocab: Vocab,
                 classes: 'list[str]',
                 max_seq_length: int):
        self.data = data
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.label_binarizer = MultiLabelBinarizer().fit([classes])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index]
        input_ids = [self.vocab[word]
                     for word in data['text'][:self.max_seq_length]]
        # TODO: why long and int tensor?
        return {
            'text': torch.LongTensor(input_ids),
            'label': torch.IntTensor(self.label_binarizer.transform([data['label']])[0])
        }


def tokenize(text):
    """Tokenize text into words. Words are non-whitespace characters delimited by whitespace characters.

    Args:
        text (str): Text to tokenize.

    Returns:
        list: A list of words.
    """
    tokenizer = RegexpTokenizer(r'\w+')
    # TODO: why default lower?
    return [t.lower() for t in tokenizer.tokenize(text) if not t.isnumeric()]


def collate_fn(data_batch):
    text_list = [data['text'] for data in data_batch]
    label_list = [data['label'] for data in data_batch]
    # TODO: where is length_list used?
    length_list = [len(data['text']) for data in data_batch]
    return {
        'text': pad_sequence(text_list, batch_first=True),
        'label': torch.stack(label_list),
        'length': torch.IntTensor(length_list)
    }


def build_vocabulary(dataset: pd.DataFrame, min_freq: int = 1) -> Vocab:
    vocab_list = [set(text) for text in dataset['text']]
    vocab = build_vocab_from_iterator(vocab_list, min_freq=min_freq,
                                      specials=[PAD, UNK])

    vocab.set_default_index(vocab[UNK])
    return vocab


def load_vocabulary(vocab_file: str) -> Vocab:
    with open(vocab_file, 'r') as fp:
        vocab_list = [[vocab.strip() for vocab in fp.readlines()]]

    # TODO: the following comment is indecipherable
    # Keep PAD index 0 to align `padding_idx` of
    # class Embedding in libmultilabel.nn.networks.modules.
    vocab = build_vocab_from_iterator(vocab_list, min_freq=1,
                                      specials=[PAD, UNK])

    vocab.set_default_index(vocab[UNK])
    return vocab


def load_embedding_weights(vocab: Vocab, name: str, cache_dir: str, normalize: bool):
    # TODO: what progress/info should be printed here (if any)?
    use_torchtext = name in pretrained_aliases
    if use_torchtext:
        vector_dict = pretrained_aliases[name](cache=cache_dir)
        embed_size = vector_dict.dim
    else:
        vector_dict = {}
        with open(name) as word_vectors:
            for word_vector in word_vectors:
                word, vector = word_vector.rstrip().split(' ', 1)
                vector = torch.Tensor(list(map(float, vector.split())))
                vector_dict[word] = vector
        embed_size = next(iter(vector_dict.values())).shape[0]

    embedding_weights = torch.zeros(len(vocab), embed_size)

    if not use_torchtext:
        # Add UNK embedding
        # AttentionXML: np.random.uniform(-1.0, 1.0, embed_size)
        # CAML: np.random.randn(embed_size)
        unk_vector = torch.randn(embed_size)
        embedding_weights[vocab[UNK]] = unk_vector

    # drop embeddings not in vocabulary
    vec_counts = 0
    for word in vocab.get_itos():
        # torchtext Vectors returns zero vectors on unknown words
        # TODO: why do we have differing behaviour here??
        if use_torchtext or word in vector_dict:
            embedding_weights[vocab[word]] = vector_dict[word]
            vec_counts += 1

    if normalize:
        # To have better precision for calculating the normalization, we convert the original
        # embedding_weights from a torch.FloatTensor to a torch.DoubleTensor.
        # After the normalization, we will convert the embedding_weights back to a torch.FloatTensor.
        embedding_weights = embedding_weights.double()
        for i, vector in enumerate(embedding_weights):
            # We use the constant 1e-6 by following https://github.com/jamesmullenbach/caml-mimic/blob/44a47455070d3d5c6ee69fb5305e32caec104960/dataproc/extract_wvs.py#L60
            # for an internal experiment of reproducing their results.
            # TODO: should we be using a constant to reproduce caml for all our use cases?
            embedding_weights[i] = vector / \
                float(torch.linalg.norm(vector) + 1e-6)
        embedding_weights = embedding_weights.float()

    return embedding_weights
