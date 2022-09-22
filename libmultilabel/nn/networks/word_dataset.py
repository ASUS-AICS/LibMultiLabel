from tqdm import tqdm
from torchtext.vocab import build_vocab_from_iterator, pretrained_aliases
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import torch
import gc
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


UNK = '<unk>'
PAD = '<pad>'


class WordDataset(Dataset):
    """Amazing docstring about this class"""

    def __init__(self, data, word_dict, classes, max_seq_length, tokenizer=None, add_special_tokens=False):
        self.data = {
            **data,
            'text': data['text'].map(tokenize),
        }
        self.word_dict = word_dict
        self.classes = classes
        self.max_seq_length = max_seq_length
        self.num_classes = len(self.classes)
        self.label_binarizer = MultiLabelBinarizer().fit([classes])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        input_ids = [self.word_dict[word] for word in data['text']]
        return {
            'text': torch.LongTensor(input_ids[:self.max_seq_length]),
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
    the vectors defined in torchtext.vocab.pretrained_aliases
    (https://github.com/pytorch/text/blob/main/torchtext/vocab/vectors.py).

    Args:
        dataset (list): List of training instances with index, label, and tokenized text.
        vocab_file (str, optional): Path to a file holding vocabuaries. Defaults to None.
        min_vocab_freq (int, optional): The minimum frequency needed to include a token in the vocabulary. Defaults to 1.
        embed_file (str): Path to a file holding pre-trained embeddings.
        embed_cache_dir (str, optional): Path to a directory for storing cached embeddings. Defaults to None.
        silent (bool, optional): Enable silent mode. Defaults to False.
        normalize_embed (bool, optional): Whether the embeddings of each word is normalized to a unit vector. Defaults to False.

    Returns:
        tuple[torchtext.vocab.Vocab, torch.Tensor]: A vocab object which maps tokens to indices and the pre-trained word vectors of shape (vocab_size, embed_dim).
    """
    if vocab_file:
        logging.info(f'Load vocab from {vocab_file}')
        with open(vocab_file, 'r') as fp:
            vocab_list = [[vocab.strip() for vocab in fp.readlines()]]
        # Keep PAD index 0 to align `padding_idx` of
        # class Embedding in libmultilabel.nn.networks.modules.
        vocabs = build_vocab_from_iterator(vocab_list, min_freq=1,
                                           specials=[PAD, UNK])
    else:
        vocab_list = [set(data['text']) for data in dataset]
        vocabs = build_vocab_from_iterator(vocab_list, min_freq=min_vocab_freq,
                                           specials=[PAD, UNK])
    vocabs.set_default_index(vocabs[UNK])
    logging.info(f'Read {len(vocabs)} vocabularies.')

    embedding_weights = get_embedding_weights_from_file(
        vocabs, embed_file, silent, embed_cache_dir)

    if normalize_embed:
        # To have better precision for calculating the normalization, we convert the original
        # embedding_weights from a torch.FloatTensor to a torch.DoubleTensor.
        # After the normalization, we will convert the embedding_weights back to a torch.FloatTensor.
        embedding_weights = embedding_weights.double()
        for i, vector in enumerate(embedding_weights):
            # We use the constant 1e-6 by following https://github.com/jamesmullenbach/caml-mimic/blob/44a47455070d3d5c6ee69fb5305e32caec104960/dataproc/extract_wvs.py#L60
            # for an internal experiment of reproducing their results.
            embedding_weights[i] = vector / \
                float(torch.linalg.norm(vector) + 1e-6)
        embedding_weights = embedding_weights.float()

    return vocabs, embedding_weights


def get_embedding_weights_from_file(word_dict, embed_file, silent=False, cache=None):
    """If the word exists in the embedding file, load the pretrained word embedding.
    Otherwise, assign a zero vector to that word.

    Args:
        word_dict (torchtext.vocab.Vocab): A vocab object which maps tokens to indices.
        embed_file (str): Path to a file holding pre-trained embeddings.
        silent (bool, optional): Enable silent mode. Defaults to False.
        cache (str, optional): Path to a directory for storing cached embeddings. Defaults to None.

    Returns:
        torch.Tensor: Embedding weights (vocab_size, embed_size)
    """
    # Load pretrained word embedding
    load_embedding_from_file = not embed_file in pretrained_aliases
    if load_embedding_from_file:
        logging.info(f'Load pretrained embedding from file: {embed_file}.')
        with open(embed_file) as f:
            word_vectors = f.readlines()
        embed_size = len(word_vectors[0].split())-1
        vector_dict = {}
        for word_vector in tqdm(word_vectors, disable=silent):
            word, vector = word_vector.rstrip().split(' ', 1)
            vector = torch.Tensor(list(map(float, vector.split())))
            vector_dict[word] = vector
    else:
        logging.info(f'Load pretrained embedding from torchtext.')
        # Adapted from https://pytorch.org/text/0.9.0/_modules/torchtext/vocab.html#Vocab.load_vectors.
        if embed_file not in pretrained_aliases:
            raise ValueError(
                "Got embed_file {}, but allowed pretrained "
                "vectors are {}".format(
                    embed_file, list(pretrained_aliases.keys())))
        vector_dict = pretrained_aliases[embed_file](cache=cache)
        embed_size = vector_dict.dim

    embedding_weights = torch.zeros(len(word_dict), embed_size)

    if load_embedding_from_file:
        # Add UNK embedding
        # AttentionXML: np.random.uniform(-1.0, 1.0, embed_size)
        # CAML: np.random.randn(embed_size)
        unk_vector = torch.randn(embed_size)
        embedding_weights[word_dict[UNK]] = unk_vector

    # Store pretrained word embedding
    vec_counts = 0
    for word in word_dict.get_itos():
        # The condition can be used to process the word that does not in the embedding file.
        # Note that torchtext vector object has already dealt with this,
        # so we can directly make a query without addtional handling.
        if (load_embedding_from_file and word in vector_dict) or not load_embedding_from_file:
            embedding_weights[word_dict[word]] = vector_dict[word]
            vec_counts += 1

    logging.info(f'loaded {vec_counts}/{len(word_dict)} word embeddings')

    return embedding_weights
