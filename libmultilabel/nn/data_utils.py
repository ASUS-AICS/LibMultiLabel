from __future__ import annotations

import csv
import gc
import logging
import re
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import transformers
import nltk
from nltk.tokenize import RegexpTokenizer, word_tokenize
from numpy import ndarray
from scipy.sparse import csr_matrix, issparse
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator, pretrained_aliases, Vocab, Vectors
from gensim.models import KeyedVectors
from tqdm import tqdm

from ..common_utils import GLOBAL_RANK

transformers.logging.set_verbosity_error()
warnings.simplefilter(action="ignore", category=FutureWarning)

# selection of UNK: https://groups.google.com/g/globalvectors/c/9w8ZADXJclA/m/hRdn4prm-XUJ
UNK = "<unk>"
PAD = "<pad>"


class TextDataset(Dataset):
    """Class for text dataset.

    Args:
        data (list[dict]): List of instances with index, label, and text.
        classes (list): List of labels.
        max_seq_length (int, optional): The maximum number of tokens of a sample.
        add_special_tokens (bool, optional): Whether to add the special tokens. Defaults to True.
        tokenizer (transformers.PreTrainedTokenizerBase, optional): HuggingFace's tokenizer of
            the transformer-based pretrained language model. Defaults to None.
        word_dict (torchtext.vocab.Vocab, optional): A vocab object for word tokenizer to
            map tokens to indices. Defaults to None.
    """

    def __init__(
        self,
        data,
        classes,
        max_seq_length,
        add_special_tokens=True,
        *,
        tokenizer=None,
        word_dict=None,
    ):
        self.data = data
        self.classes = classes
        self.max_seq_length = max_seq_length
        self.word_dict = word_dict
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

        self.num_classes = len(self.classes)
        self.label_binarizer = MultiLabelBinarizer().fit([classes])

        if not isinstance(self.word_dict, Vocab) ^ isinstance(self.tokenizer, transformers.PreTrainedTokenizerBase):
            raise ValueError("Please specify exactly one of word_dict or tokenizer")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        if self.tokenizer is not None:  # transformers tokenizer
            if self.add_special_tokens:  # tentatively hard code
                input_ids = self.tokenizer.encode(
                    data["text"], padding="max_length", max_length=self.max_seq_length, truncation=True
                )
            else:
                input_ids = self.tokenizer.encode(data["text"], add_special_tokens=False)
        else:
            input_ids = [self.word_dict[word] for word in data["text"]]
        return {
            "text": torch.LongTensor(input_ids[: self.max_seq_length]),
            "label": torch.IntTensor(self.label_binarizer.transform([data["label"]])[0]),
        }


class MultiLabelDataset(Dataset):
    """Basic class for multi-label dataset."""

    def __init__(
        self,
        x: list[list[int]],
        y: csr_matrix | ndarray | None = None,
    ):
        """General dataset class for multi-label dataset.

        Args:
            x: text.
            y: labels.
        """
        if y is not None:
            assert len(x) == y.shape[0], "Size mismatch between x and y"
        self.x = x
        self.y = y

    def __getitem__(self, idx: int) -> tuple[Sequence, ndarray] | tuple[Sequence]:
        x = self.x[idx]

        # train/valid/test
        if self.y is not None:
            if issparse(self.y):
                y = self.y[idx].toarray().squeeze(0).astype(np.float32)
            else:
                y = self.y[idx].astype(np.float32)
            return x, y
        # predict
        return x

    def __len__(self):
        return len(self.x)


class PLTDataset(MultiLabelDataset):
    """Dataset class for AttentionXML."""

    def __init__(
        self,
        x,
        y: csr_matrix | ndarray | None = None,
        *,
        num_nodes: int,
        mapping: ndarray,
        node_label: ndarray | Tensor,
        node_score: ndarray | Tensor | None = None,
    ):
        """Dataset for FastAttentionXML.
        ~ means variable length.

        Args:
            x: text
            y: labels
            num_nodes: number of nodes at the current level.
            mapping: [[0,..., 7], [8,..., 15], ...]. shape: (len(nodes), ~cluster_size). parent nodes to child nodes.
                Cluster size will only vary at the last level.
            node_label: [[7, 1, 128, 6], [21, 85, 64, 103], ...]. shape: (len(x), top_k). numbers are predicted nodes
                from last level.
            node_score: corresponding scores. shape: (len(x), top_k)
        """
        super().__init__(x, y)
        self.num_nodes = num_nodes
        self.mapping = mapping
        self.node_label = node_label
        self.node_score = node_score
        self.candidate_scores = None

        # candidate are positive nodes at the current level. shape: (len(x), ~cluster_size * top_k)
        # look like [[0, 1, 2, 4, 5, 18, 19,...], ...]
        prog = tqdm(self.node_label, leave=False, desc="Candidates") if GLOBAL_RANK == 0 else self.node_label
        self.candidates = [np.concatenate(self.mapping[labels]) for labels in prog]
        if self.node_score is not None:
            # candidate_scores are corresponding scores for candidates and
            # look like [[0.1, 0.1, 0.1, 0.4, 0.4, 0.5, 0.5,...], ...]. shape: (len(x), ~cluster_size * top_k)
            # notice how scores repeat for each cluster.
            self.candidate_scores = [
                np.repeat(scores, [len(i) for i in self.mapping[labels]])
                for labels, scores in zip(self.node_label, self.node_score)
            ]

        # top_k * n (n <= cluster_size). number of maximum possible number candidates at the current level.
        self.num_candidates = self.node_label.shape[1] * max(len(node) for node in self.mapping)

    def __getitem__(self, idx: int):
        x = self.x[idx]
        candidates = np.asarray(self.candidates[idx], dtype=np.int64)

        # train/valid/test
        if self.y is not None:
            # squeezing is necessary here because csr_matrix.toarray() always returns a 2d array
            # e.g., np.ndarray([[0, 1, 2]])
            y = self.y[idx].toarray().squeeze(0).astype(np.float32)

            # train
            if self.candidate_scores is None:
                # randomly select nodes as candidates when less than required
                if len(candidates) < self.num_candidates:
                    sample = np.random.randint(self.num_nodes, size=self.num_candidates - len(candidates))
                    candidates = np.concatenate([candidates, sample])
                # randomly select a subset of candidates when more than required
                elif len(candidates) > self.num_candidates:
                    # candidates = np.random.choice(candidates, self.num_candidates, replace=False)
                    raise ValueError("Too many candidates. Which shouldn't happen.")
                return x, y, candidates

            # valid/test
            else:
                candidate_scores = self.candidate_scores[idx]
                offset = (self.num_nodes, self.num_candidates - len(candidates))

                # add dummy elements when less than required
                if len(candidates) < self.num_candidates:
                    candidate_scores = np.concatenate(
                        [candidate_scores, [-np.inf] * (self.num_candidates - len(candidates))]
                    )
                    candidates = np.concatenate(
                        [candidates, [self.num_nodes] * (self.num_candidates - len(candidates))]
                    )

                candidate_scores = np.asarray(candidate_scores, dtype=np.float32)
                return x, y, candidates, candidate_scores

        # predict
        else:
            candidate_scores = self.candidate_scores[idx]

            # add dummy elements when less than required
            if len(candidates) < self.num_candidates:
                candidate_scores = np.concatenate(
                    [candidate_scores, [-np.inf] * (self.num_candidates - len(candidates))]
                )
                candidates = np.concatenate([candidates, [self.num_nodes] * (self.num_candidates - len(candidates))])

            candidate_scores = np.asarray(candidate_scores, dtype=np.float32)
            return x, candidates, candidate_scores


def tokenize(text: str, lowercase: bool = True, tokenizer: str = "regex") -> list[str]:
    """Tokenize text.

    Args:
        text (str): Text to tokenize.
        lowercase: Whether to convert all characters to lowercase.
        tokenizer: The tokenizer from nltk to use. Can be one of ["regex", "punkt"]

    Returns:
        list: A list of tokens.
    """
    if tokenizer == "regex":
        tokenizer = RegexpTokenizer(r"\w+").tokenize
        pattern = r"^\d+$"
    elif tokenizer == "punkt":
        tokenizer = word_tokenize
        pattern = r"\W"
    elif tokenizer == "split":
        tokenizer = lambda x: x.split()
        pattern = r""
    else:
        raise ValueError(f"unsupported tokenizer {tokenizer}")
    return [t.lower() if lowercase and t != "/SEP/" else t for t in tokenizer(text) if re.sub(pattern, "", t)]


def generate_batch(data_batch):
    text_list = [data["text"] for data in data_batch]
    label_list = [data["label"] for data in data_batch]
    length_list = [len(data["text"]) for data in data_batch]
    return {
        "text": pad_sequence(text_list, batch_first=True),
        "label": torch.stack(label_list),
        "length": torch.IntTensor(length_list),
    }


def get_dataset_loader(
    data,
    classes,
    device,
    max_seq_length=500,
    batch_size=1,
    shuffle=False,
    data_workers=4,
    add_special_tokens=True,
    *,
    tokenizer=None,
    word_dict=None,
):
    """Create a pytorch DataLoader.

    Args:
        data (list[dict]): List of training instances with index, label, and tokenized text.
        classes (list): List of labels.
        device (torch.device): One of cuda or cpu.
        max_seq_length (int, optional): The maximum number of tokens of a sample. Defaults to 500.
        batch_size (int, optional): Size of training batches. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle training data before each epoch. Defaults to False.
        data_workers (int, optional): Use multi-cpu core for data pre-processing. Defaults to 4.
        add_special_tokens (bool, optional): Whether to add the special tokens. Defaults to True.
        tokenizer (transformers.PreTrainedTokenizerBase, optional): HuggingFace's tokenizer of
            the transformer-based pretrained language model. Defaults to None.
        word_dict (torchtext.vocab.Vocab, optional): A vocab object for word tokenizer to
            map tokens to indices. Defaults to None.

    Returns:
        torch.utils.data.DataLoader: A pytorch DataLoader.
    """
    dataset = TextDataset(
        data, classes, max_seq_length, word_dict=word_dict, tokenizer=tokenizer, add_special_tokens=add_special_tokens
    )
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=data_workers,
        collate_fn=generate_batch,
        pin_memory="cuda" in device.type,
    )
    return dataset_loader


def _load_raw_data(
    data,
    is_test=False,
    tokenize_text=True,
    remove_no_label_data=False,
    lowercase=True,
    tokenizer="regex",
) -> list[dict[str, list[str]]]:
    """Load and tokenize raw data in file or dataframe.

    Args:
        data (Union[str, pandas,.Dataframe]): Training, test, or validation data in file or dataframe.
        is_test (bool, optional): Whether the data is for test or not. Defaults to False.
        remove_no_label_data (bool, optional): Whether to remove training/validation instances that have no labels.
            This is effective only when is_test=False. Defaults to False.

    Returns:
        dict: [{(optional: "index": ..., ), "label": ..., "text": ...}, ...]
    """
    assert isinstance(data, str) or isinstance(data, pd.DataFrame), "Data must be from a file or pandas dataframe."
    if isinstance(data, str):
        if GLOBAL_RANK == 0:
            logging.info(f"Loading data from {data}.")
        data = pd.read_csv(data, sep="\t", header=None, on_bad_lines="warn", quoting=csv.QUOTE_NONE).fillna("")
    data = data.astype(str)
    if data.shape[1] == 2:
        data.columns = ["label", "text"]
        data = data.reset_index()
    elif data.shape[1] == 3:
        data.columns = ["index", "label", "text"]
    else:
        raise ValueError(f"Expected 2 or 3 columns, got {data.shape[1]}.")

    data["label"] = data["label"].astype(str).map(lambda s: s.split())
    if tokenize_text:
        tqdm.pandas()
        data["text"] = data["text"].progress_map(lambda t: tokenize(t, lowercase=lowercase, tokenizer=tokenizer))
    # TODO: Can we change to "list"?
    data = data.to_dict("records")
    if not is_test:
        num_no_label_data = sum(1 for d in data if len(d["label"]) == 0)
        if num_no_label_data > 0:
            if remove_no_label_data:
                logging.info(
                    f"Remove {num_no_label_data} instances that have no labels from data.", extra={"collect": True}
                )
                data = [d for d in data if len(d["label"]) > 0]
            else:
                logging.info(
                    f"Keep {num_no_label_data} instances that have no labels from data.", extra={"collect": True}
                )
    return data


def load_datasets(
    training_data=None,
    training_sparse_data=None,
    test_data=None,
    val_data=None,
    val_size=0.2,
    merge_train_val=False,
    tokenize_text=True,
    remove_no_label_data=False,
    lowercase=True,
    random_state=42,
    tokenizer="regex",
) -> dict:
    """Load data from the specified data paths or the given dataframe.
    If `val_data` does not exist but `val_size` > 0, the validation set will be split from the training dataset.

    Args:
        training_data (Union[str, pandas,.Dataframe], optional): Path to training data or a dataframe.
        training_sparse_data (Union[str, pandas,.Dataframe], optional): Path to training sparse data or a dataframe in libsvm format.
        test_data (Union[str, pandas,.Dataframe], optional): Path to test data or a dataframe.
        val_data (Union[str, pandas,.Dataframe], optional): Path to validation data or a dataframe.
        val_size (float, optional): Training-validation split: a ratio in [0, 1] or an integer for the size of the validation set.
            Defaults to 0.2.
        merge_train_val (bool, optional): Whether to merge the training and validation data.
            Defaults to False.
        tokenize_text (bool, optional): Whether to tokenize text. Defaults to True.
        lowercase: Whether to lowercase text. Defaults to True.
        remove_no_label_data (bool, optional): Whether to remove training/validation instances that have no labels.
            Defaults to False.
        random_state:

    Returns:
        dict: A dictionary of datasets.
    """
    if isinstance(training_data, str) or isinstance(test_data, str):
        assert training_data or test_data, "At least one of `training_data` and `test_data` must be specified."
    elif isinstance(training_data, pd.DataFrame) or isinstance(test_data, pd.DataFrame):
        assert (
            not training_data.empty or not test_data.empty
        ), "At least one of `training_data` and `test_data` must be specified."

    datasets = {}
    if training_data is not None:
        if Path(training_data).with_suffix(".npy").exists():
            datasets["train"] = np.load(Path(training_data).with_suffix(".npy"), allow_pickle=True).tolist()
        else:
            datasets["train"] = _load_raw_data(
                training_data,
                tokenize_text=tokenize_text,
                lowercase=lowercase,
                tokenizer=tokenizer,
                remove_no_label_data=remove_no_label_data,
            )
            if GLOBAL_RANK == 0:
                np.save(Path(training_data).with_suffix(".npy"), datasets["train"])

    if training_sparse_data is not None:
        if GLOBAL_RANK == 0:
            logging.info(f"Loading sparse training data from {training_sparse_data}.")
        datasets["train_sparse_x"] = normalize(load_svmlight_file(training_sparse_data, multilabel=True)[0])

    if val_data is not None:
        datasets["val"] = _load_raw_data(
            val_data,
            tokenize_text=tokenize_text,
            lowercase=lowercase,
            tokenizer=tokenizer,
            remove_no_label_data=remove_no_label_data,
        )
    elif val_size > 0:
        datasets["train_full"] = datasets["train"]
        datasets["train"], datasets["val"] = train_test_split(
            datasets["train"], test_size=val_size, random_state=random_state
        )

    if Path(test_data).with_suffix(".npy").exists():
        datasets["test"] = np.load(Path(test_data).with_suffix(".npy"), allow_pickle=True).tolist()
    else:
        datasets["test"] = _load_raw_data(
            test_data,
            is_test=True,
            tokenize_text=tokenize_text,
            lowercase=lowercase,
            tokenizer=tokenizer,
            remove_no_label_data=remove_no_label_data,
        )
        if GLOBAL_RANK == 0:
            np.save(Path(test_data).with_suffix(".npy"), datasets["test"])

    if merge_train_val:
        try:
            datasets["train"] = datasets["train"] + datasets["val"]
            for i in range(len(datasets["train"])):
                datasets["train"][i]["index"] = i
            del datasets["val"]
        except KeyError:
            logging.warning(
                f"Requesting merging training and val data. But no val dataset is not provided.\nSkip merging process"
            )
        finally:
            gc.collect()

    msg = " / ".join(f"{k}: {v.shape[0] if issparse(v) else len(v)}" for k, v in datasets.items())
    if GLOBAL_RANK == 0:
        logging.info(f"Finish loading dataset ({msg})")
    return datasets


def load_or_build_text_dict(
    dataset,
    vocab_file=None,
    min_vocab_freq=1,
    embed_file=None,
    embed_cache_dir=None,
    silent=False,
    normalize_embed=False,
    max_tokens=None,
    unk_init: str | None = None,
    unk_init_param: dict | None = None,
    apply_all: bool = False,
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
        max_tokens:
        unk_init:
        unk_init_param:
        apply_all:

    Returns:
        tuple[torchtext.vocab.Vocab, torch.Tensor]: A vocab object which maps tokens to indices and the pre-trained word vectors of shape (vocab_size, embed_dim).
    """
    if vocab_file:
        logging.info(f"Load vocab from {vocab_file}")
        with open(vocab_file, "r") as fp:
            vocab_list = [[vocab.strip() for vocab in fp.readlines()]]
        # Keep PAD index 0 to align `padding_idx` of
        # class Embedding in libmultilabel.nn.networks.modules.
        vocabs = build_vocab_from_iterator(vocab_list, min_freq=1, specials=[PAD, UNK])
    else:
        vocab_list = [set(data["text"]) for data in dataset]
        vocabs = build_vocab_from_iterator(
            vocab_list, min_freq=min_vocab_freq, specials=[PAD, UNK], max_tokens=max_tokens
        )
    vocabs.set_default_index(vocabs[UNK])
    logging.info(f"Read {len(vocabs)} vocabularies.")

    embedding_weights = get_embedding_weights_from_file(
        word_dict=vocabs,
        embed_file=embed_file,
        silent=silent,
        cache=embed_cache_dir,
        unk_init=unk_init,
        unk_init_param=unk_init_param,
        apply_all=apply_all,
    )

    if normalize_embed:
        # To have better precision for calculating the normalization, we convert the original
        # embedding_weights from a torch.FloatTensor to a torch.DoubleTensor.
        # After the normalization, we will convert the embedding_weights back to a torch.FloatTensor.
        embedding_weights = embedding_weights.double()
        for i, vector in enumerate(embedding_weights):
            # We use the constant 1e-6 by following https://github.com/jamesmullenbach/caml-mimic/blob/44a47455070d3d5c6ee69fb5305e32caec104960/dataproc/extract_wvs.py#L60
            # for an internal experiment of reproducing their results.
            embedding_weights[i] = vector / float(torch.linalg.norm(vector) + 1e-6)
        embedding_weights = embedding_weights.float()

    return vocabs, embedding_weights


def load_or_build_label(datasets, label_file=None, include_test_labels=False):
    """Obtain the label set from loading a label file or from the given data sets. The label set contains
    labels in the training and validation sets. Labels in the test set are included only when
    `include_test_labels` is True.

    Args:
        datasets (dict): A dictionary of datasets. Each dataset contains list of instances
            with index, label, and tokenized text.
        label_file (str, optional): Path to a file holding all labels.
        include_test_labels (bool, optional): Whether to include labels in the test dataset.
            Defaults to False.

    Returns:
        list: A list of labels sorted in alphabetical order.
    """
    if label_file is not None:
        logging.info(f"Load labels from {label_file}.")
        with open(label_file, "r") as fp:
            classes = sorted([s.strip() for s in fp.readlines()])
    else:
        if "test" not in datasets and include_test_labels:
            raise ValueError(f"Specified the inclusion of test labels but test file does not exist")

        classes = set()

        for split, data in datasets.items():
            if (split == "test" and not include_test_labels) or split == "train_sparse_x":
                continue
            for instance in data:
                classes.update(instance["label"])
        classes = sorted(classes)
    if GLOBAL_RANK == 0:
        logging.info(f"Read {len(classes)} labels.")

    return classes


def get_embedding_weights_from_file(
    word_dict,
    embed_file,
    silent=False,
    cache=None,
    unk_init: str | None = None,
    unk_init_param: dict | None = None,
    apply_all: bool = False,
):
    """If the word exists in the embedding file, load the pretrained word embedding.
    Otherwise, assign a zero vector to that word.

    Args:
        word_dict (torchtext.vocab.Vocab): A vocab object which maps tokens to indices.
        embed_file (str): Path to a file holding pre-trained embeddings.
        silent (bool, optional): Enable silent mode. Defaults to False.
        cache (str, optional): Path to a directory for storing cached embeddings. Defaults to None.
        unk_init: (str, optional): by default, initialize out-of-vocabulary word vectors
            to zero vectors; can be ["uniform", None]
        unk_init_param: (dict, optional): works if unk_init is not None. For example, {"from": -1, "to": 1}.
        apply_all: (bool, optional): If True, apply unk_init to all unknown words. Otherwise, apply only to UNK.

    Returns:
        torch.Tensor: Embedding weights (vocab_size, embed_size)
    """
    if unk_init_param is None:
        unk_init_param = {}

    if Path(embed_file).exists():
        if embed_file.endswith(".gensim"):
            logging.info(f"Load pretrained embedding from gensim file: {embed_file}.")
            vector_dict = KeyedVectors.load(embed_file)
            embed_size = vector_dict.vector_size
        else:
            logging.info(f"Load pretrained embedding from file: {embed_file}.")

            with open(embed_file) as f:
                word_vectors = f.readlines()
            embed_size = len(word_vectors[0].split()) - 1

            if apply_all:
                if unk_init == "uniform":
                    logging.info(f"uniform is applied to all unknown words with parameters {unk_init_param}")
                    vector_dict = defaultdict(lambda: torch.empty(embed_size).uniform_(**unk_init_param))
                else:
                    raise ValueError(f"Unsupported embedding initialization {unk_init} for unknown words.")
            else:
                vector_dict = defaultdict(lambda: torch.zeros(embed_size))

            for word_vector in tqdm(word_vectors, disable=silent, desc="Building token-embedding map"):
                word, vector = word_vector.rstrip().split(" ", 1)
                vector = torch.Tensor(list(map(float, vector.split())))
                vector_dict[word] = vector
    else:
        # Adapted from https://pytorch.org/text/0.9.0/_modules/torchtext/vocab.html#Vocab.load_vectors.
        if embed_file not in pretrained_aliases:
            raise ValueError(
                "Got embed_file {}, but allowed pretrained "
                "vectors are {}".format(embed_file, list(pretrained_aliases.keys()))
            )

        logging.info(f"Load pretrained embedding from torchtext.")
        # Hotfix: Glove URLs are outdated in Torchtext
        # (https://github.com/pytorch/text/blob/main/torchtext/vocab/vectors.py#L213-L217)
        pretrained_cls = pretrained_aliases[embed_file]
        if embed_file.startswith("glove"):
            for name, url in pretrained_cls.func.url.items():
                file_name = url.split("/")[-1]
                pretrained_cls.func.url[name] = f"https://huggingface.co/stanfordnlp/glove/resolve/main/{file_name}"
        if apply_all:
            if unk_init == "uniform":
                logging.info(f"uniform is applied to all unknown words with parameters {unk_init_param}")
                vector_dict = pretrained_cls(cache=cache, unk_init=lambda x: x.uniform_(**unk_init_param))
            else:
                raise ValueError("Unsupported embedding initialization for unknown words.")
        else:
            # we do not utilize unk_int here for the sake of simplicity of logic
            vector_dict = pretrained_cls(cache=cache)

        embed_size = vector_dict.dim

    # Store pretrained word embedding
    embedding_weights = torch.zeros(len(word_dict), embed_size)

    vec_counts = 0

    # initialize embedding generator for unknown words
    if apply_all:
        if unk_init == "uniform":
            logging.info(f"uniform is applied to all unknown words with parameters {unk_init_param}")
            unk_generator = partial(uniform_embedding_generator, embed_size=embed_size, unk_init_param=unk_init_param)
        else:
            raise ValueError(f"Unsupported embedding initialization {unk_init} for unknown words.")
    # the default embedding generator for unknown words
    else:
        unk_generator = partial(torch.zeros, size=(embed_size,))

    if isinstance(vector_dict, Vectors):
        for word in tqdm(word_dict.get_itos(), desc="Retrieving pretrained embeddings"):
            # "word" in torchtext.Vectors will hang forever
            embedding_weights[word_dict[word]] = vector_dict[word]
            if word in vector_dict.itos:
                vec_counts += 1
    else:
        for word in tqdm(word_dict.get_itos(), desc="Retrieving pretrained embeddings"):
            if word in vector_dict:
                embedding_weights[word_dict[word]] = (
                    torch.tensor(vector_dict[word]) if isinstance(vector_dict, KeyedVectors) else vector_dict[word]
                )
                vec_counts += 1
            else:
                embedding_weights[word_dict[word]] = unk_generator()
    # force the embedding of PAD to be a 0 vector
    if apply_all:
        embedding_weights[0] = torch.zeros(embed_size)
    logging.info(f"Load {vec_counts} pretrained embeddings and a total number of {len(word_dict)} embeddings")

    if unk_init and not apply_all:
        # Might not deal with all situations. Change as needed.
        # Add UNK embedding
        # CAML: np.random.randn(embed_size)
        if unk_init == "random":
            unk_vector = torch.randn(embed_size)
        else:
            raise ValueError("Unsupported embedding initialization for UNK.")
        embedding_weights[word_dict[UNK]] = unk_vector

    return embedding_weights


def uniform_embedding_generator(embed_size: int, unk_init_param: dict):
    # As "from" is a python keyword, "from" is not allowed to be used as an argument.
    # However, it can be unpacked from a dict and passed to a function written in C.
    # Thus, unk_init_param: dict is used here.
    return torch.empty(embed_size).uniform_(**unk_init_param)
