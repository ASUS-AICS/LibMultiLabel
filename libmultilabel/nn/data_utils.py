import csv
import gc
import logging
import warnings

import pandas as pd
import torch
import transformers
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator, pretrained_aliases, Vocab
from tqdm import tqdm

transformers.logging.set_verbosity_error()
warnings.simplefilter(action="ignore", category=FutureWarning)

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


def tokenize(text):
    """Tokenize text.

    Args:
        text (str): Text to tokenize.

    Returns:
        list: A list of tokens.
    """
    tokenizer = RegexpTokenizer(r"\w+")
    return [t.lower() for t in tokenizer.tokenize(text) if not t.isnumeric()]


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


def _load_raw_data(data, is_test=False, tokenize_text=True, remove_no_label_data=False):
    """Load and tokenize raw data in file or dataframe.

    Args:
        data (Union[str, pandas,.Dataframe]): Training, test, or validation data in file or dataframe.
        is_test (bool, optional): Whether the data is for test or not. Defaults to False.
        remove_no_label_data (bool, optional): Whether to remove training/validation instances that have no labels.
            This is effective only when is_test=False. Defaults to False.

    Returns:
        pandas.DataFrame: Data composed of index, label, and tokenized text.
    """
    assert isinstance(data, str) or isinstance(data, pd.DataFrame), "Data must be from a file or pandas dataframe."
    if isinstance(data, str):
        logging.info(f"Load data from {data}.")
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
        data["text"] = data["text"].map(tokenize)
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
    test_data=None,
    val_data=None,
    val_size=0.2,
    merge_train_val=False,
    tokenize_text=True,
    remove_no_label_data=False,
):
    """Load data from the specified data paths or the given dataframe.
    If `val_data` does not exist but `val_size` > 0, the validation set will be split from the training dataset.

    Args:
        training_data (Union[str, pandas,.Dataframe], optional): Path to training data or a dataframe.
        test_data (Union[str, pandas,.Dataframe], optional): Path to test data or a dataframe.
        val_data (Union[str, pandas,.Dataframe], optional): Path to validation data or a dataframe.
        val_size (float, optional): Training-validation split: a ratio in [0, 1] or an integer for the size of the validation set.
            Defaults to 0.2.
        merge_train_val (bool, optional): Whether to merge the training and validation data.
            Defaults to False.
        tokenize_text (bool, optional): Whether to tokenize text. Defaults to True.
        remove_no_label_data (bool, optional): Whether to remove training/validation instances that have no labels.
            Defaults to False.

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
        datasets["train"] = _load_raw_data(
            training_data, tokenize_text=tokenize_text, remove_no_label_data=remove_no_label_data
        )

    if val_data is not None:
        datasets["val"] = _load_raw_data(
            val_data, tokenize_text=tokenize_text, remove_no_label_data=remove_no_label_data
        )
    elif val_size > 0:
        datasets["train"], datasets["val"] = train_test_split(datasets["train"], test_size=val_size, random_state=42)

    if test_data is not None:
        datasets["test"] = _load_raw_data(
            test_data, is_test=True, tokenize_text=tokenize_text, remove_no_label_data=remove_no_label_data
        )

    if merge_train_val:
        datasets["train"] = datasets["train"] + datasets["val"]
        for i in range(len(datasets["train"])):
            datasets["train"][i]["index"] = i
        del datasets["val"]
        gc.collect()

    msg = " / ".join(f"{k}: {len(v)}" for k, v in datasets.items())
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
        logging.info(f"Load vocab from {vocab_file}")
        with open(vocab_file, "r") as fp:
            vocab_list = [[vocab.strip() for vocab in fp.readlines()]]
        # Keep PAD index 0 to align `padding_idx` of
        # class Embedding in libmultilabel.nn.networks.modules.
        vocabs = build_vocab_from_iterator(vocab_list, min_freq=1, specials=[PAD, UNK])
    else:
        vocab_list = [set(data["text"]) for data in dataset]
        vocabs = build_vocab_from_iterator(vocab_list, min_freq=min_vocab_freq, specials=[PAD, UNK])
    vocabs.set_default_index(vocabs[UNK])
    logging.info(f"Read {len(vocabs)} vocabularies.")

    embedding_weights = get_embedding_weights_from_file(vocabs, embed_file, silent, embed_cache_dir)

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
            if split == "test" and not include_test_labels:
                continue
            for instance in data:
                classes.update(instance["label"])
        classes = sorted(classes)
    logging.info(f"Read {len(classes)} labels.")
    return classes


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
    load_embedding_from_file = embed_file not in pretrained_aliases
    if load_embedding_from_file:
        logging.info(f"Load pretrained embedding from file: {embed_file}.")
        with open(embed_file) as f:
            word_vectors = f.readlines()
        embed_size = len(word_vectors[0].split()) - 1
        vector_dict = {}
        for word_vector in tqdm(word_vectors, disable=silent):
            word, vector = word_vector.rstrip().split(" ", 1)
            vector = torch.Tensor(list(map(float, vector.split())))
            vector_dict[word] = vector
    else:
        logging.info(f"Load pretrained embedding from torchtext.")
        # Adapted from https://pytorch.org/text/0.9.0/_modules/torchtext/vocab.html#Vocab.load_vectors.
        if embed_file not in pretrained_aliases:
            raise ValueError(
                "Got embed_file {}, but allowed pretrained "
                "vectors are {}".format(embed_file, list(pretrained_aliases.keys()))
            )
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

    logging.info(f"loaded {vec_counts}/{len(word_dict)} word embeddings")

    return embedding_weights
