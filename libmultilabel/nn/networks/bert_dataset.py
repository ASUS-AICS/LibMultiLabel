import warnings
from typing import Any

import pandas as pd
import torch
import transformers
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# TODO: why this?
warnings.simplefilter(action='ignore', category=FutureWarning)
transformers.logging.set_verbosity_error()


class BertDataset(Dataset):
    """Amazing docstring about this class"""

    def __init__(self, data: pd.DataFrame,
                 classes: 'list[str]',
                 tokenizer: str,
                 max_seq_length: int,
                 add_special_tokens: bool):
        self.data = data
        self.classes = classes
        self.max_seq_length = max_seq_length
        self.label_binarizer = MultiLabelBinarizer().fit([classes])
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer, use_fast=True)
        self.add_special_tokens = add_special_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index]

        # TODO: fix truncation issue (bert special tokens should not be truncated)
        if self.add_special_tokens:  # tentatively hard code
            input_ids = self.tokenizer.encode(data['text'],
                                              padding='max_length',
                                              max_length=self.max_seq_length,
                                              truncation=True)
        else:
            input_ids = self.tokenizer.encode(
                data['text'], add_special_tokens=False)
        return {
            'text': torch.LongTensor(input_ids[:self.max_seq_length]),
            'label': torch.IntTensor(self.label_binarizer.transform([data['label']])[0])
        }

# TODO: fix how padding is done here
def collate_fn(data_batch):
    text_list = [data['text'] for data in data_batch]
    label_list = [data['label'] for data in data_batch]
    length_list = [len(data['text']) for data in data_batch]
    return {
        'text': pad_sequence(text_list, batch_first=True),
        'label': torch.stack(label_list),
        'length': torch.IntTensor(length_list)
    }
