from __future__ import annotations

import os
import re
from array import array
from collections import defaultdict

import pandas as pd
import scipy
import scipy.sparse as sparse
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

__all__ = ['Preprocessor']


class Preprocessor:
    def __init__(self, config) -> None:
        self.config = config
        if not config.data_format in {'txt', 'svm'}:
            raise ValueError(f'unsupported data format {config.data_format}')

    def load_data(self) -> 'dict[str, dict]':
        if self.config.data_format == 'txt':
            return self._load_txt()
        elif self.config.data_format == 'svm':
            return self._load_svm()

    def _load_txt(self) -> 'dict[str, dict]':
        datasets = defaultdict(dict)
        if not self.config.eval:
            train = read_libmultilabel_format(self.config.train_path)
            self._generate_tfidf(train['text'])
            self._generate_label_mapping(train['label'])
            datasets['train']['x'] = self.vectorizer.transform(train['text'])
            datasets['train']['y'] = self.binarizer.transform(
                train['label']).astype('d')
        if os.path.exists(self.config.test_path):
            test = read_libmultilabel_format(self.config.test_path)
            datasets['test']['x'] = self.vectorizer.transform(test['text'])
            datasets['test']['y'] = self.binarizer.transform(
                test['label']).astype('d')
        return dict(datasets)

    def _load_svm(self) -> 'dict[str, dict]':
        datasets = defaultdict(dict)
        if not self.config.eval:
            y, x = read_libsvm_format(self.config.train_path)
            self._generate_label_mapping(y)
            datasets['train']['x'] = x
            datasets['train']['y'] = self.binarizer.transform(y).astype('d')
        if os.path.exists(self.config.test_path):
            ty, tx = read_libsvm_format(self.config.test_path)
            datasets['test']['x'] = tx
            datasets['test']['y'] = self.binarizer.transform(ty).astype('d')
        return dict(datasets)

    def _generate_tfidf(self, texts):
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(texts)

    def _generate_label_mapping(self, labels):
        self.binarizer = MultiLabelBinarizer(sparse_output=True)
        self.binarizer.fit(labels)


def read_libmultilabel_format(path: str) -> 'dict[str,list[str]]':
    data = pd.read_csv(path, sep='\t', header=None,
                       on_bad_lines='skip').fillna('')
    if data.shape[1] == 2:
        data.columns = ['label', 'text']
        data = data.reset_index()
    elif data.shape[1] == 3:
        data.columns = ['index', 'label', 'text']
    else:
        raise ValueError(f'Expected 2 or 3 columns, got {data.shape[1]}.')
    data['label'] = data['label'].map(lambda s: s.split())
    return data.to_dict('list')


def read_libsvm_format(file_path: str) -> 'tuple[list[list[int]], sparse.csr_matrix]':
    """Read multi-label LIBSVM-format data.

    Args:
        file_path (str): Path to file.

    Returns:
        tuple[list[list[int]], sparse.csr_matrix]: A tuple of labels and features.
    """
    def as_ints(str):
        return [int(s) for s in str.split(',')]

    prob_y = []
    prob_x = array('d')
    row_ptr = array('l', [0])
    col_idx = array('l')

    for i, line in enumerate(open(file_path)):
        line = line.split(None, 1)
        # In case an instance with all zero features
        if len(line) == 1:
            line += ['']
        label, features = line
        prob_y.append(as_ints(label))
        nz = 0
        for e in features.split():
            ind, val = e.split(':')
            val = float(val)
            if val != 0:
                col_idx.append(int(ind) - 1)
                prob_x.append(val)
                nz += 1
        row_ptr.append(row_ptr[-1]+nz)

    prob_x = scipy.frombuffer(prob_x, dtype='d')
    col_idx = scipy.frombuffer(col_idx, dtype='l')
    row_ptr = scipy.frombuffer(row_ptr, dtype='l')
    prob_x = sparse.csr_matrix((prob_x, col_idx, row_ptr))

    return (prob_y, prob_x)
