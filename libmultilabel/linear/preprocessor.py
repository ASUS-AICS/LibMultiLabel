from __future__ import annotations

import os
import re
from array import array
from collections import defaultdict
from collections.abc import Iterable

import scipy
import scipy.sparse as sparse
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

__all__ = ['Preprocessor']

# Note the trailing newline
LibMultiLabelFormat = r'(?:\w+\t)?([\w\.]*(?: [\w\.]+)*)\t(\s*\S+(?:\s+\S+)*)\n'


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
            train = split_label_text(
                (l for l in open(self.config.train_path)), LibMultiLabelFormat)
            self._fit_txt(train['texts'], train['labels'])
            datasets['train']['x'] = self.vectorizer.transform(train['texts'])
            datasets['train']['y'] = self.binarizer.transform(
                train['labels']).astype('d')
        if os.path.exists(self.config.test_path):
            test = split_label_text(
                (l for l in open(self.config.test_path)), LibMultiLabelFormat)
            datasets['test']['x'] = self.vectorizer.transform(test['texts'])
            datasets['test']['y'] = self.binarizer.transform(
                test['labels']).astype('d')
        return dict(datasets)

    def _fit_txt(self, texts, labels):
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(texts)
        # TODO: allow integer labels
        self.binarizer = MultiLabelBinarizer(sparse_output=True)
        self.binarizer.fit(labels)

    def _load_svm(self) -> 'dict[str, dict]':
        datasets = defaultdict(dict)
        if not self.config.eval:
            y, x = svm_read_problem(self.config.train_path)
            self._fit_svm(y)
            datasets['train']['x'] = x
            datasets['train']['y'] = self.binarizer.transform(y).astype('d')
        if os.path.exists(self.config.test_path):
            ty, tx = svm_read_problem(self.config.test_path)
            datasets['test']['x'] = tx
            datasets['test']['y'] = self.binarizer.transform(ty).astype('d')
        return dict(datasets)

    def _fit_svm(self, y):
        self.binarizer = MultiLabelBinarizer(sparse_output=True)
        self.binarizer.fit(y)


def split_label_text(raw_text: 'Iterable[str]', pattern: str) -> 'dict[str,list[str]]':
    p = re.compile(pattern)
    labels = []
    texts = []
    for line in raw_text:
        m = p.fullmatch(line)
        if m is None:
            raise ValueError(f'"{pattern}" doesn\'t match:\n{line}')
        labels.append(m[1])
        texts.append(m[2])
    return {
        'labels': labels,
        'texts': texts,
    }


def svm_read_problem(file_path: str) -> 'tuple[list[list[int]], sparse.csr_matrix]':
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
