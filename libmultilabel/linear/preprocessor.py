from __future__ import annotations

import csv
import logging
import re
from array import array
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

__all__ = ['Preprocessor', 'read_libmultilabel_format', 'read_libsvm_format']


class Preprocessor:
    """Preprocessor is used to load and preprocess input data in LibSVM and LibMultiLabel formats.
    The same Preprocessor has to be used for both training and testing data;
    see save_pipeline and load_pipeline.
    """

    def __init__(self, data_format: str) -> None:
        """Initializes the preprocessor.

        Args:
            data_format (str): The data format used. 'svm' for LibSVM format, 'txt' for LibMultiLabel format in file and 'dataframe' for LibMultiLabel format in dataframe .
        """
        if not data_format in {'txt', 'svm', 'dataframe'}:
            raise ValueError(f'unsupported data format {data_format}')

        self.data_format = data_format

    def load_data(self, training_data: Union[str, pd.DataFrame] = None,
                  test_data: Union[str, pd.DataFrame] = None,
                  eval: bool = False,
                  label_file: str = None,
                  include_test_labels: bool = False,
                  remove_no_label_data: bool = False) -> 'dict[str, dict]':
        """Loads and preprocesses data.

        Args:
            training_data (Union[str, pd.DataFrame]): Training data file or dataframe in LibMultiLabel format. Ignored if eval is True. Defaults to None.
            test_data (Union[str, pd.DataFrame]): Test data file or dataframe in LibMultiLabel format. Ignored if test_data doesn't exist. Defaults to None.
            eval (bool): If True, ignores training data and uses previously loaded state to preprocess test data.
            label_file (str, optional): Path to a file holding all labels.
            include_test_labels (bool, optional): Whether to include labels in the test dataset. Defaults to False.
            remove_no_label_data (bool, optional): Whether to remove training instances that have no labels.

        Returns:
            dict[str, dict]: The training and test data, with keys 'train' and 'test' respectively. The data
            has keys 'x' for input features and 'y' for labels.
        """
        if label_file is not None:
            logging.info(f'Load labels from {label_file}.')
            with open(label_file, 'r') as fp:
                self.classes = sorted([s.strip() for s in fp.readlines()])
        else:
            if test_data is None and include_test_labels:
                raise ValueError(
                    f'Specified the inclusion of test labels but test file does not exist')
            self.classes = None
            self.include_test_labels = include_test_labels

        if self.data_format in {'txt', 'dataframe'}:
            data = self._load_text(training_data, test_data, eval)
        elif self.data_format == 'svm':
            data = self._load_svm(training_data, test_data, eval)

        if 'train' in data:
            num_labels = data['train']['y'].getnnz(axis=1)
            num_no_label_data = np.count_nonzero(num_labels == 0)
            if num_no_label_data > 0:
                if remove_no_label_data:
                    logging.info(
                        f'Remove {num_no_label_data} instances that have no labels from data.',
                        extra={'collect': True})
                    data['train']['x'] = data['train']['x'][num_labels > 0]
                    data['train']['y'] = data['train']['y'][num_labels > 0]
                else:
                    logging.info(
                        f'Keep {num_no_label_data} instances that have no labels from data.',
                        extra={'collect': True})

        return data

    def _load_text(self, training_data, test_data, eval) -> 'dict[str, dict]':
        datasets = defaultdict(dict)
        if test_data is not None:
            test = read_libmultilabel_format(test_data)

        if not eval:
            train = read_libmultilabel_format(training_data)
            self._generate_tfidf(train['text'])

            if self.classes or not self.include_test_labels:
                self._generate_label_mapping(train['label'], self.classes)
            else:
                self._generate_label_mapping(train['label'] + test['label'])
            datasets['train']['x'] = self.vectorizer.transform(train['text'])
            datasets['train']['y'] = self.binarizer.transform(
                train['label']).astype('d')

        if test_data is not None:
            datasets['test']['x'] = self.vectorizer.transform(test['text'])
            datasets['test']['y'] = self.binarizer.transform(
                test['label']).astype('d')

        return dict(datasets)

    def _load_svm(self, training_data, test_data, eval) -> 'dict[str, dict]':
        datasets = defaultdict(dict)
        if test_data is not None:
            ty, tx = read_libsvm_format(test_data)

        if not eval:
            y, x = read_libsvm_format(training_data)
            if self.classes or not self.include_test_labels:
                self._generate_label_mapping(y, self.classes)
            else:
                self._generate_label_mapping(y + ty)
            datasets['train']['x'] = x
            datasets['train']['y'] = self.binarizer.transform(y).astype('d')

        if test_data is not None:
            datasets['test']['x'] = tx
            datasets['test']['y'] = self.binarizer.transform(ty).astype('d')
        return dict(datasets)

    def _generate_tfidf(self, texts):
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(texts)

    def _generate_label_mapping(self, labels, classes=None):
        self.binarizer = MultiLabelBinarizer(
            sparse_output=True, classes=classes)
        self.binarizer.fit(labels)


def read_libmultilabel_format(data: Union[str, pd.DataFrame]) -> 'dict[str,list[str]]':
    """Read multi-label text data from file or pandas dataframe.

    Args:
        data (Union[str, pd.DataFrame]): A file path to data in `LibMultiLabel format <https://www.csie.ntu.edu.tw/~cjlin/libmultilabel/cli/ov_data_format.html#libmultilabel-format>`_
            or a pandas dataframe contains index (optional), label, and text.
    Returns:
        dict[str,list[str]]: A dictionary with a list of index (optional), label, and text.
    """
    assert isinstance(data, str) or isinstance(data, pd.DataFrame), "Data must be from a file or pandas dataframe."
    if isinstance(data, str):
        data = pd.read_csv(data, sep='\t', header=None,
                           on_bad_lines='warn', quoting=csv.QUOTE_NONE).fillna('')
    data = data.astype(str)
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

    pattern = re.compile(r'(?!^$)([+\-0-9,]+\s+)?(.*\n?)')
    for i, line in enumerate(open(file_path)):
        m = pattern.fullmatch(line)
        try:
            labels = m[1]
            prob_y.append(as_ints(labels) if labels else [])
            features = m[2] or ''
            nz = 0
            for e in features.split():
                ind, val = e.split(':')
                ind, val = int(ind), float(val)
                if ind < 1:
                    raise IndexError(
                        f'invalid svm format at line {i+1} of the file \'{file_path}\' --> Indices should start from one.')
                if val != 0:
                    col_idx.append(ind - 1)
                    prob_x.append(val)
                    nz += 1
            row_ptr.append(row_ptr[-1]+nz)
        except IndexError:
            raise
        except:
            raise ValueError(
                f'invalid svm format at line {i+1} of the file \'{file_path}\'')

    prob_x = scipy.frombuffer(prob_x, dtype='d')
    col_idx = scipy.frombuffer(col_idx, dtype='l')
    row_ptr = scipy.frombuffer(row_ptr, dtype='l')
    prob_x = sparse.csr_matrix((prob_x, col_idx, row_ptr))

    return (prob_y, prob_x)
