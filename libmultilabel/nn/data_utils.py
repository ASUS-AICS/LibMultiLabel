import gc
import logging
import warnings
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

# TODO: why this?
warnings.simplefilter(action='ignore', category=FutureWarning)


def _load_raw_data(path, is_test=False, remove_no_label_data=False) -> pd.DataFrame:
    """Load and tokenize raw data.

    Args:
        path (str): Path to training, test, or validation data.
        is_test (bool, optional): Whether the data is for test or not. Defaults to False.
        remove_no_label_data (bool, optional): Whether to remove training/validation instances that have no labels.
            This is effective only when is_test=False. Defaults to False.

    Returns:
        list: Data composed of label, text and optionally index.
    """
    logging.info(f'Load data from {path}.')
    data = pd.read_csv(path, sep='\t', header=None,
                       error_bad_lines=False, warn_bad_lines=True).fillna('')
    if data.shape[1] == 2:
        data.columns = ['label', 'text']
        data = data.reset_index()
    elif data.shape[1] == 3:
        data.columns = ['index', 'label', 'text']
    else:
        raise ValueError(f'Expected 2 or 3 columns, got {data.shape[1]}.')

    data['label'] = data['label'].astype(str).map(lambda s: s.split())
    if not is_test:
        num_no_label_data = sum(len(l) == 0 for l in data['label'])
        if num_no_label_data > 0:
            if remove_no_label_data:
                logging.info(
                    f'Remove {num_no_label_data} instances that have no labels from {path}.')
                data = data[data['label'].map(len) > 0]
            else:
                logging.info(
                    f'Keep {num_no_label_data} instances that have no labels from {path}.')
    return data


def load_datasets(
    training_file=None,
    test_file=None,
    val_file=None,
    val_size=0.2,
    merge_train_val=False,
    remove_no_label_data=False
) -> 'dict[str, pd.DataFrame]':
    """Load data from the specified data paths (i.e., `training_file`, `test_file`, and `val_file`).
    If `valid.txt` does not exist but `val_size` > 0, the validation set will be split from the training dataset.

    Args:
        training_file (str, optional): Path to training data.
        test_file (str, optional): Path to test data.
        val_file (str, optional): Path to validation data.
        val_size (float, optional): Training-validation split: a ratio in [0, 1] or an integer for the size of the validation set.
            Defaults to 0.2.
        merge_train_val (bool, optional): Whether to merge the training and validation data.
            Defaults to False.
        remove_no_label_data (bool, optional): Whether to remove training/validation instances that have no labels.
            Defaults to False.

    Returns:
        dict: A dictionary of datasets.
    """
    assert training_file or test_file, "At least one of `training_file` and `test_file` must be specified."

    datasets = {}
    if training_file is not None:
        datasets['train'] = _load_raw_data(training_file,
                                           remove_no_label_data=remove_no_label_data)

    if val_file is not None:
        datasets['val'] = _load_raw_data(val_file,
                                         remove_no_label_data=remove_no_label_data)
    elif val_size > 0:
        datasets['train'], datasets['val'] = train_test_split(
            datasets['train'], test_size=val_size, random_state=42)

    if test_file is not None:
        datasets['test'] = _load_raw_data(test_file, is_test=True,
                                          remove_no_label_data=remove_no_label_data)

    if merge_train_val:
        datasets['train'] = pd.concat([datasets['train'], datasets['val']])
        # TODO: re-indexing behaviour should be documented
        datasets['train']['index'] = datasets['train'].index
        del datasets['val']
        # TODO: why this?
        gc.collect()

    msg = ' / '.join(f'{k}: {len(v)}' for k, v in datasets.items())
    logging.info(f'Finish loading dataset ({msg})')
    return datasets


def load_or_build_label(datasets: 'dict[str, pd.DataFrame]',
                        label_file: Optional[str] = None,
                        include_test_labels: bool = False) -> 'list[str]':
    """Generate label set either by the given datasets or a predefined label file.

    Args:
        datasets (dict[str, pd.DataFrame]): A dictionary of dataframes, one for each data split.
            Every dataframe must contain a 'label' column.
        label_file (str, optional): Path to a file holding all labels.
        include_test_labels (bool, optional): Whether to include labels in the test dataset.
            Defaults to False.

    Returns:
        list[str]: A list of labels sorted in alphabetical order.
    """
    if label_file is not None:
        logging.info(f'Load labels from {label_file}.')
        with open(label_file, 'r') as fp:
            classes = sorted([s.strip() for s in fp.readlines()])
    else:
        if 'test' not in datasets and include_test_labels:
            raise ValueError(
                f'Specified the inclusion of test labels but test file does not exist')

        classes = set()

        for split, data in datasets.items():
            if split == 'test' and not include_test_labels:
                continue
            for l in data['label']:
                classes.update(l)
        classes = sorted(classes)
    logging.info(f'Read {len(classes)} labels.')
    return classes
