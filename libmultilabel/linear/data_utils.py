from __future__ import annotations

import csv
import logging
import re
from array import array
from collections import defaultdict

import pandas as pd
import scipy
import scipy.sparse as sparse

__all__ = ["load_dataset"]


def _read_libmultilabel_format(data: str | pd.Dataframe) -> dict[str, list[str]]:
    """Read multi-label text data from file or pandas dataframe.

    Args:
        data ('str | pd.Dataframe'): A file path to data in `LibMultiLabel format <https://www.csie.ntu.edu.tw/~cjlin/libmultilabel/cli/ov_data_format.html#libmultilabel-format>`_
            or a pandas dataframe contains index (optional), label, and text.

    Returns:
        dict[str,list[str]]: A dictionary with a list of index (optional), label, and text.
    """
    assert isinstance(data, str) or isinstance(data, pd.DataFrame), "Data must be from a file or pandas dataframe."
    if isinstance(data, str):
        data = pd.read_csv(data, sep="\t", header=None, on_bad_lines="warn", quoting=csv.QUOTE_NONE).fillna("")
    data = data.astype(str)
    if data.shape[1] == 2:
        data.columns = ["y", "x"]
        data = data.reset_index()
    elif data.shape[1] == 3:
        data.columns = ["idx", "y", "x"]
    else:
        raise ValueError(f"Expected 2 or 3 columns, got {data.shape[1]}.")
    data["y"] = data["y"].map(lambda s: s.split())
    return data.to_dict("list")


def _read_libsvm_format(file_path: str) -> dict[str, list[list[int]] | sparse.csr_matrix]:
    """Read multi-label LIBSVM-format data.

    Args:
        file_path (str): Path to file.

    Returns:
        tuple[list[list[int]], sparse.csr_matrix]: A tuple of labels and features.
    """
    prob_y = []
    prob_x = array("d")
    row_ptr = array("l", [0])
    col_idx = array("l")

    pattern = re.compile(r"(?!^$)([+\-0-9,]+\s+)?(.*\n?)")
    for i, line in enumerate(open(file_path)):
        m = pattern.fullmatch(line)
        try:
            labels = m[1]
            int_labels = [int(s) for s in labels.split(",")]
            prob_y.append(int_labels if labels else [])
            features = m[2] or ""
            nz = 0
            for e in features.split():
                idx, val = e.split(":")
                idx, val = int(idx), float(val)
                if idx < 1:
                    raise IndexError(
                        f"invalid svm format at line {i + 1} of the file '{file_path}' --> Indices should start from one."
                    )
                if val != 0:
                    col_idx.append(idx - 1)
                    prob_x.append(val)
                    nz += 1
            row_ptr.append(row_ptr[-1] + nz)
        except IndexError:
            raise
        except:
            raise ValueError(f"invalid svm format at line {i + 1} of the file '{file_path}'")

    prob_x = scipy.frombuffer(prob_x, dtype="d")
    col_idx = scipy.frombuffer(col_idx, dtype="l")
    row_ptr = scipy.frombuffer(row_ptr, dtype="l")
    prob_x = sparse.csr_matrix((prob_x, col_idx, row_ptr))

    return {"x": prob_x, "y": prob_y}


def load_dataset(
    data_format: str,
    train_path: str | pd.DataFrame | None = None,
    test_path: str | pd.DataFrame | None = None,
    label_path: str | None = None,
) -> dict[str, dict[str, sparse.csr_matrix | list[list[int]] | list[str]]]:
    """Load dataset in LibSVM or LibMultiLabel formats.

    Args:
        data_format (str): The data format used. 'svm' for LibSVM format, 'txt' for LibMultiLabel format in file and 'dataframe' for LibMultiLabel format in dataframe .
        train_path (str | pd.DataFrame, optional): Training data file or dataframe in LibMultiLabel format. Ignored if eval is True. Defaults to None.
        test_path (str | pd.DataFrame, optional): Test data file or dataframe in LibMultiLabel format. Ignored if test_data doesn't exist. Defaults to None.
        label_path (str, optional): Path to a file holding all labels. Defaults to None.

    Returns:
        dict[str, dict[str, sparse.csr_matrix | str]]: The training and/or test data, with keys 'train' and 'test' respectively.
        The data has keys 'x' for input features and 'y' for labels.
    """
    if data_format not in {"txt", "svm", "dataframe"}:
        raise ValueError(f"unsupported data format {data_format}")
    if train_path is None and test_path is None:
        raise ValueError("train_path and test_path cannot be both None.")

    dataset = defaultdict(dict)
    dataset["data_format"] = data_format

    # load training and test datasets
    if data_format in {"txt", "dataframe"}:
        if train_path is not None:
            train = _read_libmultilabel_format(train_path)
        if test_path is not None:
            test = _read_libmultilabel_format(test_path)
    if data_format in {"svm"}:
        if train_path is not None:
            train = _read_libsvm_format(train_path)
        if test_path is not None:
            test = _read_libsvm_format(test_path)
    if train_path is not None:
        dataset["train"]["x"] = train["x"]
        dataset["train"]["y"] = train["y"]
    if test_path is not None:
        dataset["test"]["x"] = test["x"]
        dataset["test"]["y"] = test["y"]

    # load labels
    if label_path is not None:
        logging.info(f"Load labels from {label_path}.")
        with open(label_path) as fp:
            dataset["classes"] = sorted([c.strip() for c in fp.readlines()])

    return dict(dataset)
