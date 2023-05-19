from __future__ import annotations

import logging
from collections import defaultdict
from scipy import sparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

__all__ = ["Preprocessor"]


class Preprocessor:
    """Preprocessor is used to preprocess input data in LibSVM or LibMultiLabel formats.
    The same Preprocessor has to be used for both training and test datasets;
    see save_pipeline and load_pipeline for more details.
    """

    def __init__(
        self, include_test_labels: bool = False, remove_no_label_data: bool = False, tfidf_params: dict[str, str] = {}
    ):
        """Initializes the preprocessor.

        Args:
            include_test_labels (bool, optional): Whether to include labels in the test dataset. Defaults to False.
            remove_no_label_data (bool, optional): Whether to remove training instances that have no labels.
                Defaults to False.
            tfidf_params (dict[str, str], optional): A set of parameters for sklearn.TfidfVectorizer. If empty, default
                parameters will be used.
        """
        self.include_test_labels = include_test_labels
        self.remove_no_label_data = remove_no_label_data
        self.tfidf_params = tfidf_params
        self.data_format = None
        self.vectorizer = None
        self.binarizer = None
        self.label_mapping = None
        self.is_fitted = False

    def fit(self, dataset: dict[str, dict[str, sparse.csr_matrix | list[list[int]] | list[str]]]) -> Preprocessor:
        """Fit the preprocessor according to the training and test datasets, and pre-defined labels if given.

        Args:
            dataset (dict[str, dict[str, sparse.csr_matrix | list[list[int]] | list[str]]]):
                The training and test datasets along with possibly pre-defined labels with keys 'train', 'test', and
                "labels" respectively. The dataset must have keys 'x' for input features, and 'y' for actual labels. It
                also contains 'data_format' to indicate the data format used.

        Returns:
            Preprocessor: An instance of the fitted preprocessor.
        """
        if self.is_fitted:
            raise AttributeError("Preprocessor has been fitted. An instance of Preprocessor can only been fitted once.")
        self.is_fitted = True

        self.data_format = dataset["data_format"]
        # learn vocabulary and idf from training dataset
        if self.data_format in {"txt", "dataframe"}:
            self.vectorizer = TfidfVectorizer(**self.tfidf_params)
            self.vectorizer.fit(dataset["train"]["x"])

        # learn label mapping from training and test datasets
        self.binarizer = MultiLabelBinarizer(classes=dataset.get("classes"), sparse_output=True)
        if not self.include_test_labels:
            self.binarizer.fit(dataset["train"]["y"])
        else:
            self.binarizer.fit(dataset["train"]["y"] + dataset["test"]["y"])
        self.label_mapping = self.binarizer.classes_
        return self

    def transform(self, dataset: dict[str, dict[str, sparse.csr_matrix | list[list[int]] | list[str]]]):
        """Convert x and y in the training and test datasets according to the fitted preprocessor.

        Args:
            dataset (dict[str, dict[str, sparse.csr_matrix | list[list[int]] | list[str]]]):
                The training and test datasets along with labels with keys 'train', 'test', and labels respectively.
                The dataset has keys 'x' for input features and 'y' for labels. It also contains 'data_format' to indicate
                the data format used.

        Returns:
            dict[str, dict[str, sparse.csr_matrix]]: The transformed dataset.
        """
        if not self.is_fitted:
            raise AttributeError("Preprecessor has not been fitted.")

        # "tf" indicates transformed
        dataset_tf = defaultdict(dict)
        dataset_tf["data_format"] = dataset["data_format"]
        if "classes" in dataset:
            dataset_tf["classes"] = dataset["classes"]
        # transform a collection of raw text to a matrix of TF-IDF features
        if {self.data_format, dataset["data_format"]}.issubset({"txt", "dataframe"}):
            try:
                if "train" in dataset:
                    dataset_tf["train"]["x"] = self.vectorizer.transform(dataset["train"]["x"])
                if "test" in dataset:
                    dataset_tf["test"]["x"] = self.vectorizer.transform(dataset["test"]["x"])
            except AttributeError:
                raise AttributeError("Tfidf vectorizer has not been fitted.")

        # transform a collection of raw labels to a binary matrix
        if "train" in dataset:
            dataset_tf["train"]["y"] = self.binarizer.transform(dataset["train"]["y"]).astype("d")
        if "test" in dataset:
            dataset_tf["test"]["y"] = self.binarizer.transform(dataset["test"]["y"]).astype("d")

        # remove data points with no labels
        if "train" in dataset_tf:
            num_labels = dataset_tf["train"]["y"].getnnz(axis=1)
            num_no_label_data = np.count_nonzero(num_labels == 0)
            if num_no_label_data > 0:
                if self.remove_no_label_data:
                    logging.info(
                        f"Remove {num_no_label_data} instances that have no labels in the dataset.",
                        extra={"collect": True},
                    )
                    dataset_tf["train"]["x"] = dataset_tf["train"]["x"][num_labels > 0]
                    dataset_tf["train"]["y"] = dataset_tf["train"]["y"][num_labels > 0]
                else:
                    logging.info(
                        f"Keep {num_no_label_data} instances that have no labels in the dataset.",
                        extra={"collect": True},
                    )

        return dict(dataset_tf)

    def fit_transform(self, dataset):
        """Fit the preprocessor according to the training and test datasets, and pre-defined labels if given.
        Then convert x and y in the training and test datasets according to the fitted preprocessor.

        Args:
            dataset (dict[str, dict[str, sparse.csr_matrix | list[list[int]] | list[str]]]):
                The training and test datasets along with labels with keys 'train', 'test', and labels respectively.
                The dataset has keys 'x' for input features and 'y' for labels. It also contains 'data_format' to
                indicate the data format used.

        Returns:
            dict[str, dict[str, sparse.csr_matrix]]: The transformed dataset.
        """
        return self.fit(dataset).transform(dataset)
