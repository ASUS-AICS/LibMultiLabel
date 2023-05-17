from __future__ import annotations

import copy
import logging
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

__all__ = ["Preprocessor"]


class Preprocessor:
    """Preprocessor is used to preprocess input data in LibSVM or LibMultiLabel formats.
    The same Preprocessor has to be used for both training and testing data;
    see save_pipeline and load_pipeline for more details.
    """

    tfidf_params_constraints = {
        "strip_accents",
        "stop_words",
        "ngram_range",
        "max_df",
        "min_df",
        "max_features",
        "vocabulary",
    }

    def __init__(self, include_test_labels: bool = False, remove_no_label_data: bool = False, tfidf_params: dict = {}):
        """Initializes the preprocessor.

        Args:
            include_test_labels (bool, optional): Whether to include labels in the test dataset. Defaults to False.
            remove_no_label_data (bool, optional): Whether to remove training instances that have no labels. Defaults to False.
            tfidf_params (dict, optional): A selected group of parameters for sklearn.TfidfVectorizer. Available arguments are
                strip_accents : {‘ascii’, ‘unicode’} or callable, default=None
                    Remove accents and perform other character normalization during the preprocessing step. ‘ascii’ is
                    a fast method that only works on characters that have a direct ASCII mapping. ‘unicode’ is a
                    slightly slower method that works on any characters. None (default) does nothing. Defaults to {}.

                stop_words : {‘english’}, list, default=None
                    If a string, it is passed to _check_stop_list and the appropriate stop list is returned. ‘english’
                    is currently the only supported string value. There are several known issues with ‘english’ and you
                    should consider an alternative (see Using stop words).

                    If a list, that list is assumed to contain stop words, all of which will be removed from the
                    resulting tokens. Only applies if analyzer == 'word'.

                    If None, no stop words will be used. In this case, setting max_df to a higher value, such as in the
                    range (0.7, 1.0), can automatically detect and filter stop words based on intra corpus document
                    frequency of terms.

                ngram_range : tuple (min_n, max_n), default=(1, 1)
                    The lower and upper boundary of the range of n-values for different n-grams to be extracted. All
                    values of n such that min_n <= n <= max_n will be used. For example an ngram_range of (1, 1) means
                    only unigrams, (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams. Only applies if
                    analyzer is not callable.

                max_df: float or int, default=1.0
                    When building the vocabulary ignore terms that have a document frequency strictly higher than the
                    given threshold (corpus-specific stop words). If float in range [0.0, 1.0], the parameter represents
                    a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not
                    None.

                min_df: float or int, default=1
                    When building the vocabulary ignore terms that have a document frequency strictly lower than the
                    given threshold. This value is also called cut-off in the literature. If float in range of
                    [0.0, 1.0], the parameter represents a proportion of documents, integer absolute counts. This
                    parameter is ignored if vocabulary is not None.

                max_features : int, default=None
                    If not None, build a vocabulary that only consider the top max_features ordered by term frequency
                    across the corpus. Otherwise, all features are used.

                    This parameter is ignored if vocabulary is not None.

                vocabulary : Mapping or iterable, default=None
                    Either a Mapping (e.g., a dict) where keys are terms and values are indices in the feature matrix,
                    or an iterable over terms. If not given, a vocabulary is determined from the input documents.
        """
        self.include_test_labels = include_test_labels
        self.remove_no_label_data = remove_no_label_data
        if not set(tfidf_params.keys()).issubset(self.tfidf_params_constraints):
            logging.warning(
                "Some of the parameters in tfidf_params are not supported. " "Supported parameters are: %s",
                self.tfidf_params_constraints,
            )
        self.tfidf_params = tfidf_params
        self.data_format = None
        self.vectorizer = None
        self.binarizer = None
        self.label_mapping = None

    def fit(self, dataset):
        """Fit the preprocessor according to the training and test datasets, and pre-defined labels if given.

        Parameters
        ----------
        dataset : The training and test datasets along with possibly pre-defined labels with keys 'train', 'test', and
            "labels" respectively. The dataset must have keys 'x' for input features, and 'y' for actual labels. It also
            contains 'data_format' to indicate the data format used.

        Returns
        -------
        self : object
            An instance of the fitted preprocessor.
        """
        self.data_format = dataset["data_format"]
        # learn vocabulary and idf from training dataset
        if dataset["data_format"] in {"txt", "dataframe"}:
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

    def transform(self, dataset):
        """Convert x and y in the training and test datasets according to the fitted preprocessor.

        Args:
            dataset : The training and test datasets along with labels with keys 'train', 'test', and labels respectively.
            The dataset has keys 'x' for input features and 'y' for labels. It also contains 'data_format' to indicate
            the data format used.

        Returns:
            dict[str, dict[str, sparse.csr_matrix]]: The transformed dataset.
        """
        if self.binarizer is None:
            raise AttributeError("Preprecessor has not been fitted.")

        dataset_t = defaultdict(dict)
        dataset_t["data_format"] = dataset["data_format"]
        if "classes" in dataset:
            dataset_t["classes"] = dataset["classes"]
        # transform a collection of raw text to a matrix of TF-IDF features
        if {self.data_format, dataset["data_format"]}.issubset({"txt", "dataframe"}):
            try:
                if "train" in dataset:
                    dataset_t["train"]["x"] = self.vectorizer.transform(dataset["train"]["x"])
                if "test" in dataset:
                    dataset_t["test"]["x"] = self.vectorizer.transform(dataset["test"]["x"])
            except AttributeError:
                raise AttributeError("Tfidf vectorizer has not been fitted.")

        # transform a collection of raw labels to a binary matrix
        if "train" in dataset:
            dataset_t["train"]["y"] = self.binarizer.transform(dataset["train"]["y"]).astype("d")
        if "test" in dataset:
            dataset_t["test"]["y"] = self.binarizer.transform(dataset["test"]["y"]).astype("d")

        # remove data points with no labels
        if "train" in dataset_t:
            num_labels = dataset_t["train"]["y"].getnnz(axis=1)
            num_no_label_data = np.count_nonzero(num_labels == 0)
            if num_no_label_data > 0:
                if self.remove_no_label_data:
                    logging.info(
                        f"Remove {num_no_label_data} instances that have no labels in the dataset.",
                        extra={"collect": True},
                    )
                    dataset_t["train"]["x"] = dataset_t["train"]["x"][num_labels > 0]
                    dataset_t["train"]["y"] = dataset_t["train"]["y"][num_labels > 0]
                else:
                    logging.info(
                        f"Keep {num_no_label_data} instances that have no labels in the dataset.",
                        extra={"collect": True},
                    )

        return dict(dataset_t)

    def fit_transform(self, dataset):
        """Fit the preprocessor according to the training and test datasets, and pre-defined labels if given.
        Then Convert x and y in the training and test datasets according to the fitted preprocessor.

        Args:
            dataset : The training and test datasets along with labels with keys 'train', 'test', and labels respectively.
            The dataset has keys 'x' for input features and 'y' for labels. It also contains 'data_format' to indicate
            the data format used.

        Returns:
            dict[str, dict[str, sparse.csr_matrix]]: The transformed dataset.
        """
        return self.fit(dataset).transform(dataset)
