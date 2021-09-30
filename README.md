# LibMultiLabel — a Library for Multi-label Text Classification

LibMultiLabel is a simple tool with the following functionalities.

- end-to-end services from raw texts to final evaluation/analysis
- support for common neural network architectures and linear classsifiers
- easy hyper-parameter selection

This is an on-going development so many improvements are still being made. Comments are very welcome.

## Environments and Installation
- Python: 3.7+
- CUDA: 10.2 (if GPU used)
- Pytorch 1.8+

If you have a different version of CUDA, go to the [website](https://pytorch.org/) for the detail of PyTorch installation.

To install the latest development version, run:
```
pip3 install -r requirements.txt
```

## Table of Contents
- [LibMultiLabel — a Library for Multi-label Text Classification](#libmultilabel--a-library-for-multi-label-text-classification)
  - [Environments and Installation](#environments-and-installation)
  - [Table of Contents](#table-of-contents)
  - [Quick Start via an Example](#quick-start-via-an-example)
    - [Step 1. Data Preparation](#step-1-data-preparation)
    - [Step 2. Training and Prediction](#step-2-training-and-prediction)
  - [Usage](#usage)
  - [Data Format](#data-format)
    - [Examples of a training file:](#examples-of-a-training-file)
    - [Examples of a test file:](#examples-of-a-test-file)
  - [Training and Prediction for Neural Networks](#training-and-prediction-for-neural-networks)
    - [Training](#training)
    - [Prediction](#prediction)
  - [Training and Prediction for Linear Classifiers](#training-and-prediction-for-linear-classifiers)
    - [Training](#training-1)
    - [Prediction](#prediction-1)
  - [Hyperparameter Search for Neural Networks](#hyperparameter-search-for-neural-networks)

## Quick Start via an Example
### Step 1. Data Preparation
- Create a data sub-directory within `LibMultiLabel` and go to this sub-directory.
```sh
mkdir -p data/rcv1
cd data/rcv1
```
- Download the `rcv1` dataset from [LIBSVM data sets](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets) by the following commands.

```sh
wget -O train.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_train.txt.bz2
wget -O test.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_test.txt.bz2
```
- Uncompress data files and change the directory back to `LibMultiLabel`.
```sh
bzip2 -d *.bz2
cd ../..
```

### Step 2. Training and Prediction
Train a cnn model and predict the test set by an example config. Use `--cpu` to run the program on the cpu.
```
python3 main.py --config example_config/rcv1/kim_cnn.yml
```

## Usage
The LibMultiLabel toolkit uses a yaml file to configure the dataset and the training process. At a high level, the config file is split into four parts:

- **data**: the data config consists of data paths that tell where to place the datasets, pre-trained word embeddings, and vocabularies.

- **model**: the model config defines the parameters that are related to the network definition (i.e., model name).

- **train**: the train config specifies the hyperparameters (i.e., batch size, learning rate, etc.) used when training a model.

- **eval**: the eval config decides metrics monitored or reported during the evaluation process.


For more information, run `--help` or check [the example configs](./example_config).
```
python3 main.py --help
```
Each parameter can also be specified through the command line.

## Data Format

Put labels and texts in the training, validation, and test set separately in `train.txt`, `valid.txt`, and `test.txt`, or specifying the path by the arguments  `train_path`, `valid_path`, and `test_path`. If validation set is not provided, then the program internally splits the training set to two parts for training and validation. To specify the size of the validation set, use `val_size` in one of the following two ways:
- a ratio in [0,1] to indicate the percentage of training data allocated for validation
- an integer to indicate the number of training data allocated for validation

### Examples of a training file:
- one sample per line
- seperate ID, labels and texts by `<TAB>` (the ID column is optional)
- labels are split by spaces
- should not contain any `<TAB>` in each field

With ID column:
```
2286<TAB>E11 ECAT M11 M12 MCAT<TAB>recov recov recov recov excit ...
2287<TAB>C24 CCAT<TAB>uruguay uruguay compan compan compan ...
```

Without ID column:
```
E11 ECAT M11 M12 MCAT<TAB>recov recov recov recov excit ...
C24 CCAT<TAB>uruguay uruguay compan compan compan ...
```

### Examples of a test file:
In the test set, the labels are used to calculate accuracy or errors. If it's unknown, any string (even an empty one) is fine. For example,

With ID column:
```
2286<TAB><TAB>recov recov recov recov excit ...
2287<TAB><TAB>uruguay uruguay compan compan compan ...
```

Without ID column:
```
<TAB>recov recov recov recov excit ...
<TAB>uruguay uruguay compan compan compan ...
```

## Training and Prediction for Neural Networks
### Training
In the training procedure, you can build a model from scratch or start from some pre-obtained information.
```
python3 main.py --config CONFIG_PATH [--load_checkpoint CHECKPOINT_PATH] \
[--embed_file EMBED_NAME_OR_EMBED_PATH] [--vocab_file VOCAB_CSV_PATH]
```
- **config**: configure parameters in a yaml file. See the section [Usage](#Usage).

If a model was trained before by this package, the training procedure can start with it.

- **load_checkpoint**: specify the path to a pre-trained model.

To use your own word embeddings or vocabulary set, specify the following parameters:

- **embed_file**: choose one of the pretrained embeddings defined in [torchtext](https://pytorch.org/text/0.9.0/vocab.html#torchtext.vocab.Vocab.load_vectors) or specify the path to your word embeddings with each line containing a word followed by its vectors. Example:
    ```=
    the 0.04656 0.21318 -0.0074364 ...
    a -0.29712 0.094049 -0.096662 ...
    an -0.3206 0.43316 -0.086867 ...
    ```
- **vocab_file**: set the file path to a predefined vocabulary set that contains lines of words.
    ```=
    the
    a
    an
    ```
For the validation process in the training procedure, you can evaluate the model with a set of evaluation metrics. Set `monitor_metrics` to define what you want to print on the screen. The argument `val_metric` is the metric for picking the best model. Example:
```yaml
monitor_metrics: [P@1, P@3, P@5]
val_metric: P@1
```

If `test_path` is specified or `DATA_DIR/test.txt` exists, the model with the highest `val_metric` will be evaluated after training.

### Prediction
To deploy/evaluate a model (i.e., a pre-obtained checkpoint), you can predict a test set by the following command.
```
python3 main.py --eval --config CONFIG_PATH --load_checkpoint CHECKPOINT_PATH --test_path TEST_DATA_PATH --save_k_predictions K --predict_out_path PREDICT_OUT_PATH
```
- Use `--save_k_predictions` to save the top K predictions for each instance in the test set. K=100 if not specified.
- Use `--predict_out_path` to specify the file for storing the predicted top-K labels/scores.


## Training and Prediction for Linear Classifiers
### Training
For training, use
```
python3 main.py --config CONFIG_PATH --linear [--liblinear_options LIBLINEAR_OPTIONS]
```
- **config**: configure parameters in a yaml file. See the section [Usage](#Usage).

The linear classifiers are based on LIBLINEAR, and its options may be specified.

- **liblinear_options**: An [option string for LIBLINEAR](https://github.com/cjlin1/liblinear).

### Prediction
To deploy/evaluate a model, you can predict a test set by the following command.

```
python3 main.py --eval --config CONFIG_PATH --linear --load_checkpoint CHECKPOINT_PATH
```


## Hyperparameter Search for Neural Networks
Parameter selection is known to be extremely important in machine learning practice; see a powerful reminder in "[this paper](https://www.csie.ntu.edu.tw/~cjlin/papers/parameter_selection/acl2021_parameter_selection.pdf)". Here we leverage [Ray Tune](https://docs.ray.io/en/master/tune/index.html), which is a python library for hyperparameter tuning, to select parameters. Due to the dependency of Ray Tune, first make sure your python version is not greater than 3.8. Then, install the related packages with:
```
pip3 install -Ur requirements_parameter_search.txt
```
We provide a program `search_params.py` to demonstrate how to run LibMultiLabel with Ray Tune. An example is as follows.
```
python3 search_params.py --config example_config/rcv1/cnn_tune.yml --search_alg basic_variant
```

- **config**: configure *all* parameters in a yaml file. You can define a continuous, a discrete, or other types of search space (see a list [here](https://docs.ray.io/en/master/tune/api_docs/search_space.html#tune-sample-docs)). An example of configuring the parameters is presented as follows:
```yaml
dropout: ['grid_search', [0.2, 0.4, 0.6, 0.8]] # grid search
num_filter_per_size: ['choice', [350, 450, 550]] # discrete
learning_rate: ['uniform', 0.2, 0.8] # continuous
activation: tanh # not for hyperparameter search
```
- **search_alg**: specify a search algorithm considered in [Ray Tune](https://docs.ray.io/en/master/tune/api_docs/suggestion.html). We support basic_variant (e.g., grid/random), bayesopt, and optuna. You can also define `search_alg` in the config file. For example, if you want to run grid search over `learning_rate`, the config is like this:
```yaml
search_alg: basic_variant
learning_rate: ['grid_search', [0.2, 0.4, 0.6, 0.8]]
```
