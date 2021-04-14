# LibMultiLabel -- a Library for Multi-label Text Classification

LibMultiLabel is a simple tool with the following functionalities.

- end-to-end services from raw texts to final evaluation/analysis
- support of common network architectures for multi-label text classification

This is an on-going development so many improvements are still being made. Comments are very welcome.

## Environment
- Python: 3.6+
- CUDA: 10.2 (if GPU used)
- Pytorch 1.8+

## Table of Contents
- [Quick Start via an Example](#Quick-Start-via-an-Example)
- [Usage](#Usage)
- [Data Format](#Data-Format)
- [Training and Prediction](#Training-and-Prediction)

## Quick Start via an Example
### Step 1. Data Preparation
- Create the data directory.
```sh
mkdir data/rcv1
cd data/rcv1
```
- Download the `rcv1` dataset from [the LIBSVM website](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html#rcv1v2%20(topics;%20full%20sets)).

```sh
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/train_texts.txt.bz2
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/train_labels.txt.bz2
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/test_texts.txt.bz2
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/test_labels.txt.bz2
```
- Uncompress data and go back to the main directory.
```sh
bzip2 -d *.bz2
cd ../..
```

### Step 2. Training and Prediction
Train a cnn model and predict the test set by  an example config. Use `--cpu` to run the program on the cpu.
```
python main.py --config example_config/rcv1/cnn.yml
```

## Usage
The LibMultiLabel toolkit uses a yaml file to configure the dataset and the training process. At a high level, the config file is split into four parts:

- **data**: the data config consists of data paths that tell where to place the datasets, pre-trained word embeddings, and vocabularies.

- **model**: the model config defines the parameters that are related to the network definition (i.e., model name).

- **train**: the train config specifies the hyperparameters (i.e., batch size, learning rate, etc.) used when training a model.

- **eval**: the eval config decides metrics monitored or reported during the evaluation process.


For more information, run `--help` or check [the example configs](./example_config).
```
python main.py --help
```
Each parameter can also be specified through the command line.

## Data Format

Put texts and labels in the training and test set separately in `train_text.txt`, `test_text.txt`, `train_labels.txt`, and `test_labels.txt`. The program internally splits the training set to two parts for training and validation. To specify the size of the validation set, use `dev_size` in one of the following two ways:
- a ratio in [0,1] to indicate the percentage of training data allocated for validation
- an integer to indicate the number of training data allocated for validation

### Examples of a text file and a label file: 
- **texts**: one sample per line
```
recov recov recov recov excit ...
uruguay uruguay compan compan compan ...
```

- **labels**: one sample per line; labels are split by space
```
E11 ECAT M11 M12 MCAT
C24 CCAT
```

## Training and Prediction
### Training
In the training progress, you can build a model from scratch or start from some pre-obtained information. 
```
python main.py --config CONFIG_PATH [--load_checkpoint CHECKPOINT_PATH] \
[--embed_file EMBED_NAME_OR_EMBED_PATH] [--vocab_file VOCAB_CSV_PATH]
```
- **config**: configure parameters in a yaml file. See the section [Usage](#Usage).

If a model was trained before by this package, the training procedure can start with it.

- **load_checkpoint**: specify the path to a pre-trained model. 

To use your own word embeddings or vocabulary set, specify the following parameters:

- **embed_file**: choose one of the pretrained embeddings defined in [torchtext](https://pytorch.org/text/0.9.0/vocab.html#torchtext.vocab.Vocab.load_vectors) or specify the path to your word embeddings with each line containing a word followed by its vectors. Example:
    ```=
    team -0.17901678383350372 1.2405720949172974 ...
    language 0.8667483925819397 5.001194953918457 ...
    ```
- **vocab_file**: set the file path to a predefined vocabulary set that contains lines of words.


### Evaluation
In the evaluation progress, you can evaluate the model with a set of evaluation metrics. Set `monitor_metrics` to define what you want to print on the screen. `val_metric` is the metric for picking the best model. Example:
```yaml
monitor_metrics: [P@1, P@3, P@5]
val_metric: P@1
```
