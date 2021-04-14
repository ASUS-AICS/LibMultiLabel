import argparse
from datetime import datetime
from pathlib import Path

import os
import torch
import yaml
import numpy as np

import data_utils
from model import Model
from utils import log
from utils.utils import ArgDict
from evaluate import evaluate, FewShotMetrics


def get_config():
    parser = argparse.ArgumentParser(
        add_help=False,
        description='multi-label learning for text classification')

    # load params from config file
    parser.add_argument('-c', '--config', help='Path to configuration file')
    args, _ = parser.parse_known_args()
    config = {}
    if args.config:
        with open(args.config) as fp:
            config = yaml.load(fp, Loader=yaml.SafeLoader)

    # path / directory
    parser.add_argument('--data_dir', default='./data', help='The directory to load data (default: %(default)s)')
    parser.add_argument('--result_dir', default='./runs', help='The directory to save checkpoints and logs (default: %(default)s)')

    # data
    parser.add_argument('--data_name', default='rcv1', help='Dataset name (default: %(default)s)')
    parser.add_argument('--dev_size', type=float, default=0.2, help='Training-validation split: a ratio in [0, 1] or an integer for the size of the validation set (default: %(default)s).')
    parser.add_argument('--min_vocab_freq', type=int, default=1, help='The minimum frequency needed to include a token in the vocabulary (default: %(default)s)')
    parser.add_argument('--max_seq_length', type=int, default=500, help='The maximum number of tokens of a sample (default: %(default)s)')
    parser.add_argument('--vocab_label_map', help='Path to a file storing vocabulary and label mappings (default: %(default)s)')

    # train
    parser.add_argument('--seed', type=int, help='Random seed (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train (default: %(default)s)')
    parser.add_argument('--batch_size', type=int, default=16, help='Size of training batches (default: %(default)s)')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], help='Optimizer: SGD or Adam (default: %(default)s)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer (default: %(default)s)')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay factor (default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum factor for SGD only (default: %(default)s)')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait for improvement before early stopping (default: %(default)s)')

    # model
    parser.add_argument('--model_name', default='cnn', choices=['caml', 'cnn'], help='Model to be used (default: %(default)s)')
    # default
    parser.add_argument('--num_filter_maps', type=int, default=128, help='Number of filters in convolutional layers (default: %(default)s)')
    parser.add_argument('--filter_size', type=int, default=4, help='Size of convolutional filter (default: %(default)s)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Optional specification of dropout (default: %(default)s)')

    # eval
    parser.add_argument('--eval_batch_size', type=int, default=256, help='Size of evaluating batches (default: %(default)s)')
    parser.add_argument('--metrics_thresholds', type=float, nargs='+', default=[0.5], help='Thresholds to monitor for metrics (default: %(default)s)')
    parser.add_argument('--monitor_metrics', nargs='+', default=['P@1', 'P@3', 'P@5'], help='Metrics to monitor while validating (default: %(default)s)')
    parser.add_argument('--val_metric', default='P@1', help='The metric to monitor for early stopping (default: %(default)s)')

    # pretrained vocab / embeddings
    parser.add_argument('--vocab_file', type=str, help='Path to a file holding vocabuaries (default: %(default)s)')
    parser.add_argument('--embed_file', type=str, help='Path to a file holding pre-trained embeddings (default: %(default)s)')

    # others
    parser.add_argument('--cpu', action='store_true', help='Disable CUDA')
    parser.add_argument('--display_iter', type=int, default=100, help='Log state after every n epochs (default: %(default)s)')
    parser.add_argument('--data_workers', type=int, default=1, help='Use multi-cpu core for data pre-processing (default: %(default)s)')
    parser.add_argument('--eval', action='store_true', help='Only run evaluation on the test set (default: %(default)s)')
    parser.add_argument('--load_checkpoint', help='The checkpoint to warm-up with (default: %(default)s)')
    parser.add_argument('-h', '--help', action='help')

    parser.set_defaults(**config)
    args = parser.parse_args()
    config = ArgDict(vars(args))
    return config


def init_env(config):
    # set a debug environment variable CUBLAS_WORKSPACE_CONFIG to ":16:8" (may limit overall performance) or ":4096:8" (will increase library footprint in GPU memory by approximately 24MiB).
    # https://docs.nvidia.com/cuda/cublas/index.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    if config.seed is not None:
        if config.seed >= 0:
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            torch.set_deterministic(True)
            torch.backends.cudnn.benchmark = False
        else:
            log.warning(f'the random seed should be a non-negative integer')

    config.device = None
    if not config.cpu and torch.cuda.is_available():
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy('file_system')
    log.info(f'Using device: {config.device}')

    config.run_name = '{}_{}_{}'.format(
        config.data_name,
        Path(config.config).stem if config.config else config.model_name,
        datetime.now().strftime('%Y%m%d%H%M%S'),
    )
    log.info(f'Run name: {config.run_name}')
    return config


@log.enter('main')
def main():
    config = get_config()
    config = init_env(config)
    datasets = data_utils.load_dataset(config)

    eval_metric = FewShotMetrics(config, datasets)
    if config.eval:
        model = Model.load(config, config.load_checkpoint)
    else:
        if config.load_checkpoint:
            model = Model.load(config, config.load_checkpoint)
        else:
            word_dict = datasets['train'].word_dict
            classes = datasets['train'].classes
            model = Model(config, word_dict, classes)
        model.train(datasets['train'], datasets['dev'], eval_metric)
        model.load_best()
    test_loader = data_utils.get_dataset_loader(config, datasets['test'], train=False)
    evaluate(config, model, test_loader, eval_metric, split='test', dump=not config.eval)


if __name__ == '__main__':
    main()
