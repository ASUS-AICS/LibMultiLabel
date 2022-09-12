import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import yaml

from libmultilabel.common_utils import Timer, AttributeDict


def add_all_arguments(parser):
    # path / directory
    parser.add_argument('--result_dir', default='./runs',
                        help='The directory to save checkpoints and logs (default: %(default)s)')

    # data
    parser.add_argument('--data_name', default='unnamed',
                        help='Dataset name (default: %(default)s)')
    parser.add_argument('--training_file',
                        help='Path to training data (default: %(default)s)')
    parser.add_argument('--val_file',
                        help='Path to validation data (default: %(default)s)')
    parser.add_argument('--test_file',
                        help='Path to test data (default: %(default)s')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Training-validation split: a ratio in [0, 1] or an integer for the size of the validation set (default: %(default)s).')
    parser.add_argument('--min_vocab_freq', type=int, default=1,
                        help='The minimum frequency needed to include a token in the vocabulary (default: %(default)s)')
    parser.add_argument('--max_seq_length', type=int, default=500,
                        help='The maximum number of tokens of a sample (default: %(default)s)')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Whether to shuffle training data before each epoch (default: %(default)s)')
    parser.add_argument('--merge_train_val', action='store_true',
                        help='Whether to merge the training and validation data. (default: %(default)s)')
    parser.add_argument('--include_test_labels', action='store_true',
                        help='Whether to include labels in the test dataset. (default: %(default)s)')
    parser.add_argument('--remove_no_label_data', action='store_true',
                        help='Whether to remove training and validation instances that have no labels. (default: %(default)s)')
    parser.add_argument('--add_special_tokens', action='store_true',
                        help='Whether to add the special tokens for inputs of the transformer-based language model. (default: %(default)s)')

    # train
    parser.add_argument('--seed', type=int,
                        help='Random seed (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='The number of epochs to train (default: %(default)s)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Size of training batches (default: %(default)s)')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'adamw', 'adamax', 'sgd'],
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for optimizer (default: %(default)s)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay factor (default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum factor for SGD only (default: %(default)s)')
    parser.add_argument('--patience', type=int, default=5,
                        help='The number of epochs to wait for improvement before early stopping (default: %(default)s)')
    parser.add_argument('--normalize_embed', action='store_true',
                        help='Whether the embeddings of each word is normalized to a unit vector (default: %(default)s)')

    # model
    parser.add_argument('--model_name', default='KimCNN',
                        help='Model to be used (default: %(default)s)')
    parser.add_argument('--init_weight', default='kaiming_uniform',
                        help='Weight initialization to be used (default: %(default)s)')

    # eval
    parser.add_argument('--eval_batch_size', type=int, default=256,
                        help='Size of evaluating batches (default: %(default)s)')
    parser.add_argument('--metric_threshold', type=float, default=0.5,
                        help='Thresholds to monitor for metrics (default: %(default)s)')
    parser.add_argument('--monitor_metrics', nargs='+', default=['P@1', 'P@3', 'P@5'],
                        help='Metrics to monitor while validating (default: %(default)s)')
    parser.add_argument('--val_metric', default='P@1',
                        help='The metric to monitor for early stopping (default: %(default)s)')

    # pretrained vocab / embeddings
    parser.add_argument('--vocab_file', type=str,
                        help='Path to a file holding vocabuaries (default: %(default)s)')
    parser.add_argument('--embed_file', type=str,
                        help='Path to a file holding pre-trained embeddings (default: %(default)s)')
    parser.add_argument('--label_file', type=str,
                        help='Path to a file holding all labels (default: %(default)s)')

    # log
    parser.add_argument('--save_k_predictions', type=int, nargs='?', const=100, default=0,
                        help='Save top k predictions on test set. k=%(const)s if not specified. (default: %(default)s)')
    parser.add_argument('--predict_out_path',
                        help='Path to the an output file holding top k label results (default: %(default)s)')

    # auto-test
    parser.add_argument('--limit_train_batches', type=float, default=1.0,
                        help='Percentage of train dataset to use for auto-testing (default: %(default)s)')
    parser.add_argument('--limit_val_batches', type=float, default=1.0,
                        help='Percentage of validation dataset to use for auto-testing (default: %(default)s)')
    parser.add_argument('--limit_test_batches', type=float, default=1.0,
                        help='Percentage of test dataset to use for auto-testing (default: %(default)s)')

    # others
    parser.add_argument('--cpu', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--silent', action='store_true',
                        help='Enable silent mode')
    parser.add_argument('--data_workers', type=int, default=4,
                        help='Use multi-cpu core for data pre-processing (default: %(default)s)')
    parser.add_argument('--embed_cache_dir', type=str,
                        help='For parameter search only: path to a directory for storing embeddings for multiple runs. (default: %(default)s)')
    parser.add_argument('--eval', action='store_true',
                        help='Only run evaluation on the test set (default: %(default)s)')
    parser.add_argument('--checkpoint_path',
                        help='The checkpoint to warm-up with (default: %(default)s)')

    # linear options
    parser.add_argument('--linear', action='store_true',
                        help='Train linear model')
    parser.add_argument('--data_format', type=str, default='txt',
                        help='\'svm\' for SVM format or \'txt\' for LibMultiLabel format (default: %(default)s)')
    parser.add_argument('--liblinear_options', type=str,
                        help='Options passed to liblinear (default: %(default)s)')
    parser.add_argument('--linear_technique', type=str, default='1vsrest',
                        choices=['1vsrest', 'thresholding', 'cost_sensitive', 'cost_sensitive_micro'],
                        help='Technique for linear classification (default: %(default)s)')

    parser.add_argument('-h', '--help', action='help',
                        help="If you are trying to specify network config such as dropout or activation, use a yaml file instead. "
                             "See example configs in example_config")


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

    add_all_arguments(parser)

    parser.set_defaults(**config)
    args = parser.parse_args()
    config = AttributeDict(vars(args))

    config.run_name = '{}_{}_{}'.format(
        config.data_name,
        Path(config.config).stem if config.config else config.model_name,
        datetime.now().strftime('%Y%m%d%H%M%S'),
    )
    config.checkpoint_dir = os.path.join(config.result_dir, config.run_name)
    config.log_path = os.path.join(config.checkpoint_dir, 'logs.json')
    config.predict_out_path = config.predict_out_path or os.path.join(
        config.checkpoint_dir, 'predictions.txt')

    return config


def check_config(config):
    """Check if the configuration has invalid arguments.

    Args:
        config (AttributeDict): Config of the experiment from `get_args`.
    """
    if config.model_name == 'XMLCNN' and config.seed is not None:
        raise ValueError("nn.AdaptiveMaxPool1d doesn't have a deterministic implementation but seed is"
                         "specified. Please do not specify seed.")

    if config.eval and config.test_file is None:
        raise ValueError('--eval is specified but there is no test data set')

    return None

def main():
    # Get config
    config = get_config()
    check_config(config)

    # Set up logger
    log_level = logging.WARNING if config.silent else logging.INFO
    logging.basicConfig(
        level=log_level, format='%(asctime)s %(levelname)s:%(message)s')

    logging.info(f'Run name: {config.run_name}')

    if config.linear:
        from linear_trainer import linear_run
        linear_run(config)
    else:
        from torch_trainer import TorchTrainer
        trainer = TorchTrainer(config)  # initialize trainer
        # train
        if not config.eval:
            trainer.train()
        # test
        if 'test' in trainer.datasets:
            trainer.test()


if __name__ == '__main__':
    wall_time = Timer()
    main()
    print(f'Wall time: {wall_time.time():.2f} (s)')
