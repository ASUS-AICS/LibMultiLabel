import argparse
import logging
import yaml

import numpy as np
from pytorch_lightning.utilities.parsing import AttributeDict

# from libmultilabel import linear
from torch_trainer import TorchTrainer
from libmultilabel.utils import Timer


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
    parser.add_argument('--data_dir', default='./data/rcv1',
                        help='The directory to load data (default: %(default)s)')
    parser.add_argument('--result_dir', default='./runs',
                        help='The directory to save checkpoints and logs (default: %(default)s)')

    # data
    parser.add_argument('--data_name', default='rcv1',
                        help='Dataset name (default: %(default)s)')
    parser.add_argument('--train_path',
                        help='Path to training data (default: [data_dir]/train.txt)')
    parser.add_argument('--val_path',
                        help='Path to validation data (default: [data_dir]/valid.txt)')
    parser.add_argument('--test_path',
                        help='Path to test data (default: [data_dir]/test.txt)')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Training-validation split: a ratio in [0, 1] or an integer for the size of the validation set (default: %(default)s).')
    parser.add_argument('--min_vocab_freq', type=int, default=1,
                        help='The minimum frequency needed to include a token in the vocabulary (default: %(default)s)')
    parser.add_argument('--max_seq_length', type=int, default=500,
                        help='The maximum number of tokens of a sample (default: %(default)s)')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Whether to shuffle training data before each epoch (default: %(default)s)')

    # train
    parser.add_argument('--seed', type=int,
                        help='Random seed (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of epochs to train (default: %(default)s)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Size of training batches (default: %(default)s)')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'],
                        help='Optimizer: SGD or Adam (default: %(default)s)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for optimizer (default: %(default)s)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay factor (default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum factor for SGD only (default: %(default)s)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs to wait for improvement before early stopping (default: %(default)s)')
    parser.add_argument('--normalize_embed', action='store_true',
                        help='Whether the embeddings of each word is normalized to a unit vector (default: %(default)s)')

    # model
    parser.add_argument('--model_name', default='KimCNN',
                        help='Model to be used (default: %(default)s)')
    parser.add_argument('--init_weight', default='kaiming_uniform',
                        help='Weight initialization to be used (default: %(default)s)')
    parser.add_argument('--activation', default='relu',
                        help='Activation function to be used (default: %(default)s)')
    parser.add_argument('--num_filter_per_size', type=int, default=128,
                        help='Number of filters in convolutional layers in each size (default: %(default)s)')
    parser.add_argument('--filter_sizes', type=int, nargs='+',
                        default=[4], help='Size of convolutional filters (default: %(default)s)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Optional specification of dropout (default: %(default)s)')
    parser.add_argument('--dropout2', type=float, default=0.2,
                        help='Optional specification of the second dropout (default: %(default)s)')
    parser.add_argument('--num_pool', type=int, default=1,
                        help='Number of pool for dynamic max-pooling (default: %(default)s)')

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
                        help='Path to the an output file holding top 100 label results (default: %(default)s)')

    # others
    parser.add_argument('--linear', action='store_true',
                        help='Train linear model')
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
    parser.add_argument('-h', '--help', action='help')

    parser.set_defaults(**config)
    args = parser.parse_args()
    config = AttributeDict(vars(args))
    return config


def check_config(config):
    """Check if the configuration has invalid arguments.

    Args:
        config (AttributeDict): Config of the experiment from `get_args`.
    """
    if config.model_name == 'XMLCNN' and config.seed is not None:
        raise ValueError("nn.AdaptiveMaxPool1d doesn't have a deterministic implementation but seed is"
                         "specified. Please do not specify seed.")


def save_predictions(trainer, model, dataloader, predict_out_path):
    batch_predictions = trainer.predict(model, dataloaders=dataloader)
    pred_labels = np.vstack([batch['top_k_pred'] for batch in batch_predictions])
    pred_scores = np.vstack([batch['top_k_pred_scores'] for batch in batch_predictions])
    with open(predict_out_path, 'w') as fp:
        for pred_label, pred_score in zip(pred_labels, pred_scores):
            out_str = ' '.join([f'{model.classes[label]}:{score:.4}' for label, score in zip(pred_label, pred_score)])
            fp.write(out_str+'\n')
    logging.info(f'Saved predictions to: {predict_out_path}')


def main():
    # Get config
    config = get_config()
    check_config(config)

    # Set up logger
    log_level = logging.WARNING if config.silent else logging.INFO
    logging.basicConfig(
        level=log_level, format='%(asctime)s %(levelname)s:%(message)s')

    if not config.linear:
        trainer = TorchTrainer(config) # initialize trainer

    # train
    if not config.eval:
        if config.linear:
            # y, x = linear.read_tfidf()
            # model = linear.train_1vsrest(y, x)
            pass
        else:
            trainer.train()
    # test
    if 'test' in trainer.datasets:
        if config.linear:
            # linear.predict_values(model, x)
            pass
        else:
            trainer.test()


if __name__ == '__main__':
    wall_time = Timer()
    main()
    print(f'Wall time: {wall_time.time():.2f} (s)')
