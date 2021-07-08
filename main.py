import argparse
import logging
import os
import yaml
from datetime import datetime
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.parsing import AttributeDict

from libmultilabel import data_utils
from libmultilabel.model import Model
from libmultilabel.utils import Timer, dump_log, init_device, set_seed


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
    parser.add_argument('--metrics_threshold', type=float, default=0.5,
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


def save_predictions(trainer, model, dataloader, predict_out_path):
    batch_predictions = trainer.predict(model, dataloaders=dataloader)
    pred_labels = np.vstack([batch['top_k_pred'] for batch in batch_predictions])
    pred_scores = np.vstack([batch['top_k_pred_scores'] for batch in batch_predictions])
    with open(predict_out_path, 'w') as fp:
        for pred_label, pred_score in zip(pred_labels, pred_scores):
            out_str = ' '.join([f'{label}:{score:.4}' for label, score in zip(pred_label, pred_score)])
            fp.write(out_str+'\n')
    logging.info(f'Saved predictions to: {predict_out_path}')


def main():
    config = get_config()
    log_level = logging.WARNING if config.silent else logging.INFO
    logging.basicConfig(
        level=log_level, format='%(asctime)s %(levelname)s:%(message)s')
    set_seed(seed=config.seed)
    config.device = init_device(use_cpu=config.cpu)

    config.run_name = '{}_{}_{}'.format(
        config.data_name,
        Path(config.config).stem if config.config else config.model_name,
        datetime.now().strftime('%Y%m%d%H%M%S'),
    )
    logging.info(f'Run name: {config.run_name}')

    datasets = data_utils.load_datasets(config)

    checkpoint_dir = os.path.join(config.result_dir, config.run_name)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir,
                                          filename='best_model',
                                          save_last=True, save_top_k=1,
                                          monitor=config.val_metric, mode='max')
    earlystopping_callback = EarlyStopping(patience=config.patience,
                                           monitor=config.val_metric, mode='max')

    trainer = pl.Trainer(logger=False,
                         num_sanity_val_steps=0,
                         gpus=0 if config.cpu else 1,
                         progress_bar_refresh_rate=0 if config.silent else 1,
                         max_epochs=config.epochs,
                         callbacks=[checkpoint_callback, earlystopping_callback])

    if config.eval:
        model = Model.load_from_checkpoint(config.checkpoint_path)
        model.config = config
    else:
        if config.checkpoint_path:
            model = Model.load_from_checkpoint(config.checkpoint_path)
            model.config = config
        else:
            word_dict = data_utils.load_or_build_text_dict(
                config, datasets['train'])
            classes = data_utils.load_or_build_label(config, datasets)
            model = Model(config, word_dict, classes)

        train_loader = data_utils.get_dataset_loader(
            model.config, datasets['train'], model.word_dict, model.classes,
            shuffle=model.config.shuffle, train=True)
        val_loader = data_utils.get_dataset_loader(
            model.config, datasets['val'], model.word_dict, model.classes, train=False)

        trainer.fit(model, train_loader, val_loader)

        logging.info(f'Loading best model from `{checkpoint_callback.best_model_path}`...')
        model = Model.load_from_checkpoint(checkpoint_callback.best_model_path)

    if 'test' in datasets:
        test_loader = data_utils.get_dataset_loader(
            model.config, datasets['test'], model.word_dict, model.classes, train=False)
        trainer.test(model, test_dataloaders=test_loader)
        if config.save_k_predictions > 0:
            if not config.predict_out_path:
                config.predict_out_path = os.path.join(checkpoint_dir, 'predictions.txt')
            save_predictions(trainer, model, test_loader, config.predict_out_path)


if __name__ == '__main__':
    wall_time = Timer()
    main()
    print(f'Wall time: {wall_time.time():.2f} (s)')
