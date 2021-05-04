import argparse
import os
import time
import torch
import yaml

from datetime import datetime
from ray import tune

import data_utils
from evaluate import evaluate
from main import init_env
from model import Model
from utils import ArgDict


def init_model_config(config_path):
    with open(config_path) as fp:
        args = yaml.load(fp, Loader=yaml.SafeLoader)

    # set relative path to absolute path (_path, _file, _dir)
    for k, v in args.items():
        if isinstance(v, str) and (os.path.isfile(v) or os.path.isdir(v)):
            args[k] = os.path.abspath(v)

    model_config = ArgDict(args)
    model_config = init_env(model_config)
    return model_config.__dict__


def get_search_algorithm(search_alg):
    """Specify a search algorithm. You should pip install this search algorithm first.
    See more details here: https://docs.ray.io/en/master/tune/api_docs/suggestion.html"""
    if search_alg == 'optuna':
        from ray.tune.suggest.optuna import OptunaSearch
        return OptunaSearch()
    elif search_alg == 'hyperopt':
        from ray.tune.suggest.hyperopt import HyperOptSearch
        return HyperOptSearch()
    print('No search algorithm is found, run BasicVariantGenerator().')


def training_function(config):
    config = ArgDict(config)
    datasets = data_utils.load_datasets(config)
    word_dict = data_utils.load_or_build_text_dict(config, datasets['train'])
    classes = data_utils.load_or_build_label(config, datasets)

    model = Model(config, word_dict, classes)
    model.train(datasets['train'], datasets['val'])
    model.load_best()

    # return best eval metric
    dev_loader = data_utils.get_dataset_loader(config, datasets['dev'], model.word_dict, model.classes, train=False)
    metrics = evaluate(config, model, dev_loader, split='dev', dump=True)
    tune.report(pak=metrics[config['val_metric']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', help='Path to configuration file (default: %(default)s). Please specify a config with all arguments in LibMultiLabel/main.py::get_config.')
    parser.add_argument('--cpu_count', type=int, default=1, help='Number of CPU (default: %(default)s)')
    parser.add_argument('--gpu_count', type=int, default=1, help='Number of GPU (default: %(default)s)')
    parser.add_argument('--local_dir', default=os.getcwd(), help='Directory to save training results (default: %(default)s)')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of running samples (default: %(default)s)')
    parser.add_argument('--mode', default='max', choices=['min', 'max'], help='Determines whether objective is minimizing or maximizing the metric attribute. (default: %(default)s)')
    parser.add_argument('--search_alg', default=None, help='Number of running samples (default: %(default)s)')
    args = vars(parser.parse_args())

    config = init_model_config(args['config'])
    config.learning_rate = tune.choice([0.001, 0.003, 0.0001, 0.0003])
    config.dropout = tune.choice([0.2, 0.4, 0.6, 0.8])
    config.num_filter_per_size = tune.choice([(50 + 100*i) for i in range(6)])
    config.filter_sizes = [tune.choice([2, 4, 6, 8, 10])]

    # run tune analysis
    """If no search algorithm is specified, the defautl search algorighm is BasicVariantGenerator.
    https://docs.ray.io/en/master/tune/api_docs/suggestion.html#tune-basicvariant
    """
    tune.run(
        training_function,
        search_alg=get_search_algorithm(args['search_alg']),
        local_dir=args['local_dir'],
        metric='pak',
        mode=args['mode'],
        num_samples=args['num_samples'],
        resources_per_trial={
            'cpu': args['cpu_count'], 'gpu': args['gpu_count']},
        config=config)


# calculate wall time.
wall_time_start = time.time()
main()
print(f"\nWall time: {time.time()-wall_time_start}")
