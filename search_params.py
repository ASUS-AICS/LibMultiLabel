import argparse
import os
import time
import yaml

from datetime import datetime
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch

import data_utils
from evaluate import evaluate
from model import Model
from main import init_env
from utils import ArgDict


def init_model_config(config_path):
    with open(config_path) as fp:
        args = yaml.load(fp, Loader=yaml.SafeLoader)
    model_config = ArgDict(args)
    model_config = init_env(model_config)

    # set relative path to absolute path (_path, _file, _dir)
    for k, v in model_config.items():
        if isinstance(v, str) and (os.path.isfile(v) or os.path.isdir(v)):
            model_config[k] = os.path.abspath(v)
    return model_config


def training_function(config):
    # update hyperparameters to ray's
    model_config = config['model_config']
    model_config.learning_rate = config['tune_learning_rate']
    model_config.dropout = config['tune_dropout']
    model_config.num_filter_per_size = config['tune_num_filter_per_size']
    model_config.filter_sizes = config['tune_filter_sizes']

    datasets = data_utils.load_datasets(model_config)
    word_dict = data_utils.load_or_build_text_dict(model_config, datasets['train'])
    classes = data_utils.load_or_build_label(model_config, datasets)

    model = Model(model_config, word_dict, classes)
    model.train(datasets['train'], datasets['val'])
    model.load_best()

    # return best eval metric
    dev_loader = data_utils.get_dataset_loader(model_config, datasets['dev'], model.word_dict, model.classes, train=False)
    metrics = evaluate(model_config, model, dev_loader, split='dev', dump=True)
    tune.report(pak=metrics[model_config['val_metric']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', help='Path to configuration file (default: %(default)s). Please specify a config with all arguments in LibMultiLabel/main.py::get_config.')
    parser.add_argument('--cpu_count', type=int, default=1, help='Number of CPU (default: %(default)s)')
    parser.add_argument('--gpu_count', type=int, default=1, help='Number of GPU (default: %(default)s)')
    parser.add_argument('--local_dir', default=os.getcwd(), help='Directory to save training results (default: %(default)s)')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of running samples (default: %(default)s)')
    parser.add_argument('--mode', default='max', choices=['min', 'max'], help='Determines whether objective is minimizing or maximizing the metric attribute. (default: %(default)s)')
    args = vars(parser.parse_args())

    config = {
        "tune_learning_rate": tune.choice([0.001, 0.003, 0.0001, 0.0003]),
        "tune_dropout": tune.choice([0.2, 0.4, 0.6, 0.8]),
        "tune_num_filter_per_size": tune.choice([(50 + 100*i) for i in range(6)]),
        "tune_filter_sizes": tune.choice([2, 4, 6, 8, 10]),
        "model_config": init_model_config(args['config'])
    }

    # run tune analysis
    tune.run(
        training_function,
        # search_alg=config['algo'],
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
