import argparse
import os
import time
import yaml

from datetime import datetime
from ray import tune
# from ray.tune.suggest.optuna import OptunaSearch

import data_utils
from evaluate import evaluate
from model import Model
from main import init_env
from utils import ArgDict


def objective(step, model_config):
    datasets = data_utils.load_datasets(model_config)
    word_dict = data_utils.load_or_build_text_dict(model_config, datasets['train'])
    classes = data_utils.load_or_build_label(model_config, datasets)

    model = Model(model_config, word_dict, classes)
    model.train(datasets['train'], datasets['val'])
    model.load_best()

    # return best eval metric
    dev_loader = data_utils.get_dataset_loader(model_config, datasets['dev'], model.word_dict, model.classes, train=False)
    metrics = evaluate(model_config, model, dev_loader, split='dev', dump=True)
    return metrics[model_config['val_metric']]


def init_model_config(config_path):
    with open(config_path) as fp:
        args = yaml.load(fp, Loader=yaml.SafeLoader)
    model_config = ArgDict(args)
    model_config = init_env(model_config)
    return model_config


def training_function(config):
    # Hyperparameters
    model_config = init_model_config(config["config_path"])
    model_config["learning_rate"] = config["learning_rate"]
    model_config["dropout"] = config["dropout"]
    model_config["num_filter_maps"] = config["num_filter_maps"]
    model_config["filter_size"] = config["filter_size"]

    for step in range(1):
        intermediate_score = objective(step, model_config)
        tune.report(pak=intermediate_score)


def run_random_sampling(config):
    analysis = tune.run(
        training_function,
        # search_alg=config['algo'],
        metric='pak',
        mode=config['mode'],
        num_samples=config['num_samples'],
        resources_per_trial={'cpu': 2, 'gpu': 1},
        config=config)

    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', help='Path to configuration file')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of running samples (default: %(default)s)')
    parser.add_argument('--mode', default='max', choices=['min', 'max'], help='Determines whether objective is minimizing or maximizing the metric attribute. (default: %(default)s)')
    args = vars(parser.parse_args())

    config = {
        "learning_rate": tune.choice([0.001, 0.003, 0.0001, 0.0003]),
        "dropout": tune.choice([0.2, 0.4, 0.6, 0.8]),
        "num_filter_maps": tune.choice([(50 + 100*i) for i in range(6)]),
        "filter_size": tune.choice([2, 4, 6, 8, 10]),
        **args
    }
    run_random_sampling(config)


# Calculate wall time.
wall_time_start = time.time()
main()
print(f"\nWall time: {time.time()-wall_time_start}")
