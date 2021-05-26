import argparse
import logging
import os
import time
import yaml
from datetime import datetime
from pathlib import Path

from ray import tune

from libmultilabel import data_utils
from libmultilabel.evaluate import evaluate
from libmultilabel.model import Model
from libmultilabel.utils import ArgDict, set_seed, init_device


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


def training_function(config):
    model_config = ArgDict(config)
    datasets = data_utils.load_datasets(model_config)
    word_dict = data_utils.load_or_build_text_dict(model_config, datasets['train'])
    classes = data_utils.load_or_build_label(model_config, datasets)

    model = Model(model_config, word_dict, classes)
    model.train(datasets['train'], datasets['val'])
    model.load_best()

    # return best eval metric
    val_loader = data_utils.get_dataset_loader(model_config, datasets['val'], model.word_dict, model.classes, train=False)
    results = evaluate(model, val_loader, model_config.monitor_metrics)
    yield results.get_metric_dict(use_cache=False)


def init_model_config(config_path):
    with open(config_path) as fp:
        args = yaml.load(fp, Loader=yaml.SafeLoader)

    # set relative path to absolute path (_path, _file, _dir)
    for k, v in args.items():
        if isinstance(v, str) and (os.path.isfile(v) or os.path.isdir(v)):
            args[k] = os.path.abspath(v)

    model_config = ArgDict(args)
    set_seed(seed=model_config.seed)
    model_config.device = init_device(model_config.cpu)
    model_config.run_name = '{}_{}_{}'.format(
        model_config.data_name,
        Path(model_config.config).stem if model_config.config else model_config.model_name,
        datetime.now().strftime('%Y%m%d%H%M%S'),
    )
    logging.info(f'Run name: {model_config.run_name}')
    return model_config


def init_search_space(search_alg, search_func, values):
    """Initialize the search space by the given search algorithm.
    Currently, the search algorithm decides what search function is used for all parameters.
    """
    if search_func == 'grid':
        if search_alg != 'grid':
            raise ValueError(f'{search_alg} does not support grid search.')
        return tune.grid_search(values)
    elif search_func == 'uniform':
        assert len(values) == 2 and values[0] < values[1]
        # sample an option uniformly between values[0] and values[1]
        return tune.uniform(values[0], values[1])
    else:
        # sample an option uniformly from list of values
        if search_alg == 'bayesopt':
            raise ValueError(f'{search_alg} does not support discrete search spaces.')
        return tune.choice(values)


def get_search_algorithm(search_alg, metric=None, mode=None):
    """Specify a search algorithm and you must pip install it first.
    See more details here: https://docs.ray.io/en/master/tune/api_docs/suggestion.html
    """
    if search_alg == 'optuna':
        assert metric and mode, "Metric and mode cannot be None for optuna."
        from ray.tune.suggest.optuna import OptunaSearch
        return OptunaSearch(metric=metric, mode=mode)
    elif search_alg == 'bayesopt':
        assert metric and mode, "Metric and mode cannot be None for bayesian optimization."
        from ray.tune.suggest.bayesopt import BayesOptSearch
        return BayesOptSearch(metric=metric, mode=mode)
    logging.info(f'{search_alg} search is found, run BasicVariantGenerator().')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', help='Path to configuration file (default: %(default)s). Please specify a config with all arguments in LibMultiLabel/main.py::get_config.')
    parser.add_argument('--cpu_count', type=int, default=4, help='Number of CPU (default: %(default)s)')
    parser.add_argument('--gpu_count', type=int, default=1, help='Number of GPU (default: %(default)s)')
    parser.add_argument('--local_dir', default=os.getcwd(), help='Directory to save training results (default: %(default)s)')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of running samples (default: %(default)s)')
    parser.add_argument('--mode', default='max', choices=['min', 'max'], help='Determines whether objective is minimizing or maximizing the metric attribute. (default: %(default)s)')
    parser.add_argument('--search_alg', default=None, choices=['random', 'grid', 'bayesopt', 'optuna'], help='Search algorithms (default: %(default)s)')
    parser.add_argument('--search_params', default=None, nargs='+',
                        help='List of search parameters.(default: %(default)s)')
    args = parser.parse_args()

    """Other args in the model config are viewed as resolved values that are ignored from tune.
    https://github.com/ray-project/ray/blob/34d3d9294c50aea4005b7367404f6a5d9e0c2698/python/ray/tune/suggest/variant_generator.py#L333
    """
    model_config = init_model_config(args.config)
    for param in args.search_params:
        assert param in model_config, f"Please specify {param} in the config. (Ex. dropout: [[0.2, 0.4, 0.6, 0.8], 'choice')"
        if isinstance(model_config[param][0], list): # filter_sizes
            if args.search_alg == 'bayesopt' or args.search_algo == 'optuna':
                raise TypeError(f'{args.search_alg} does not support list of search spaces.')
            model_config[param] = [init_search_space(args.search_alg, search_func, values) for search_func, values in model_config[param]]
        else:
            model_config[param] = init_search_space(args.search_alg, *model_config[param])

    """Run tune analysis.
    If no search algorithm is specified, the default search algorighm is BasicVariantGenerator.
    https://docs.ray.io/en/master/tune/api_docs/suggestion.html#tune-basicvariant
    """
    search_alg = get_search_algorithm(args.search_alg, metric=model_config.val_metric, mode=args.mode)
    tune.run(
        training_function,
        search_alg=search_alg,
        local_dir=args.local_dir,
        metric=model_config.val_metric,
        mode=args.mode,
        num_samples=args.num_samples,
        resources_per_trial={
            'cpu': args.cpu_count, 'gpu': args.gpu_count},
        config=model_config)


# calculate wall time.
wall_time_start = time.time()
main()
logging.info(f"\nWall time: {time.time()-wall_time_start}")
