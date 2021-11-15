import argparse
import glob
import itertools
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import yaml
from pytorch_lightning.utilities.parsing import AttributeDict
from ray import tune

from libmultilabel.nn import data_utils
from libmultilabel.nn.nn_utils import init_device, set_seed
from torch_trainer import TorchTrainer


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')


class Trainable(tune.Trainable):
    def setup(self, config, data):
        self.config = AttributeDict(config)
        self.datasets = data['datasets']
        self.word_dict = data['word_dict']
        self.classes = data['classes']
        self.device = init_device(config.cpu)
        set_seed(seed=self.config.seed)

    def step(self):
        self.config.run_name = '{}_{}_{}_{}'.format(
            self.config.data_name,
            Path(
                self.config.config).stem if self.config.config else self.config.model_name,
            datetime.now().strftime('%Y%m%d%H%M%S'),
            self.trial_id
        )
        logging.info(f'Run name: {self.config.run_name}')

        self.config.checkpoint_dir = os.path.join(self.config.result_dir, self.config.run_name)
        self.config.log_path = os.path.join(self.config.checkpoint_dir, 'logs.json')

        trainer = TorchTrainer(config=self.config,
                               datasets=self.datasets,
                               classes=self.classes,
                               word_dict=self.word_dict)
        trainer.train()

        # Test the model on the validation set.
        metric_dict = trainer.test(split='val')
        val_results = {f'val_{k}': v*100 for k, v in metric_dict.items()}

        # Remove *.ckpt.
        for model_path in glob.glob(os.path.join(self.config.result_dir, self.config.run_name, '*.ckpt')):
            logging.info(f'Removing {model_path} ...')
            os.remove(model_path)

        return val_results


def load_config_from_file(config_path):
    """Initialize the model config.

    Args:
        config_path (str): Path to the config file.

    Returns:
        AttributeDict: Config of the experiment.
    """
    with open(config_path) as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    # create directories that hold the shared data
    os.makedirs(config['result_dir'], exist_ok=True)
    if config['embed_cache_dir']:
        os.makedirs(config['embed_cache_dir'], exist_ok=True)

    # set relative path to absolute path (_path, _file, _dir)
    for k, v in config.items():
        if isinstance(v, str) and os.path.exists(v):
            config[k] = os.path.abspath(v)

    set_seed(seed=config['seed'])
    return config


def init_search_params_spaces(config, parameter_columns, prefix):
    """Initialize the sample space defined in ray tune.
    See the random distributions API listed here: https://docs.ray.io/en/master/tune/api_docs/search_space.html#random-distributions-api

    Args:
        config (dict): Config of the experiment.
        parameter_columns (dict): Names of parameters to include in the CLIReporter.
                                  The keys are parameter names and the values are displayed names.
        prefix(str): The prefix of a nested parameter such as network_config/dropout.

    Returns:
        dict: Config with parsed sample spaces.
    """
    search_spaces = ['choice', 'grid_search', 'uniform', 'quniform', 'loguniform',
                     'qloguniform', 'randn', 'qrandn', 'randint', 'qrandint']
    for key, value in config.items():
        if isinstance(value, list) and len(value) >= 2 and value[0] in search_spaces:
            search_space, search_args = value[0], value[1:]
            if isinstance(search_args[0], list) and any(isinstance(x, list) for x in search_args[0]) and search_space != 'grid_search':
                raise ValueError(
                    """If the search values are lists, the search space must be `grid_search`.
                    Take `filter_sizes: ['grid_search', [[2,4,8], [4,6]]]` for example, the program will grid search over
                    [2,4,8] and [4,6]. This is the same as assigning `filter_sizes` to either [2,4,8] or [4,6] in two runs.
                    """)
            else:
                config[key] = getattr(tune, search_space)(*search_args)
                parameter_columns[prefix+key] = key
        elif isinstance(value, dict):
            config[key] = init_search_params_spaces(value, parameter_columns, f'{prefix}{key}/')

    return config


def init_search_algorithm(search_alg, metric=None, mode=None):
    """Specify a search algorithm and you must pip install it first.
    If no search algorithm is specified, the default search algorithm is BasicVariantGenerator.
    See more details here: https://docs.ray.io/en/master/tune/api_docs/suggestion.html

    Args:
        search_alg (str): One of 'basic_variant', 'bayesopt', or 'optuna'.
        metric (str): The metric to monitor for early stopping.
        mode (str): One of 'min' or 'max' to determine whether to minimize or maximize the metric.
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


def load_static_data(config, merge_train_val=False):
    """Preload static data once for multiple trials.

    Args:
        config (AttributeDict): Config of the experiment.
        merge_train_val (bool, optional): Whether to merge the training and validation data.
            Defaults to False.

    Returns:
        dict: A dict of static data containing datasets, classes, and word_dict.
    """
    datasets = data_utils.load_datasets(data_dir=config.data_dir,
                                        train_path=config.train_path,
                                        test_path=config.test_path,
                                        val_path=config.val_path,
                                        val_size=config.val_size,
                                        is_eval=config.eval,
                                        merge_train_val=merge_train_val)
    return {
        "datasets": datasets,
        "word_dict": data_utils.load_or_build_text_dict(
            dataset=datasets['train'],
            vocab_file=config.vocab_file,
            min_vocab_freq=config.min_vocab_freq,
            embed_file=config.embed_file,
            embed_cache_dir=config.embed_cache_dir,
            silent=config.silent,
            normalize_embed=config.normalize_embed
        ),
        "classes": data_utils.load_or_build_label(datasets, config.label_file, config.silent)
    }


def retrain_best_model(log_path, merge_train_val=False):
    """Retrain the best model with the best hyperparameters.
    The model will train on the whole training data if `merge_train_val` is True.

    Args:
        log_path (str): The log file of the best trail generated by ray tune.
        merge_train_val (bool, optional): Whether to merge the training and validation data.
            Defaults to False.
    """
    best_config = AttributeDict(json.load(open(log_path, 'r'))['config'])
    best_config.run_name = f'{best_config.run_name}_retrain'
    best_config.checkpoint_dir = os.path.join(best_config.result_dir, best_config.run_name)
    best_config.log_path = os.path.join(best_config.checkpoint_dir, 'logs.json')

    if merge_train_val:
        logging.info('Use the full training data to retrain the best model.')
        data = load_static_data(best_config, merge_train_val=merge_train_val)

    logging.info(f'Retraining with best config: \n{best_config}')
    trainer = TorchTrainer(config=best_config, **data)
    trainer.train()

    if 'test' in data:
        trainer.test()
    logging.info(f'Best model saved to {trainer.checkpoint_callback.best_model_path or trainer.checkpoint_callback.last_model_path}.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', help='Path to configuration file (default: %(default)s). Please specify a config with all arguments in LibMultiLabel/main.py::get_config.')
    parser.add_argument('--cpu_count', type=int, default=4,
                        help='Number of CPU per trial (default: %(default)s)')
    parser.add_argument('--gpu_count', type=int, default=1,
                        help='Number of GPU per trial (default: %(default)s)')
    parser.add_argument('--local_dir', default=os.getcwd(),
                        help='Directory to save training results of tune (default: %(default)s)')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of running trials. If the search space is `grid_search`, the same grid will be repeated `num_samples` times. (default: %(default)s)')
    parser.add_argument('--mode', default='max', choices=['min', 'max'],
                        help='Determines whether objective is minimizing or maximizing the metric attribute. (default: %(default)s)')
    parser.add_argument('--search_alg', default=None, choices=['basic_variant', 'bayesopt', 'optuna'],
                        help='Search algorithms (default: %(default)s)')
    parser.add_argument('--merge_train_val', action='store_true',
                        help='Merge the training and validation data after parameter search.')
    parser.add_argument('--retrain_best', action='store_true', help='Retrain the best model.')
    args, _ = parser.parse_known_args()

    # Load config from the config file and overwrite values specified in CLI.
    parameter_columns = dict()  # parameters to include in progress table of CLIReporter
    config = load_config_from_file(args.config)
    config = init_search_params_spaces(config, parameter_columns, prefix='')
    parser.set_defaults(**config)
    config = AttributeDict(vars(parser.parse_args()))

    # Check if the validation set is provided.
    val_path = config.val_path or os.path.join(config.data_dir, 'valid.txt')
    assert config.val_size > 0 or os.path.exists(val_path), \
        "You should specify either a positive `val_size` or a `val_path` defaults to `data_dir/valid.txt` for parameter search."

    """Run tune analysis.
    - If no search algorithm is specified, the default search algorighm is BasicVariantGenerator.
      https://docs.ray.io/en/master/tune/api_docs/suggestion.html#tune-basicvariant
    - Arguments without search spaces will be ignored by `tune.run`
      (https://github.com/ray-project/ray/blob/34d3d9294c50aea4005b7367404f6a5d9e0c2698/python/ray/tune/suggest/variant_generator.py#L333),
      so we parse the whole config to `tune.run` here for simplicity.
    """
    data = load_static_data(config)
    reporter = tune.CLIReporter(metric_columns=[f'val_{metric}' for metric in config.monitor_metrics],
                                parameter_columns=parameter_columns,
                                metric=f'val_{config.val_metric}',
                                mode=args.mode,
                                sort_by_metric=True)
    analysis = tune.run(
        tune.with_parameters(Trainable, data=data),
        # run one step "libmultilabel.model.train"
        stop={"training_iteration": 1},
        search_alg=init_search_algorithm(
            config.search_alg, metric=config.val_metric, mode=args.mode),
        local_dir=args.local_dir,
        num_samples=config.num_samples,
        resources_per_trial={
            'cpu': args.cpu_count, 'gpu': args.gpu_count},
        progress_reporter=reporter,
        config=config)

    # Save best model after parameter search.
    if args.retrain_best:
        log_path = os.path.join(analysis.get_best_logdir(f'val_{config.val_metric}', args.mode), 'result.json')
        retrain_best_model(log_path, args.merge_train_val)


# calculate wall time.
wall_time_start = time.time()
main()
logging.info(f'\nWall time: {time.time()-wall_time_start}')
