import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import ray.train as ray_train
import yaml
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from libmultilabel.common_utils import AttributeDict, timer
from libmultilabel.nn import data_utils
from libmultilabel.nn.nn_utils import set_seed
from torch_trainer import TorchTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")


def train_libmultilabel_tune(config, datasets, classes, word_dict):
    """The training function for ray tune.

    Args:
        config (dict): Config of the experiment.
        datasets (dict): A dictionary of datasets.
        classes(list): List of class names.
        word_dict(torchtext.vocab.Vocab): A vocab object which maps tokens to indices.
    """

    # ray convert AttributeDict to dict
    config = AttributeDict(config)
    set_seed(seed=config.seed)
    config.run_name = ray_train.get_context().get_trial_dir()
    logging.info(f"Run name: {config.run_name}")
    config.checkpoint_dir = os.path.join(config.result_dir, config.run_name)
    config.log_path = os.path.join(config.checkpoint_dir, "logs.json")

    trainer = TorchTrainer(
        config=config,
        datasets=datasets,
        classes=classes,
        word_dict=word_dict,
        save_checkpoints=False,
        is_tune_mode=True,
    )
    val_score = trainer.train()
    return {f"val_{config.val_metric}": val_score}


def load_config_from_file(config_path):
    """Initialize the model config.

    Args:
        config_path (str): Path to the config file.

    Returns:
        dict: Config of the experiment.
    """
    with open(config_path) as fp:
        config = yaml.safe_load(fp)

    # create directories that hold the shared data
    os.makedirs(config["result_dir"], exist_ok=True)
    if config["embed_cache_dir"]:
        os.makedirs(config["embed_cache_dir"], exist_ok=True)

    # set relative path to absolute path (_path, _file, _dir)
    for k, v in config.items():
        if isinstance(v, str) and os.path.exists(v):
            config[k] = os.path.abspath(v)

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
    search_spaces = [
        "choice",
        "grid_search",
        "uniform",
        "quniform",
        "loguniform",
        "qloguniform",
        "randn",
        "qrandn",
        "randint",
        "qrandint",
    ]
    for key, value in config.items():
        if isinstance(value, list) and len(value) >= 2 and value[0] in search_spaces:
            search_space, search_args = value[0], value[1:]
            if (
                isinstance(search_args[0], list)
                and any(isinstance(x, list) for x in search_args[0])
                and search_space != "grid_search"
            ):
                raise ValueError(
                    """If the search values are lists, the search space must be `grid_search`.
                    Take `filter_sizes: ['grid_search', [[2,4,8], [4,6]]]` for example, the program will grid search over
                    [2,4,8] and [4,6]. This is the same as assigning `filter_sizes` to either [2,4,8] or [4,6] in two runs.
                    """
                )
            else:
                config[key] = getattr(tune, search_space)(*search_args)
                parameter_columns[prefix + key] = key
        elif isinstance(value, dict):
            config[key] = init_search_params_spaces(value, parameter_columns, f"{prefix}{key}/")

    return config


def init_search_algorithm(search_alg, metric=None, mode=None):
    """Specify a search algorithm and you must pip install it first.
    If no search algorithm is specified, the default search algorithm is BasicVariantGenerator.
    See more details here: https://docs.ray.io/en/latest/tune/api/suggestion.html

    Args:
        search_alg (str): One of 'basic_variant', 'bayesopt', or 'optuna'.
        metric (str): The metric to monitor for early stopping.
        mode (str): One of 'min' or 'max' to determine whether to minimize or maximize the metric.
    """
    if search_alg == "optuna":
        assert metric and mode, "Metric and mode cannot be None for optuna."
        from ray.tune.search.optuna import OptunaSearch

        return OptunaSearch(metric=metric, mode=mode)
    elif search_alg == "bayesopt":
        assert metric and mode, "Metric and mode cannot be None for bayesian optimization."
        from ray.tune.search.bayesopt import BayesOptSearch

        return BayesOptSearch(metric=metric, mode=mode)
    logging.info(f"{search_alg} search is found, run BasicVariantGenerator().")


def prepare_retrain_config(best_config, best_log_dir, retrain):
    """Prepare the configuration for re-training.

    Args:
        best_config (AttributeDict): The best hyper-parameter configuration.
        best_log_dir (str): The directory of the best trial of the experiment.
        retrain (bool): Whether to retrain the model with merged training and validation data.
    """
    if retrain:
        best_config.merge_train_val = True

        log_path = os.path.join(best_log_dir, "logs.json")
        if os.path.isfile(log_path):
            with open(log_path) as fp:
                log = json.load(fp)
        else:
            raise FileNotFoundError("The log directory does not contain a log.")

        # For re-training with validation data,
        # we use the number of epochs at the point of optimal validation performance.
        log_metric = np.array([l[best_config.val_metric] for l in log["val"]])
        optimal_idx = log_metric.argmax() if best_config.mode == "max" else log_metric.argmin()
        best_config.epochs = optimal_idx.item() + 1  # plus 1 for epochs
    else:
        best_config.merge_train_val = False


def load_static_data(config, merge_train_val=False):
    """Preload static data once for multiple trials.

    Args:
        config (AttributeDict): Config of the experiment.
        merge_train_val (bool, optional): Whether to merge the training and validation data.
            Defaults to False.

    Returns:
        dict: A dict of static data containing datasets, classes, and word_dict.
    """
    datasets = data_utils.load_datasets(
        training_data=config.training_file,
        test_data=config.test_file,
        val_data=config.val_file,
        val_size=config.val_size,
        merge_train_val=merge_train_val,
        tokenize_text="lm_weight" not in config.network_config,
        remove_no_label_data=config.remove_no_label_data,
    )
    return {
        "datasets": datasets,
        "word_dict": None
        if config.embed_file is None
        else data_utils.load_or_build_text_dict(
            dataset=datasets["train"],
            vocab_file=config.vocab_file,
            min_vocab_freq=config.min_vocab_freq,
            embed_file=config.embed_file,
            embed_cache_dir=config.embed_cache_dir,
            silent=config.silent,
            normalize_embed=config.normalize_embed,
        ),
        "classes": data_utils.load_or_build_label(datasets, config.label_file, config.include_test_labels),
    }


def retrain_best_model(exp_name, best_config, best_log_dir, retrain):
    """Re-train the model with the best hyper-parameters.
    A new model is trained on the combined training and validation data if `retrain` is True.
    If a test set is provided, it will be evaluated by the obtained model.

    Args:
        exp_name (str): The directory to save trials generated by ray tune.
        best_config (dict): The best hyper-parameter configuration.
        best_log_dir (str): The directory of the best trial of the experiment.
        retrain (bool): Whether to retrain the model with merged training and validation data.
    """
    best_config = AttributeDict(best_config)
    best_config.silent = False
    checkpoint_dir = os.path.join(best_config.result_dir, exp_name, "trial_best_params")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_config.run_name = "_".join(exp_name.split("_")[:-1]) + "_best"
    best_config.checkpoint_dir = checkpoint_dir
    best_config.log_path = os.path.join(best_config.checkpoint_dir, "logs.json")
    prepare_retrain_config(best_config, best_log_dir, retrain)
    set_seed(seed=best_config.seed)
    with open(os.path.join(checkpoint_dir, "params.yml"), "w") as fp:
        yaml.dump(dict(best_config), fp)

    data = load_static_data(best_config, merge_train_val=best_config.merge_train_val)

    if retrain:
        logging.info(f"Re-training with best config: \n{best_config}")
        trainer = TorchTrainer(config=best_config, **data)
        trainer.train()
        best_model_path = trainer.checkpoint_callback.last_model_path
    else:
        # If not merging training and validation data, load the best result from tune experiments.
        logging.info(f"Loading best model with best config: \n{best_config}")
        best_checkpoint = os.path.join(best_log_dir, "best_model.ckpt")
        last_checkpoint = os.path.join(best_log_dir, "last.ckpt")
        best_model_path = os.path.join(checkpoint_dir, "best_model.ckpt")
        best_config.checkpoint_path = best_checkpoint
        trainer = TorchTrainer(config=best_config, **data)
        os.popen(f"cp {best_checkpoint} {best_model_path}")
        os.popen(f"cp {last_checkpoint} {os.path.join(checkpoint_dir, 'last.ckpt')}")

    if "test" in data["datasets"]:
        test_results = trainer.test()
        if retrain:
            logging.info(f"Test results after re-training: {test_results}")
        else:
            logging.info(f"Test results of best config: {test_results}")
    logging.info(f"Best model saved to {best_model_path}.")


@timer
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Path to configuration file (default: %(default)s). Please specify a config with all arguments in LibMultiLabel/main.py::get_config.",
    )
    parser.add_argument(
        "--cpu_count",
        type=int,
        default=4,
        help="Number of CPU per trial (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu_count",
        type=int,
        default=1,
        help="Number of GPU per trial (default: %(default)s)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of running trials. If the search space is `grid_search`, the same grid will be repeated `num_samples` times. (default: %(default)s)",
    )
    parser.add_argument(
        "--search_alg",
        default=None,
        choices=["basic_variant", "bayesopt", "optuna"],
        help="Search algorithms (default: %(default)s)",
    )
    parser.add_argument(
        "--no_retrain",
        action="store_true",
        help="Do not retrain the model with validation set after hyperparameter search.",
    )
    args, _ = parser.parse_known_args()

    # Load config from the config file and overwrite values specified in CLI.
    parameter_columns = dict()  # parameters to include in progress table of CLIReporter
    config = load_config_from_file(args.config)
    config = init_search_params_spaces(config, parameter_columns, prefix="")
    parser.set_defaults(**config)
    config = AttributeDict(vars(parser.parse_args()))
    # no need to include validation during parameter search
    config.merge_train_val = False
    config.mode = "min" if config.val_metric == "Loss" else "max"

    # Check if the validation set is provided.
    if "val_file" not in config:
        config.val_file = None
    assert (
        config.val_size > 0 or config.val_file is not None
    ), "You should specify either a positive `val_size` or a `val_file` for parameter search."

    """Run tune analysis.
    - If no search algorithm is specified, the default search algorighm is BasicVariantGenerator.
      https://docs.ray.io/en/master/tune/api_docs/suggestion.html#tune-basicvariant
    - Arguments without search spaces will be ignored by `tune.run`
      (https://github.com/ray-project/ray/blob/34d3d9294c50aea4005b7367404f6a5d9e0c2698/python/ray/tune/suggest/variant_generator.py#L333),
      so we parse the whole config to `tune.run` here for simplicity.
    """
    data = load_static_data(config)
    reporter = tune.CLIReporter(
        metric_columns=[f"val_{metric}" for metric in config.monitor_metrics],
        parameter_columns=parameter_columns,
        metric=f"val_{config.val_metric}",
        mode=config.mode,
        sort_by_metric=True,
    )
    if config.scheduler is not None:
        scheduler = ASHAScheduler(metric=f"val_{config.val_metric}", mode=config.mode, **config.scheduler)
    else:
        scheduler = None

    # Fix issue: https://github.com/ray-project/ray/issues/28197
    import ray

    ray.init(runtime_env={"env_vars": {"PL_DISABLE_FORK": "1"}})

    exp_name = "{}_{}_{}".format(
        config.data_name,
        Path(config.config).stem if config.config else config.model_name,
        datetime.now().strftime("%Y%m%d%H%M%S"),
    )
    analysis = tune.run(
        tune.with_parameters(train_libmultilabel_tune, **data),
        search_alg=init_search_algorithm(config.search_alg, metric=f"val_{config.val_metric}", mode=config.mode),
        scheduler=scheduler,
        local_dir=config.result_dir,
        num_samples=config.num_samples,
        resources_per_trial={"cpu": config.cpu_count, "gpu": config.gpu_count},
        progress_reporter=reporter,
        config=config,
        name=exp_name,
    )
    # Save best model after parameter search.
    best_trial = analysis.get_best_trial(metric=f"val_{config.val_metric}", mode=config.mode, scope="all")
    retrain_best_model(exp_name, best_trial.config, best_trial.local_path, retrain=not config.no_retrain)


if __name__ == "__main__":
    main()
