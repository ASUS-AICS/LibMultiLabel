import logging
from math import ceil

import numpy as np
from tqdm import tqdm

import libmultilabel.linear as linear
from libmultilabel.common_utils import dump_log
from libmultilabel.linear.utils import LINEAR_TECHNIQUES


def linear_test(config, model, datasets, label_mapping, task):
    metrics = linear.get_metrics(
        config.monitor_metrics, datasets["test"]["y"].shape[1], multiclass=task != "multilabel"
    )
    num_instance = datasets["test"]["x"].shape[0]
    k = config.save_k_predictions
    if k > 0:
        labels = np.zeros((num_instance, k), dtype=object)
        scores = np.zeros((num_instance, k), dtype="d")
    else:
        labels = []
        scores = []
    for i in tqdm(range(ceil(num_instance / config.eval_batch_size))):
        slice = np.s_[i * config.eval_batch_size : (i + 1) * config.eval_batch_size]
        preds = linear.predict_values(model, datasets["test"]["x"][slice])
        target = datasets["test"]["y"][slice].toarray()
        metrics.update(preds, target)
        if k > 0:
            labels[slice], scores[slice] = linear.get_topk_labels(preds, label_mapping, config.save_k_predictions)
        elif config.save_positive_predictions:
            res = linear.get_positive_labels(preds, label_mapping)
            labels.append(res[0])
            scores.append(res[1])
    metric_dict = metrics.compute()
    return metric_dict, labels, scores


def linear_train(datasets, config):
    if config.linear_technique == "tree":
        model = LINEAR_TECHNIQUES[config.linear_technique](
            datasets["train"]["y"],
            datasets["train"]["x"],
            config.liblinear_options,
            config.tree_degree,
            config.tree_max_depth,
        )
    else:
        model = LINEAR_TECHNIQUES[config.linear_technique](
            datasets["train"]["y"],
            datasets["train"]["x"],
            config.liblinear_options,
        )
    return model


def linear_run(config):
    if config.seed is not None:
        np.random.seed(config.seed)

    if config.eval:
        preprocessor, model = linear.load_pipeline(config.checkpoint_path)
        datasets = linear.load_dataset(config.data_format, config.training_file, config.test_file)
        datasets = preprocessor.transform(datasets)
    else:
        preprocessor = linear.Preprocessor(config.include_test_labels, config.remove_no_label_data)
        datasets = linear.load_dataset(
            config.data_format,
            config.training_file,
            config.test_file,
            config.label_file,
        )
        datasets = preprocessor.fit_transform(datasets)

        # detect the task type
        task = "multilabel"
        split = "test" if "test" in datasets else "train"
        if datasets[split]["y"].nnz == datasets[split]["y"].shape[0]:
            if datasets[split]["y"].shape[1] == 2:
                task = "binary"
            else:
                task = "multiclass"

        model = linear_train(datasets, config)
        linear.save_pipeline(config.checkpoint_dir, preprocessor, model)

    if config.test_file is not None:
        assert not (
            config.save_positive_predictions and config.save_k_predictions > 0
        ), """
            If save_k_predictions is larger than 0, only top k labels are saved.
            Save all labels with decision value larger than 0 by using save_positive_predictions and save_k_predictions=0."""
        metric_dict, labels, scores = linear_test(config, model, datasets, preprocessor.label_mapping, task)
        dump_log(config=config, metrics=metric_dict, split="test", log_path=config.log_path)
        print(linear.tabulate_metrics(metric_dict, "test"))
        if config.save_k_predictions > 0:
            with open(config.predict_out_path, "w") as fp:
                for label, score in zip(labels, scores):
                    out_str = " ".join([f"{i}:{s:.4}" for i, s in zip(label, score)])
                    fp.write(out_str + "\n")
            logging.info(f"Saved predictions to: {config.predict_out_path}")
        elif config.save_positive_predictions:
            with open(config.predict_out_path, "w") as fp:
                for batch_labels, batch_scores in zip(labels, scores):
                    for label, score in zip(batch_labels, batch_scores):
                        out_str = " ".join([f"{i}:{s:.4}" for i, s in zip(label, score)])
                        fp.write(out_str + "\n")
            logging.info(f"Saved predictions to: {config.predict_out_path}")
