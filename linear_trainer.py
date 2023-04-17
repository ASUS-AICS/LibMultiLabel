import logging
from math import ceil

import numpy as np
from tqdm import tqdm

import libmultilabel.linear as linear
from libmultilabel.common_utils import argsort_top_k, dump_log
from libmultilabel.linear.utils import LINEAR_TECHNIQUES


def linear_test(config, model, datasets):
    metrics = linear.get_metrics(
        config.monitor_metrics,
        datasets['test']['y'].shape[1],
        multiclass=model.name == 'binary_and_multiclass'
    )
    num_instance = datasets['test']['x'].shape[0]
    assert not (config.save_positive_predictions and config.save_k_predictions >
                0), "If save_k_predictions is larger than 0, only top k labels are saved. Save all labels with decision value larger than 0 by using save_all and save_k_predictions=0."
    k = config.save_k_predictions
    if k > 0:
        idx = np.zeros((num_instance, k), dtype='i')
        scores = np.zeros((num_instance, k), dtype='d')
    else:
        idx = []
        scores = []

    for i in tqdm(range(ceil(num_instance / config.eval_batch_size))):
        slice = np.s_[i*config.eval_batch_size:(i+1)*config.eval_batch_size]
        preds = linear.predict_values(model, datasets['test']['x'][slice])
        target = datasets['test']['y'][slice].toarray()
        metrics.update(preds, target)

        if k > 0:
            idx[slice] = argsort_top_k(preds, k, axis=1)
            scores[slice] = np.take_along_axis(preds, idx[slice], axis=1)

        elif config.save_positive_predictions:
            for i_pred in preds:
                pos_idx = i_pred > 0
                idx.append(pos_idx)
                scores.append(i_pred[pos_idx])
    metric_dict = metrics.compute()
    return metric_dict, idx, scores


def linear_train(datasets, config):
    if config.linear_technique == 'tree':
        model = LINEAR_TECHNIQUES[config.linear_technique](
            datasets['train']['y'],
            datasets['train']['x'],
            config.liblinear_options,
            config.tree_degree,
            config.tree_max_depth,
        )
    else:
        model = LINEAR_TECHNIQUES[config.linear_technique](
            datasets['train']['y'],
            datasets['train']['x'],
            config.liblinear_options,
        )
    return model


def linear_run(config):
    if config.seed is not None:
        np.random.seed(config.seed)

    if config.eval:
        preprocessor, model = linear.load_pipeline(config.checkpoint_path)
        datasets = preprocessor.load_data(
            config.training_file, config.test_file, config.eval)
    else:
        preprocessor = linear.Preprocessor(data_format=config.data_format)
        datasets = preprocessor.load_data(
            config.training_file,
            config.test_file,
            config.eval,
            config.label_file,
            config.include_test_labels,
            config.remove_no_label_data)
        model = linear_train(datasets, config)
        linear.save_pipeline(config.checkpoint_dir, preprocessor, model)

    if config.test_file is not None:
        metric_dict, idx, scores = linear_test(
            config, model, datasets)

        dump_log(config=config, metrics=metric_dict,
                 split='test', log_path=config.log_path)
        print(linear.tabulate_metrics(metric_dict, 'test'))
        if config.save_positive_predictions or config.save_k_predictions > 0:
            with open(config.predict_out_path, 'w') as fp:
                for idx, score in zip(idx, scores):
                    out_str = ' '.join([f'{i}:{s:.4}' for i, s in zip(
                        preprocessor.label_mapping[idx], score)])
                    fp.write(out_str+'\n')
            logging.info(f'Saved predictions to: {config.predict_out_path}')
