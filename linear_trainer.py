import os
import logging
from math import ceil

import numpy as np

import libmultilabel.linear as linear
from libmultilabel.common_utils import dump_log, argsort_top_k


def linear_test(config, model, datasets):
    metrics = linear.get_metrics(
        config.metric_threshold,
        config.monitor_metrics,
        datasets['test']['y'].shape[1]
    )
    num_instance = datasets['test']['x'].shape[0]

    k = config.save_k_predictions
    top_k_idx = np.zeros((num_instance, k), dtype='i')
    top_k_scores = np.zeros((num_instance, k), dtype='d')

    for i in range(ceil(num_instance / config.eval_batch_size)):
        slice = np.s_[i*config.eval_batch_size:(i+1)*config.eval_batch_size]
        preds = linear.predict_values(model, datasets['test']['x'][slice])
        target = datasets['test']['y'][slice].toarray()
        metrics.update(preds, target)

        if k > 0:
            top_k_idx[slice] = argsort_top_k(preds, k, axis=1)
            top_k_scores[slice] = np.take_along_axis(
                preds, top_k_idx[slice], axis=1)

    metric_dict = metrics.compute()
    return (metric_dict, top_k_idx, top_k_scores)


def linear_train(datasets, config):
    techniques = {'1vsrest': linear.train_1vsrest,
                  'thresholding': linear.train_thresholding,
                  'cost_sensitive': linear.train_cost_sensitive,
                  'cost_sensitive_micro': linear.train_cost_sensitive_micro}
    model = techniques[config.linear_technique](
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
            config.train_path, config.test_path, config.eval)
    else:
        preprocessor = linear.Preprocessor(data_format=config.data_format)
        datasets = preprocessor.load_data(
            config.train_path,
            config.test_path,
            config.eval,
            config.label_file,
            config.include_test_labels,
            config.keep_no_label_data)
        model = linear_train(datasets, config)
        linear.save_pipeline(config.checkpoint_dir, preprocessor, model)

    if os.path.exists(config.test_path):
        metric_dict, top_k_idx, top_k_scores = linear_test(
            config, model, datasets)

        dump_log(config=config, metrics=metric_dict,
                 split='test', log_path=config.log_path)
        print(linear.tabulate_metrics(metric_dict, 'test'))

        if config.save_k_predictions > 0:
            classes = preprocessor.binarizer.classes_
            with open(config.predict_out_path, 'w') as fp:
                for idx, score in zip(top_k_idx, top_k_scores):
                    out_str = ' '.join([f'{classes[i]}:{s:.4}' for i, s in zip(
                        idx, score)])
                    fp.write(out_str+'\n')
            logging.info(f'Saved predictions to: {config.predict_out_path}')
