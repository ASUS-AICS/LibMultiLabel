import os
from math import ceil

import numpy as np

import libmultilabel.linear as linear


def linear_test(config, model, datasets):
    metrics = linear.get_metrics(
        config.metric_threshold,
        config.monitor_metrics,
        datasets['test']['y'].shape[1]
    )
    num_instance = datasets['test']['x'].shape[0]
    for i in range(ceil(num_instance / config.eval_batch_size)):
        slice = np.s_[i*config.eval_batch_size:(i+1)*config.eval_batch_size]
        preds = linear.predict_values(model, datasets['test']['x'][slice])
        target = datasets['test']['y'][slice].toarray()
        metrics.update(preds, target)
    print(linear.tabulate_metrics(metrics.compute(), 'test'))


def linear_train(datasets, config):
    techniques = {'1vsrest': linear.train_1vsrest,
                  'thresholding': linear.train_thresholding,
                  'cost_sensitive': linear.train_cost_sensitive}
    model = techniques[config.linear_technique](
        datasets['train']['y'],
        datasets['train']['x'],
        config.liblinear_options,
    )
    return model


def linear_run(config):
    if config.eval:
        preprocessor, model = linear.load_pipeline(config.checkpoint_path)
        datasets = preprocessor.load_data(
            config.train_path, config.test_path, config.eval)
    else:
        preprocessor = linear.Preprocessor(data_format=config.data_format)
        datasets = preprocessor.load_data(
            config.train_path, config.test_path, config.eval)
        model = linear_train(datasets, config)
        linear.save_pipeline(config.checkpoint_dir, preprocessor, model)

    if os.path.exists(config.test_path):
        linear_test(config, model, datasets)
    # TODO: dump logs?
