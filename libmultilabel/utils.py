import copy
import json
import logging
import os
import time

import numpy as np


class ArgDict(dict):
    def __init__(self, *args, **kwargs):
        super(ArgDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elasped time."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total


def dump_log(config, metrics, split):
    log_path = os.path.join(config.result_dir, config.run_name, 'logs.json')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.isfile(log_path):
        with open(log_path) as fp:
            result = json.load(fp)
    else:
        config_to_save = copy.deepcopy(config)
        del config_to_save['device']
        result = {'config': config_to_save}

    if split in result:
        result[split].append(metrics)
    else:
        result[split] = [metrics]
    with open(log_path, 'w') as fp:
        json.dump(result, fp)


def dump_top_k_prediction(config, classes, y_pred, k=100):
    """Dump top k predictions to the predict_out_path. The format of this file is:
    <label1>:<value1> <label2>:<value2> ...

    Parameters:
    classes (list): list of class names
    y_pred (ndarray): predictions (shape: number of samples * number of classes)
    k (int): number of classes considered as the correct labels
    """

    if config.predict_out_path:
        predict_out_path = config.predict_out_path
    else:
        predict_out_path = os.path.join(config.result_dir, config.run_name, 'predictions.txt')

    logging.info(f'Dump top {k} predictions to {predict_out_path}.')
    with open(predict_out_path, 'w') as fp:
        for pred in y_pred:
            label_ids = np.argsort(-pred).tolist()[:k]
            out_str = ' '.join([f'{classes[i]}:{pred[i]:.4}' for i in label_ids])
            fp.write(out_str+'\n')
