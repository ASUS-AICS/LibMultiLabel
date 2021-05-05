import os
import yaml
from datetime import datetime
from pathlib import Path


class Config:
    def __init__(self, config_path):
        self.set_to_default_values()
        self.config = config_path
        if os.path.exists(config_path):
            self.load(config_path)

    def load(self, config_path):
        """Read a configuration from a yaml file."""
        with open(config_path) as fp:
            config = yaml.load(fp, Loader=yaml.SafeLoader)
        self.__dict__.update(**config)

    def set_to_default_values(self):
        """Set parameters to the default values"""
        # env
        self.cpu = False
        self.data_workers = 4
        self.device = None
        self.seed = 1337

        # path / dirs
        self.train_path = None
        self.test_path = None
        self.val_path = None
        self.embed_file = None
        self.vocab_file = None
        self.label_file = None
        self.predict_out_path = None

        # dataset
        self.data_dir = './data/rcv1'
        self.result_dir = './runs'
        self.data_name = 'rcv1'
        self.val_size = 0.2
        self.min_vocab_freq = 1
        self.max_seq_length = 500
        self.fixed_length = False
        self.batch_size = 16
        self.eval_batch_size = 256

        # train
        self.eval = False
        self.epochs = 10000
        self.optimizer = 'adam'
        self.learning_rate = 0.0001
        self.weight_decay = 0
        self.momentum = 0.9
        self.patience = 5

        # model
        self.model_name = 'KimCNN'
        self.init_weight = 'kaiming_uniform'
        self.activation = 'relu'
        self.num_filter_per_size = 128
        self.filter_sizes = [4]
        self.dropout = 0.2
        self.dropout2 = 0.2
        self.pool_size = 2

        # eval
        self.metrics_thresholds = 0.5 # not used
        self.monitor_metrics = ['P@1', 'P@3', 'P@5']
        self.val_metric = 'P@1'

        # log
        self.save_k_predictions = 100

    def set_run_name(self):
        self.run_name = '{}_{}_{}'.format(
            self.data_name,
            Path(self.config).stem if self.config else self.model_name,
            datetime.now().strftime('%Y%m%d%H%M%S'),
        )
