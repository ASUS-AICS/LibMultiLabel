import logging
import os
import pickle

from main import get_config, check_config
from libmultilabel.common_utils import Timer, AttributeDict
from libmultilabel.logging import add_stream_handler
from tests.nn.utils import get_names, get_components_from_trainer


def store_components_from_trainer(trainer):
    """Store the components in trainer to conduct API test.

    Args:
        trainer (TorchTrainer): A wrapper for training neural network models with pytorch lightning trainer.
    """

    names = get_names()
    components = get_components_from_trainer(trainer)

    # Store the components by pickle
    for name, component in zip(names, components):
        with open(os.path.join(trainer.checkpoint_dir, f"{name}.p"), "wb") as f:
            pickle.dump(component, f)

    logging.info(f"Components for testing saved to {trainer.checkpoint_dir}.")


def main():
    # Get config
    config = get_config()
    check_config(config)

    # Set up logger
    log_level = logging.WARNING if config.silent else logging.INFO
    add_stream_handler(log_level)

    logging.info(f"Run name: {config.run_name}")

    if config.linear:
        from linear_trainer import linear_run

        linear_run(config)
    else:
        from torch_trainer import TorchTrainer

        trainer = TorchTrainer(config)  # initialize trainer
        store_components_from_trainer(trainer)
        # train
        if not config.eval:
            trainer.train()
        # test
        if "test" in trainer.datasets:
            trainer.test()


if __name__ == "__main__":
    wall_time = Timer()
    main()
    print(f"Wall time: {wall_time.time():.2f} (s)")
