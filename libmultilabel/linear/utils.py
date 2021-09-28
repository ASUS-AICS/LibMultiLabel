import pickle
import os
from pathlib import Path

from .preprocessor import Preprocessor

__all__ = ['save_pipeline', 'load_pipeline']


def save_pipeline(checkpoint_dir: str, preprocessor: Preprocessor, model):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'linear_pipeline.pickle')

    with open(checkpoint_path, 'wb') as f:
        pickle.dump({
            'preprocessor': preprocessor,
            'model': model,
        }, f)


def load_pipeline(checkpoint_path: str):
    with open(checkpoint_path, 'rb') as f:
        pipeline = pickle.load(f)
    return (pipeline['preprocessor'], pipeline['model'])
