import shutil

from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class EarlyStoppingWithCheckpoint(EarlyStopping):
    def __init__(self, best_checkpoint_path, last_checkpoint_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_checkpoint_path = best_checkpoint_path
        self.last_checkpoint_path = last_checkpoint_path
        assert best_checkpoint_path and last_checkpoint_path, '`best_checkpoint_path` and `last_checkpoint_path` must be specified.'

    def on_validation_end(self, trainer, pl_module):
        ret = super().on_validation_end(trainer, pl_module)
        if trainer.is_global_zero:  # only save for main process
            trainer.save_checkpoint(self.last_checkpoint_path)
            if self.wait_count == 0:  # best metric
                shutil.copyfile(self.last_checkpoint_path,
                                self.best_checkpoint_path)
                if self.verbose:
                    print(f'Saved best model to `{self.best_checkpoint_path}`')
        return ret
