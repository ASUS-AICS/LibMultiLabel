import warnings

import torch
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint, warning_cache
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        monitor: str,
        dirpath=None,
        filename: str = None,
        save_weights_only: bool = False,
        verbose: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
    ):
        """Cache the best model during validation and save it at the end of training"""
        if monitor is None:
            raise ValueError("Monitor has to be set")

        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_top_k=1,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
        )
        # As we only want the top-1 model, these values are equal.
        self.best_model_score = self.kth_value
        self.best_state_dict = {}
        # For compatibility reasons, we use 'saved' instead of 'cached'.
        self._last_epoch_saved = 0
        self.best_monitor_candidates = None

        # variables of our interest initialized in the parent class
        # self.monitor = monitor
        # self.verbose = verbose
        # self.save_weights_only = save_weights_only
        # self.current_score = None # tensor
        # self.best_model_path = ""
        # self._last_global_step_saved = 0

    @property
    def state_key(self) -> str:
        return self._generate_state_key(monitor=self.monitor, mode=self.mode)

    def on_train_start(self, trainer, pl_module):
        return

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        return

    def on_train_end(self, trainer, pl_module):
        """Save the checkpoint with the best validation result at the end of training."""
        self.best_model_path = self._get_metric_interpolated_filepath_name(self.best_monitor_candidates, trainer)
        if self.save_weights_only:
            checkpoint = self.best_state_dict
        else:
            checkpoint = trainer._checkpoint_connector.dump_checkpoint(weights_only=False)
            checkpoint["state_dict"] = self.best_state_dict
            checkpoint["epoch"] = self._last_epoch_saved
            checkpoint["global_step"] = self._last_global_step_saved
        trainer.strategy.save_checkpoint(checkpoint, self.best_model_path)
        trainer.strategy.barrier("Trainer.save_checkpoint")

    def on_validation_end(self, trainer, pl_module):
        """Cache the checkpoint with the best validation result at the end of validation."""
        if not self._should_skip_saving_checkpoint(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self.monitor not in monitor_candidates:
                m = (
                    f"`ModelCheckpoint(monitor={self.monitor!r})` could not find the monitored key in the returned"
                    f" metrics: {list(monitor_candidates)}."
                    f" HINT: Did you call `log({self.monitor!r}, value)` in the `LightningModule`?"
                )
                if trainer.fit_loop.epoch_loop.val_loop._has_run:
                    raise MisconfigurationException(m)
                warning_cache.warn(m)
            self.current_score = monitor_candidates.get(self.monitor)
            assert self.current_score is not None

            monitor_op = {"min": torch.lt, "max": torch.gt}[self.mode]
            should_update_best_and_save = monitor_op(self.current_score, self.best_model_score)

            # If using multiple devices, make sure all processes are unanimous on the decision.
            should_update_best_and_save = trainer.strategy.reduce_boolean_decision(bool(should_update_best_and_save))

            if should_update_best_and_save:
                # do not save nan, replace with +/- inf
                if isinstance(self.current_score, torch.Tensor) and torch.isnan(self.current_score):
                    self.current_score = torch.tensor(
                        float("inf" if self.mode == "min" else "-inf"), device=self.current_score.device
                    )

                self.best_model_score = self.current_score

                if self.verbose:
                    rank_zero_info(
                        f"Epoch {self._last_epoch_saved:d}, global step {self._last_global_step_saved:d}: "
                        f"{repr(self.monitor)} reached {self.current_score:.5f}"
                    )

                # cache checkpoint
                state_dict = trainer._checkpoint_connector.dump_checkpoint(weights_only=True)["state_dict"]
                for k in state_dict:
                    self.best_state_dict[k] = state_dict[k].detach().cpu()

                self._last_epoch_saved = monitor_candidates["epoch"]
                self._last_global_step_saved = monitor_candidates["step"]
                self.best_monitor_candidates = monitor_candidates

                # skip notifying logger because their behaviors are not clear
            elif self.verbose:
                epoch = monitor_candidates["epoch"]
                step = monitor_candidates["step"]
                rank_zero_info(f"Epoch {epoch:d}, global step {step:d}: {repr(self.monitor)} was not the best")

    def state_dict(self):
        return {
            "monitor": self.monitor,
            "best_model_score": self.best_model_score,
            "best_model_path": self.best_model_path,
            "current_score": self.current_score,
            "dirpath": self.dirpath,
        }

    def load_state_dict(self, state_dict):
        dirpath_from_ckpt = state_dict.get("dirpath", self.dirpath)

        if self.dirpath == dirpath_from_ckpt:
            self.best_model_score = state_dict["best_model_score"]
        else:
            warnings.warn(
                f"The dirpath has changed from {repr(dirpath_from_ckpt)} to {repr(self.dirpath)},"
                " therefore `best_model_score`, `kth_best_model_path`, `kth_value`, `last_model_path` and"
                " `best_k_models` won't be reloaded. Only `best_model_path` will be reloaded."
            )

        self.best_model_path = state_dict["best_model_path"]
