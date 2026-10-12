"""Trainer callbacks: hooks into the training loop (checkpointing, early stopping, progress bar or your own).

Every rank runs the callbacks, and the epoch metrics they see are the
all-reduced, dataset-level values, so decisions agree across ranks.
"""
from __future__ import annotations

import math
import os

__all__ = ["Callback", "ModelCheckpoint", "EarlyStopping", "ProgressBar", "monitored_value"]


def monitored_value(metrics: dict, monitor: str) -> float | None:
    """The value of ``monitor`` in ``metrics``, or ``None`` when its split was not evaluated.

    Parameters
    ----------
    metrics : dict of str to float
        Epoch metrics (``train/...``, ``val/...``).
    monitor : str
        The monitored key, e.g. ``"val/forces_mae"``.

    Returns
    -------
    float or None
        ``None`` when no value of the monitor's split is present (a
        validation set skipped this epoch).

    Raises
    ------
    KeyError
        When the split is present but the key is not.
    """
    if monitor in metrics:
        return float(metrics[monitor])
    split = monitor.split("/", 1)[0] + "/"
    present = sorted(k for k in metrics if k.startswith(split))
    if present:
        raise KeyError(f"monitor {monitor!r} is not a logged value; available: {present}")
    return None


def _better(value: float, best: float | None, mode: str, min_delta: float = 0.0) -> bool:
    """Whether ``value`` improves on ``best`` by more than ``min_delta``."""
    if best is None:
        return True
    return value < best - min_delta if mode == "min" else value > best + min_delta


class Callback:
    """Base class of the trainer callbacks; override any of the hooks."""

    def on_fit_start(self, trainer) -> None:
        """Before the first epoch."""

    def on_train_epoch_start(self, trainer, epoch: int) -> None:
        """Before an epoch's training pass."""

    def on_train_batch_end(self, trainer, logs: dict, batch_idx: int) -> None:
        """After each optimizer step, with that batch's loss terms and metrics."""

    def on_epoch_end(self, trainer, epoch: int, metrics: dict) -> None:
        """After an epoch's training, validation and scheduler step.

        ``metrics`` holds ``epoch``, ``lr``, ``train/<key>`` and, on
        validation epochs, ``val/<key>``. Set ``trainer.should_stop`` to end
        the run after this epoch.
        """

    def on_train_end(self, trainer) -> None:
        """After the last epoch, before the test evaluation."""

    def on_fit_end(self, trainer, results: dict) -> None:
        """At the end of :meth:`~xnn.common.train.trainer.Trainer.fit`, with its return value."""


class ModelCheckpoint(Callback):
    """Saves the best model by a monitored value, the last model, and optional periodic snapshots.

    Parameters
    ----------
    monitor : str or None, optional
        The value to monitor; defaults to the trainer's (``trainer.monitor``).
    mode : str or None, optional
        ``"min"`` or ``"max"``; defaults to the trainer's.
    dirpath : str or None, optional
        Directory of the checkpoints; defaults to ``cfg.output_dir``.
    filename : str, optional
        Name of the best checkpoint. Defaults to ``"best.pt"``.
    save_last : bool, optional
        Also write ``last.pt`` after the last epoch. Defaults to ``True``.
    every_n_epochs : int, optional
        When ``> 0``, also write ``epoch_<n>.pt`` every n epochs. Defaults to ``0``.

    Attributes
    ----------
    best_score : float or None
        The best monitored value so far.
    best_epoch : int or None
        The epoch it was reached in.
    """

    def __init__(self, monitor: str | None = None, mode: str | None = None,
                 dirpath: str | None = None, filename: str = "best.pt",
                 save_last: bool = True, every_n_epochs: int = 0):
        self.monitor, self.mode, self.dirpath = monitor, mode, dirpath
        self.filename, self.save_last, self.every_n_epochs = filename, save_last, every_n_epochs
        self.best_score = self.best_epoch = None

    def on_fit_start(self, trainer) -> None:
        """Fill the defaults from the trainer."""
        self.monitor = self.monitor or trainer.monitor
        self.mode = self.mode or trainer.mode
        self.dirpath = self.dirpath or trainer.cfg.output_dir
        os.makedirs(self.dirpath, exist_ok=True)

    def on_epoch_end(self, trainer, epoch: int, metrics: dict) -> None:
        """Save on improvement and on the periodic epochs."""
        value = monitored_value(metrics, self.monitor)
        if value is not None and math.isfinite(value) and _better(value, self.best_score, self.mode):
            self.best_score, self.best_epoch = value, epoch
            trainer.save(os.path.join(self.dirpath, self.filename))
        if self.every_n_epochs > 0 and (epoch + 1) % self.every_n_epochs == 0:
            trainer.save(os.path.join(self.dirpath, f"epoch_{epoch + 1}.pt"))

    def on_train_end(self, trainer) -> None:
        """Write ``last.pt``."""
        if self.save_last:
            trainer.save(os.path.join(self.dirpath, "last.pt"))


class EarlyStopping(Callback):
    """Stops training when a monitored value stops improving, reaches a threshold, or is not finite.

    Parameters
    ----------
    monitor : str or None, optional
        The value to monitor; defaults to the trainer's.
    mode : str or None, optional
        ``"min"`` or ``"max"``; defaults to the trainer's.
    patience : int, optional
        Evaluations without improvement before stopping; ``0`` disables
        this rule. Defaults to ``10``.
    min_delta : float, optional
        The least change that counts as an improvement. Defaults to ``0.0``.
    stopping_threshold : float or None, optional
        Stop as soon as the value is better than this. Defaults to ``None``.
    check_finite : bool, optional
        Stop when the value is NaN or infinite. Defaults to ``True``.

    Attributes
    ----------
    stopped_epoch : int or None
        The epoch training stopped after, ``None`` if it ran to the end.
    reason : str or None
        Why it stopped.
    """

    def __init__(self, monitor: str | None = None, mode: str | None = None, patience: int = 10,
                 min_delta: float = 0.0, stopping_threshold: float | None = None,
                 check_finite: bool = True):
        self.monitor, self.mode, self.patience = monitor, mode, patience
        self.min_delta, self.stopping_threshold = min_delta, stopping_threshold
        self.check_finite = check_finite
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = self.reason = None

    def on_fit_start(self, trainer) -> None:
        """Fill the defaults from the trainer."""
        self.monitor = self.monitor or trainer.monitor
        self.mode = self.mode or trainer.mode

    def on_epoch_end(self, trainer, epoch: int, metrics: dict) -> None:
        """Count evaluations without improvement and stop when a rule fires."""
        value = monitored_value(metrics, self.monitor)
        if value is None:
            return
        reason = None
        if self.check_finite and not math.isfinite(value):
            reason = f"{self.monitor} = {value} is not finite"
        elif self.stopping_threshold is not None and _better(value, self.stopping_threshold,
                                                              self.mode):
            reason = f"{self.monitor} = {value:.4e} passed the threshold {self.stopping_threshold}"
        elif _better(value, self.best_score, self.mode, self.min_delta):
            self.best_score, self.wait = value, 0
        else:
            self.wait += 1
            if self.patience > 0 and self.wait >= self.patience:
                reason = (f"{self.monitor} did not improve for {self.wait} evaluations "
                          f"(best {self.best_score:.4e})")
        if reason is not None:
            self.stopped_epoch, self.reason = epoch, reason
            trainer.should_stop = True
            if trainer.is_main:
                print(f"early stopping after epoch {epoch}: {reason}")


class ProgressBar(Callback):
    """A ``tqdm`` bar over each epoch's batches with the running training loss (rank 0 only)."""

    def __init__(self):
        self._bar = None

    def on_train_epoch_start(self, trainer, epoch: int) -> None:
        """Open the bar."""
        if trainer.is_main:
            from tqdm.auto import tqdm
            try:
                total = len(trainer.train_loader)
            except TypeError:
                total = None
            self._bar = tqdm(total=total, desc=f"epoch {epoch}", leave=False, dynamic_ncols=True)
            self._sum = self._n = 0.0

    def on_train_batch_end(self, trainer, logs: dict, batch_idx: int) -> None:
        """Advance and show the mean batch loss of the epoch so far."""
        if self._bar is not None:
            self._sum += logs["loss"]
            self._n += 1
            self._bar.set_postfix(loss=f"{self._sum / self._n:.4e}", refresh=False)
            self._bar.update(1)

    def on_epoch_end(self, trainer, epoch: int, metrics: dict) -> None:
        """Close the bar."""
        if self._bar is not None:
            self._bar.close()
            self._bar = None
