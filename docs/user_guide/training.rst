.. _training:

********
Training
********

The Trainer
===========
:class:`~xnn.common.train.trainer.Trainer` owns the whole loop:

.. code-block:: python

   from xnn.common.train import Trainer

   trainer = Trainer(cfg, train_set, val_set, test_set)   # val and test optional
   metrics = trainer.fit()                                # {"train": ..., "val": ..., "test": ...}

It builds the model from ``cfg.model``, wraps it in
:class:`~xnn.common.models.outputs.ForceStressOutput` (force and stress
heads follow the nonzero loss weights), resolves the device, and sets up the
optimizer, scheduler and batched loaders. When no validation or test set is
given, ``data.val_fraction`` and ``data.test_fraction`` carve them out of
the training set with the run's seed.

``fit()`` validates every epoch and writes ``best.pt`` (lowest validation
loss, or another monitored value) and ``last.pt`` to ``cfg.output_dir``. A test set is evaluated once
after the last epoch. A checkpoint holds the state dict and the config, and
loads with ``from_pretrained(path)``; to score the best checkpoint on the
test set, load it and call :meth:`~xnn.common.train.trainer.Trainer.evaluate`.
``trainer.save_pretrained("my-model")`` writes a portable model directory
(:ref:`howto-pretrained-models`).

Each epoch line prints the loss and its terms (``energy``, ``force``,
``stress``, ...) for the training and validation sets, then the validation
MAE and RMSE. Every value is a
mean over the whole set, weighted like the loss (structures for energy,
atoms for forces), so batch sizes and partially labelled batches do not
skew it; the training values are gathered while the weights change.

The training loader reshuffles every epoch. To batch differently, for
instance molecules and periodic cells in separate batches of their own
sizes, pass ``train_batch_sampler`` (and ``val_batch_sampler`` /
``test_batch_sampler``): any iterable of index lists over the dataset as
given, with an optional ``set_epoch(epoch)`` that the trainer calls before
every epoch.

.. code-block:: python

   trainer = Trainer(cfg, train_set, val_set, train_batch_sampler=my_sampler)

Metrics, checkpoints and stopping
=================================
Besides the loss terms, every split reports the MAE and RMSE of each
labelled property (``energy_mae`` per atom, ``forces_rmse``, ``stress_mae``,
``dipole_*``, ``polarizability_*``, ``charges_*``), pooled over the whole set.
The ``trainer`` section decides what drives the run:

.. code-block:: yaml

   trainer:
     metrics: [mae, rmse]
     monitor: val/forces_mae       # picks best.pt, drives plateau and early stopping
     mode: min
     val_every_n_epochs: 1
     save_every_n_epochs: 0        # > 0 also keeps epoch_<n>.pt
     early_stopping_patience: 20   # validations without improvement; 0 = off
     early_stopping_min_delta: 0.0
     early_stopping_threshold: 0.02  # stop once forces_mae < 0.02

Any logged key can be monitored. Without a validation set ``val/`` reads as
``train/``. In Python these are callbacks
(:mod:`~xnn.common.train.callbacks`), and you can pass your own:

.. code-block:: python

   from xnn.common.train import Callback, EarlyStopping, ModelCheckpoint

   class Snapshot(Callback):
       def on_epoch_end(self, trainer, epoch, metrics):
           if epoch % 50 == 0:
               trainer.save(f"runs/exp/epoch_{epoch}.pt")

   trainer = Trainer(cfg, train_set, val_set, callbacks=[
       ModelCheckpoint(monitor="val/energy_rmse", filename="best_energy.pt"),
       EarlyStopping(monitor="val/forces_mae", patience=30),
       Snapshot()])

A callback has ``on_fit_start``, ``on_train_epoch_start``,
``on_train_batch_end``, ``on_epoch_end`` (set ``trainer.should_stop`` to
end the run), ``on_train_end`` and ``on_fit_end``. The epoch metrics it sees
are the dataset-level values, identical on every rank.

Logging
=======
The metrics go to the loggers in ``trainer.loggers``: ``console`` (the
epoch lines), ``csv`` (``metrics.csv`` and ``hparams.yaml`` in
``output_dir``) and ``tensorboard`` (``output_dir/tensorboard``, needs the
``tensorboard`` extra).

.. code-block:: yaml

   trainer:
     loggers: [console, csv, tensorboard]
     log_every_n_epochs: 1
     log_every_n_steps: 0        # > 0 also logs training values every n steps
     step_average: batch         # step values: the batch, or running (the epoch so far)
     progress_bar: false

Every logged epoch carries ``epoch``, ``lr``, ``train/<key>`` and
``val/<key>``, the test set ``test/<key>``, and step values
``train_step/<key>``, all at the optimizer step count. In Python, pass
loggers to the trainer, or subclass
:class:`~xnn.common.train.loggers.Logger`:

.. code-block:: python

   from xnn.common.train import ConsoleLogger, Logger, TensorBoardLogger

   class PrintBest(Logger):
       best = float("inf")

       def log_metrics(self, metrics, step):
           if metrics.get("val/loss", self.best) < self.best:
               self.best = metrics["val/loss"]
               print(f"new best {self.best:.4e} at step {step}")

   trainer = Trainer(cfg, train_set, val_set,
                     logger=[ConsoleLogger(errors=False), TensorBoardLogger("tb_logs"), PrintBest()])

``TensorBoardLogger.experiment`` is the ``SummaryWriter`` itself, for
histograms or images, and ``trainer.log_metrics(...)`` sends extra values to
every logger. Only rank 0 logs.

The loss
========
A weighted sum of the per-property errors:

.. math::

   \mathcal{L} = w_E \, \mathcal{L}_\text{energy}
               + w_F \, \mathcal{L}_\text{forces}
               + w_\sigma \, \mathcal{L}_\text{stress}

with the defaults :math:`w_E = 1`, :math:`w_F = 10`, :math:`w_\sigma = 0`.
A property enters only when the data provides it and its weight is nonzero.
Train on forces whenever you have them; it is far more data-efficient than
energies alone. The energy term is per atom and averages over structures;
the force term averages over atoms.

- **Per-structure weights.** A ``weight`` in the structure dictionary
  scales that structure's share of every term, for instance to balance a
  sparse set against a dense scan. Uniform weights reproduce the plain loss.
- **Term and atom weights.** ``energy_weight`` / ``force_weight`` /
  ``stress_weight`` scale one term of one structure, and
  ``atom_force_weight`` the force error of each atom, for instance to train
  on the buried atoms of a cluster only (:ref:`howto-weight-force-labels`).
  Weight 0 drops the label; structures without an energy are masked the
  same way.
- **Huber tails.** ``optim.huber_delta`` (or the per-term
  ``huber_delta_energy`` / ``_forces`` / ``_stress``) replaces the squared
  error by a function that is quadratic up to :math:`\delta` and linear
  beyond, which caps the pull of a few large residuals. It is scaled to
  agree with :math:`d^2` below :math:`\delta`, twice the textbook Huber, so
  halve ``delta`` values ported from MACE.
- **Several heads.** With head labels in the batch the loss sums each head's
  terms over that head's own structures; ``optim.head_weights`` gives a head
  its own weights (:ref:`howto-finetune`).

Optimizer, clipping, weight averaging
=====================================
``optim.optimizer`` selects Adam, AMSGrad or AdamW (decoupled weight decay),
``optim.clip_grad`` clips the gradient norm, and ``optim.ema_decay`` keeps an
exponential moving average of the weights that validation and the
checkpoints use. ``optim.freeze`` and ``optim.train_only`` select trainable
parameters by name pattern. All are off by default; the fine-tuning recipes
rely on them. ``model.pretrained``, ``model.lora`` and ``model.heads`` turn
the same trainer into a fine-tuning run (:ref:`howto-finetune`).

Several GPUs or nodes
=====================
Distributed training is native PyTorch DDP and needs no code or config
changes, only a launcher that sets ``RANK``, ``LOCAL_RANK`` and
``WORLD_SIZE``:

.. code-block:: bash

   torchrun --nproc-per-node 2 -m xnn train --config train.yaml

   torchrun --nnodes 2 --nproc-per-node 4 \
            --rdzv-backend c10d --rdzv-endpoint "$HEAD_NODE":29500 \
            -m xnn train --config train.yaml

Each rank is pinned to its GPU, every loader is sharded, metrics are
all-reduced so the best-checkpoint decision and the scheduler agree across
ranks, and only rank 0 prints and writes. Validation and test shards are
disjoint, so their metrics equal a serial run's; the training shards are
padded to equal length, repeating up to ``WORLD_SIZE - 1`` structures per
epoch. A custom batch sampler must yield each rank's own share. Checkpoints hold the bare model
and are interchangeable with serial runs. ``data.batch_size`` is per
process, so the effective batch is ``batch_size × WORLD_SIZE``. Sharded
strategies (FSDP, DeepSpeed) are not supported: force and stress losses
need a double backward, which only DDP provides.

Reproducibility
===============
``cfg.seed`` seeds the run. Bit-exact results across devices and CUDA
versions are not guaranteed by PyTorch itself.
