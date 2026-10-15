.. _howto-lammps:

**********************************
Export to LAMMPS and TorchScript
**********************************

SchNet, DimeNet, GemNet, NequIP, MACE and Allegro export to a self-contained
TorchScript file that needs nothing but ``libtorch`` or ``torch.jit.load``: no
xnn, no Python model code, no config. The scripted models match their eager
variants to about 1e-15.

Export
======

You can export a model checkpoint to its TorchScript format using the command
line interface:

.. code-block:: bash

   xnn export --ckpt runs/exp/best.pt --out deployed.pt
   xnn export --ckpt mace-off23-small --out mace_off.pt     # Any valid model hub name works

or using the Python API:

.. code-block:: python

   from xnn.common.deploy import export_torchscript_potential

   export_torchscript_potential(model, cutoff=5.0, path="deployed.pt")

Options: ``--head`` picks the head of a multi-head checkpoint (LoRA adapters
are always folded in), ``--total-charge`` and ``--spin-multiplicity`` fix
the charge state of the deployed system, and ``--no-dispersion`` leaves out
a dispersion term the checkpoint records as subtracted from its labels
(:ref:`deployment`).

Using the exported model
========================
The export framework offers two entry points for inference. The ``forward``
function, which takes the whole system and builds its own neighbor list from it:

.. code-block:: python

   import torch

   model = torch.jit.load("deployed.pt")

   # cell / pbc optional for molecules
   out = model(pos, atomic_numbers, cell, pbc)

   energy, forces, stress = out["energy"], out["forces"], out["stress"]

The ``forward_lammps(pos, edge_index, cell_shifts, atomic_numbers, cell)``
function, which implements the pair-style interface of the ``pair_nequip`` /
``pair_allegro`` / ``pair_mace`` pattern where the MD engine supplies the
neighbor list. One structure is processed per call with no batch dimension.
Using ``torch.no_grad()`` is fine but using ``torch.inference_mode()`` is not
(because forces are computed with autograd).
