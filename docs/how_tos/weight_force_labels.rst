.. _howto-weight-force-labels:

******************************
Weight Energy and Force Labels
******************************

A large cluster cut from a bulk snapshot is cheaper to label than the periodic
cell, but only its buried atoms see a bulk-like environment. Per-atom force
weights train on those atoms and keep the rest as environment, and term
weights decide whether the cluster's energy counts.

The weights
===========
Every weight is optional and defaults to 1.

.. code-block:: python

   {
       "weight": 1.0,                  # the structure, every term
       "energy_weight": 0.0,           # this structure's energy only
       "force_weight": 1.0,            # its forces only
       "stress_weight": 1.0,           # its stress only
       "atom_force_weight": [...],     # (N,), each atom's force error
   }

In an extxyz file the structure weights are ``info`` keys and the atom weights
a per-atom column:

.. code-block:: text

   Properties=species:S:1:pos:R:3:forces:R:3:atom_force_weight:R:1 energy=-1234.5 energy_weight=0.0 pbc="F F F"
   Li   0.000  0.000  0.000    0.012 -0.034  0.101    1.0
   O    2.010  0.130 -0.220    0.150  0.020 -0.090    0.7
   C    9.870  1.040  0.330   -0.410  0.270  0.060    0.0

Clusters and periodic cells can share a file and a batch; frames without the
keys keep weight 1.

What the loss does
==================
Atom :math:`i` of structure :math:`b` gets
:math:`u_i = w_b \, f_b \, a_i` (``weight``, ``force_weight``,
``atom_force_weight``), and the force term is the weighted mean

.. math::

   \mathcal{L}_\text{forces} = \frac{\sum_i u_i \, |\mathbf{F}_i - \mathbf{F}_i^\text{ref}|^2}{3 \sum_i u_i}.

- A weighted atom counts like an atom of a periodic cell, so
  ``optim.force_weight`` and the learning rate keep their meaning however
  many surface atoms a batch holds.
- An atom of weight 0 stays in the graph: the atoms around it still see it,
  only its own force label is dropped. It still costs compute.
- The energy and stress terms take ``weight`` times their term weight. A
  term whose weights are all zero in a batch is skipped.
- Structures without an energy label are masked out of the energy term, like
  missing forces and stress.

Reference energies fitted from the data (``atomic_energies: average`` or
``estimated``) skip energies that are missing or weighted to zero. The
validation and test losses use the same weights; benchmark metrics skip
labels of weight 0 and score the others unweighted.

Weight a carved cluster
=======================
Buriedness is your choice and xnn only reads the numbers. For a cluster cut as
a sphere, a cosine ramp  works in the depth below the surface:

.. code-block:: python

   import numpy as np
   from ase.io import read, write

   def depth_weights(atoms, d_buf, d_core):
       """0 within d_buf of the surface, 1 deeper than d_core, a cosine ramp between."""
       r = np.linalg.norm(atoms.positions - atoms.positions.mean(0), axis=1)
       x = np.clip((r.max() - r - d_buf) / (d_core - d_buf), 0.0, 1.0)
       return 0.5 * (1.0 - np.cos(np.pi * x))

   frames = read("clusters.extxyz", ":")
   for atoms in frames:
       atoms.arrays["atom_force_weight"] = depth_weights(atoms, d_buf=4.0, d_core=8.0)
       atoms.info["energy_weight"] = 0.0
   write("clusters_weighted.extxyz", frames)

Take ``d_buf`` and ``d_core`` from a convergence test: the depth at which the
reference forces on a central atom match those of the periodic system. The
buffer is set by the reference method and long-range electrostatics, not by
the model's cutoff or number of layers.

Cluster energies
================
A cluster's total energy includes its surface. Start with
``energy_weight: 0`` on clusters and keep the energies (and stress) of the
periodic cells, which anchor the bulk energetics. If too few periodic cells
remain to cover the compositions that matter, give cluster energies a small
weight (for example 0.1) and judge it on held-out periodic cells.
