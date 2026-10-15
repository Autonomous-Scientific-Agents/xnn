.. _howto-upstream-configs:

***********************
Reuse Upstream Configs
***********************

xnn allows automatic translation of config keywords from various upstream YAML
files. A per-model translation table (:mod:`xnn.common.config.translate`)
converts the foreign keys into their xnn-native counterparts when the config
file loads. Below, we show two examples from MACE and NequIP configs where the
keys from the upstream models are translated into xnn-native spellings (shown in
comments).

.. code-block:: yaml

   model:                       # MACE CLI spellings
     name: mace
     r_max: 4.0                 # -> cutoff
     num_radial_basis: 8        # -> n_rbf
     atomic_numbers: [18]       # -> species
     E0s: {18: -0.05}           # -> atomic_energies

.. code-block:: yaml

   model:                       # NequIP spellings
     name: nequip
     r_max: 4.0                 # -> cutoff
     num_layers: 3              # -> n_interactions
     num_features: 32           # -> n_features
     chemical_symbols: [Ar]     # -> species

In cases where the key spellings of the same variable from both xnn and an
upstream model collide, the xnn one wins. Values are coerced too: per-species
energies can be a list, a ``{Z: E0}`` mapping or a string, and ``species``
accepts symbols or numbers.

Extend xnn's translation table
==============================

If xnn does not yet support the translation of a particular upstream model, you
can extend the translation table with a new entry. 

.. code-block:: python

   from xnn.common.config.translate import register_key_translation

   register_key_translation("mymodel", {"r_cut": "cutoff", "n_bessel": "n_rbf"})

Once registered, the translation table applies to every config with ``model.name
== "mymodel"``, whatever the frontend.
