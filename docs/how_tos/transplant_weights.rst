.. _howto-transplant:

**************************************
Transplant Weights from Upstream Codes
**************************************

Every xnn model with a reference implementation loads the original code's
weights and reproduces its energies and forces to round-off error. That is how
the xnn implementations are verified (:ref:`fidelity`), and it is how a model
trained elsewhere comes into xnn.

Published checkpoints need no transplant
========================================
The MACE and AIMNet2 pre-trained foundation models are offered by their
developers. They can be converted into xnn models in one call. Ensemble models
such as torchani's ANI can load through presets:

.. code-block:: python

   from xnn.common.models import from_pretrained
   from xnn.dnn.models import ANI

   # Load pretrained models (MACE and AIMNet2)
   mace = from_pretrained("mace-off23-small")
   aim = from_pretrained("aimnet2")

   # Load the ANI-2x model and its ensemble of 8 networks
   ani = ANI.ani2x()

See :ref:`howto-pretrained-models` for more information. 

A `RuNNer <https://gitlab.com/runner-suite/runner2>`_ model directory
(``input.nn`` with the weights and scaling files of a 2G, 3G or 4G HDNNP) can
also be loaded directly into xnn:

.. code-block:: python

   from xnn.dnn.common.runner import load_runner_model

   model = load_runner_model("runner_model/")

The pattern for using external upstream weights
===============================================
1. Build the xnn model with the upstream hyperparameters (cutoff, angular
   order, channels, layers, radial basis, average neighbors, per-species
   energies and scales),
2. Rename the upstream state dict to the xnn parameter names,
3. Load it and compare on a batch of datapoints to verify that the energies,
   forces or other target properties match the reference code's outputs. The
   following example shows the transplant for a MACE model trained in the
   original code and verified in xnn:

.. code-block:: python

   # Load the reference model and its weights
   missing, unexpected = model.load_state_dict(mapped_state, strict=False)

   # Check that all parameters were found and no unexpected ones were present
   assert not missing and not unexpected

   # Compare the energies on a batch of datapoints to verify that the transplant
   # worked correctly
   torch.testing.assert_close(model(batch)["energy"], reference_energy)

The same method can be applied in reverse to load xnn-trained weights into the
reference code. Each ``examples/fidelity_checks/<model>_verification`` notebook
ends with the transplant for its original model.
