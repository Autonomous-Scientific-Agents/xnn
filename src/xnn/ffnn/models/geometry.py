"""Shared bonded-term geometry for fixed-topology force fields.

Minimum-image displacement vectors, the dihedral cosine and a
differentiable-at-zero geometric mean, used by every fixed-topology model
(OPLS, DREIDING). All functions operate on a batched
:class:`~xnn.common.data.AtomicGraph` plus explicit atom-index tensors, so
bonded terms are correct for molecules wrapped across periodic boundaries
while gradients still flow to positions and cell.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from xnn.common.data import AtomicGraph

# Small numerical guard applied under square roots.
TINY = 1.0e-12


def geometric_mean(x: Tensor, y: Tensor) -> Tensor:
    """Geometric mean with a differentiable zero (for zero-LJ hydrogens).

    Parameters
    ----------
    x, y : Tensor
        Non-negative parameter values, gathered per pair.

    Returns
    -------
    Tensor
        ``sqrt(x * y)``, exactly zero (with zero gradient) where the
        product vanishes.
    """
    prod = x * y
    safe = torch.sqrt(torch.clamp(prod, min=TINY))
    return torch.where(prod > 0.0, safe, torch.zeros_like(prod))


def minimum_image_vectors(vec: Tensor, cell: Tensor, pbc: Optional[Tensor] = None) -> Tensor:
    """Find the shortest images in a possibly triclinic cell.

    Parameters
    ----------
    vec : Tensor
        Displacement vectors ``(M, 3)``.
    cell : Tensor
        Row-vector lattice ``(3, 3)``.
    pbc : Tensor or None, optional
        Periodic axes ``(3,)``; ``None`` means all axes.

    Returns
    -------
    Tensor
        Shortest vectors ``(M, 3)``, differentiable in ``vec`` and ``cell``
        through fixed integer image shifts.
    """
    periodic = (torch.ones(3, dtype=torch.bool, device=vec.device) if pbc is None
                else pbc.to(device=vec.device, dtype=torch.bool))
    if vec.shape[0] == 0 or not bool(periodic.any()):
        return vec
    with torch.no_grad():
        lattice = cell.detach()
        inv = torch.linalg.inv(lattice)
        frac = vec.detach() @ inv
        shift = -torch.round(frac) * periodic.to(vec.dtype)
        wrapped = vec.detach() + shift @ lattice
        gram = lattice @ lattice.t()
        if bool((gram - torch.diag(torch.diagonal(gram)) != 0).any()):
            best2 = (wrapped * wrapped).sum(-1)
            # Any shorter image has |fractional_i| <= R |reciprocal_i|.
            # This bounds every candidate, including in unreduced cells.
            radius = best2.max().sqrt()
            bounds = torch.ceil(radius * torch.linalg.norm(inv, dim=0)
                                + (frac + shift).abs().amax(dim=0)).to(torch.long)
            bounds = bounds * periodic.to(torch.long)
            ranges = [torch.arange(-int(b), int(b) + 1, device=vec.device) for b in bounds]
            offsets = torch.cartesian_prod(*ranges).to(vec.dtype)
            best_shift = shift.clone()
            for offset in offsets:
                candidate = wrapped + offset @ lattice
                dist2 = (candidate * candidate).sum(-1)
                better = dist2 < best2
                best_shift = torch.where(better[:, None], shift + offset, best_shift)
                best2 = torch.minimum(best2, dist2)
            shift = best_shift
    return vec + shift @ cell


def pair_vectors(data: AtomicGraph, a: Tensor, b: Tensor) -> Tensor:
    """Minimum-image displacement vectors ``pos[b] - pos[a]``.

    For periodic structures the integer image shift is recomputed from the
    closest lattice image (with detached integer shifts), so bonded terms are correct
    for molecules wrapped across the boundary while gradients still flow to
    positions and cell.

    Parameters
    ----------
    data : AtomicGraph
        The batched graph.
    a, b : Tensor
        Atom indices of shape ``(M,)``.

    Returns
    -------
    Tensor
        Displacements of shape ``(M, 3)``.
    """
    # formed in the geometry's dtype, returned in the model's (see
    # AtomicGraph.edge_vectors)
    vec = data.pos[b] - data.pos[a]
    if data.cell is None or vec.shape[0] == 0:
        return vec.to(data.model_dtype)
    result = vec.clone()
    for i in torch.unique(data.batch[a]).tolist():
        mask = data.batch[a] == i
        pbc = data.pbc[i] if data.pbc is not None else None
        result[mask] = minimum_image_vectors(vec[mask], data.cell[i].to(vec.dtype), pbc)
    return result.to(data.model_dtype)


def dihedral_cos(data: AtomicGraph, idx: Tensor) -> Tensor:
    """Cosine of the dihedral angle over each atom quadruple.

    Uses the plane-normal formula with ``phi = 0`` at *cis*. Only
    ``cos phi`` is returned; periodic torsion terms are even in ``phi``, so
    the multiple angles come from Chebyshev identities and no
    ``arccos``/``atan2`` (whose derivatives are singular at planar
    geometries) enters the graph.

    Parameters
    ----------
    data : AtomicGraph
        The batched graph.
    idx : Tensor
        Atom indices of shape ``(4, M)``.

    Returns
    -------
    Tensor
        ``cos phi`` of shape ``(M,)``.
    """
    b1 = pair_vectors(data, idx[0], idx[1])
    b2 = pair_vectors(data, idx[1], idx[2])
    b3 = pair_vectors(data, idx[2], idx[3])
    n1 = torch.cross(b1, b2, dim=1)
    n2 = torch.cross(b2, b3, dim=1)
    denom = torch.sqrt((n1 * n1).sum(1) * (n2 * n2).sum(1) + TINY)
    return (n1 * n2).sum(1) / denom
