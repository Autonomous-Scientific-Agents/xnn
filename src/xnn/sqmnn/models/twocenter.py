"""Two-center integrals of atom pairs: overlaps, electron repulsion, core attraction.

All integrals of a pair are formed in the diatomic frame (``z`` from atom A
to atom B) and rotated to the molecular frame:

* the ``sp``-``sp`` electron-repulsion integrals with the MNDO multipole
  model (monopole, dipole ``dd``, linear quadrupoles ``2 qq`` and square
  quadrupoles ``qq`` with the additive terms ``am``, ``ad``, ``aq``);
  ``(p_x p_y | p_x p_y)`` follows from rotational invariance, ``((xx|xx) -
  (xx|yy)) / 2``;
* every integral that involves a ``d`` orbital with the Thiel-Voityuk scheme
  (multipoles up to ``L = 2`` of each charge distribution, separations
  ``ddp`` and additive terms ``po``);
* the attraction between the electrons of one atom and the core of the
  other, the core being a monopole with additive term ``po9``.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from ..common.constants import BOHR, HARTREE
from .angular import L_OF, LOCAL, M_OF, MULTIPOLES, ORDERS, multipole_coefficient, pair_type
from .integrals import SPD_CONFIG, config_interaction, config_tensors, local_frames, overlap_local

# per-atom sources of the charge separations and additive terms
D_SOURCES = ("none", "dd", "2qq", "qq", "ddp2", "ddp3", "ddp4", "ddp5", "ddp6", "sq2ddp3")
RHO_SOURCES = ("rho0", "rho1", "rho2") + tuple(f"po{i}" for i in range(1, 10))
_DDP_OF = {"sp": "ddp2", "pp": "ddp3", "sd": "ddp4", "pd": "ddp5", "dd": "ddp6"}
_PO_OF = {"sp": "po2", "pp": "po3", "sd": "po4", "pd": "po5", "dd": "po6"}
_MONO_PO_OF = {"ss": "po1", "pp": "po7", "dd": "po8"}
_D_TYPES = ("sd", "pd", "dd")


def _coef(entries) -> np.ndarray:
    c = np.zeros((9, 9))
    for (i, j), v in entries:
        c[i, j] = v
    return c


@lru_cache(maxsize=None)
def slot_tables():
    """The multipole components ("slots") of an atom and the interacting slot pairs.

    Returns a dict of numpy tables: for each of ``ee_classic``, ``ee_general``,
    ``core_classic``, ``core_general`` the configuration names, the
    separation and additive-term sources and the ``(S, 9, 9)`` coefficients
    over the local orbital pairs; plus the ``(a, b)`` slot pairs whose
    interaction is needed.
    """
    s, z, x, y = 0, 1, 2, 3
    cls = [  # name, config, D source, rho source, entries
        ("Q", "Q", "none", "rho0", [((s, s), 1), ((z, z), 1), ((x, x), 1), ((y, y), 1)]),
        ("Dz", "Dz", "dd", "rho1", [((s, z), 1), ((z, s), 1)]),
        ("Dx", "Dx", "dd", "rho1", [((s, x), 1), ((x, s), 1)]),
        ("Dy", "Dy", "dd", "rho1", [((s, y), 1), ((y, s), 1)]),
        ("Qzz", "Qzz", "2qq", "rho2", [((z, z), 1)]),
        ("Qxx", "Qxx", "2qq", "rho2", [((x, x), 1)]),
        ("Qyy", "Qyy", "2qq", "rho2", [((y, y), 1)]),
        ("Qxz", "Qxz", "qq", "rho2", [((z, x), 1), ((x, z), 1)]),
        ("Qyz", "Qyz", "qq", "rho2", [((z, y), 1), ((y, z), 1)]),
    ]
    groups = ({"Q", "Dz", "Qzz", "Qxx", "Qyy"}, {"Dx", "Qxz"}, {"Dy", "Qyz"})
    cls_pairs = [(a, b) for a in range(len(cls)) for b in range(len(cls))
                 if any(cls[a][0] in g and cls[b][0] in g for g in groups)]

    gen = []   # config, D source, rho source, coefficients, m, type
    for t in ("ss", "sp", "pp", "sd", "pd", "dd"):
        for L in ORDERS[t]:
            for comp in MULTIPOLES[L]:
                entries = []
                for i, a in enumerate(LOCAL):
                    for j, b in enumerate(LOCAL):
                        if pair_type(L_OF[a], L_OF[b]) != t:
                            continue
                        v = multipole_coefficient(a, b, comp)
                        if v != 0.0:
                            entries.append(((i, j), v))
                d_src = "none" if L == 0 else _DDP_OF[t]
                rho_src = _MONO_PO_OF[t] if L == 0 else _PO_OF[t]
                gen.append((SPD_CONFIG[comp], d_src, rho_src, _coef(entries), M_OF[comp], t))
    gen_pairs = [(a, b) for a in range(len(gen)) for b in range(len(gen))
                 if gen[a][4] == gen[b][4] and (gen[a][5] in _D_TYPES or gen[b][5] in _D_TYPES)]

    core_cls = [
        ("Q", "none", "po1", [((s, s), 1)]),
        ("Q", "none", "po7", [((z, z), 1), ((x, x), 1), ((y, y), 1)]),
        ("Dz", "ddp2", "po2", [((s, z), 1), ((z, s), 1)]),
        ("Qzz", "sq2ddp3", "po3", [((z, z), 1)]),
        ("Qxx", "sq2ddp3", "po3", [((x, x), 1)]),
        ("Qyy", "sq2ddp3", "po3", [((y, y), 1)]),
    ]
    core_gen = [g for g in gen if g[4] == 0 and g[5] in _D_TYPES]

    def pack(rows, coef_index, cfg_index=0, d_index=1, rho_index=2):
        return dict(
            config=[r[cfg_index] for r in rows],
            d_src=np.array([D_SOURCES.index(r[d_index]) for r in rows]),
            rho_src=np.array([RHO_SOURCES.index(r[rho_index]) for r in rows]),
            coef=np.stack([r[coef_index] if isinstance(r[coef_index], np.ndarray)
                           else _coef(r[coef_index]) for r in rows]),
        )

    return dict(
        ee_classic=pack(cls, 4, 1, 2, 3), ee_classic_pairs=np.array(cls_pairs),
        ee_general=pack(gen, 3, 0, 1, 2), ee_general_pairs=np.array(gen_pairs),
        core_classic=pack(core_cls, 3, 0, 1, 2), core_general=pack(core_gen, 3, 0, 1, 2),
    )


def atom_sources(q: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
    """Per-element separation ``(n, len(D_SOURCES))`` and additive-term
    ``(n, len(RHO_SOURCES))`` tables from :func:`~.derived.element_quantities`."""
    zero = torch.zeros_like(q["dd"])
    ddp, po = q["ddp"], q["po"]
    d = torch.stack([zero, q["dd"], 2.0 * q["qq"], q["qq"], ddp[:, 2], ddp[:, 3], ddp[:, 4],
                     ddp[:, 5], ddp[:, 6], math.sqrt(2.0) * ddp[:, 3]], dim=-1)
    rho = torch.stack([q["rho0"], q["rho1"], q["rho2"]] + [po[:, i] for i in range(1, 10)], dim=-1)
    return d, rho


def _ee_block(kind: str, pairs_key: str, k: int, dA: Tensor, rA: Tensor, dB: Tensor, rB: Tensor,
              r: Tensor) -> Tensor:
    tab = slot_tables()
    t, pairs = tab[kind], tab[pairs_key]
    dtype, device = r.dtype, r.device
    cfg_q, cfg_u = config_tensors(t["config"], dtype, device)
    a, b = pairs[:, 0], pairs[:, 1]
    da = dA[:, t["d_src"][a]]
    db = dB[:, t["d_src"][b]]
    rho = rA[:, t["rho_src"][a]] + rB[:, t["rho_src"][b]]
    K = config_interaction(cfg_q[a], cfg_u[a], da, cfg_q[b], cfg_u[b], db, r, rho)   # (P, S)
    coef = torch.as_tensor(t["coef"][:, :k, :k], dtype=dtype, device=device)
    return torch.einsum("ps,sij,skl->pijkl", K, coef[a], coef[b])


def _core_block(kind: str, k: int, d_e: Tensor, r_e: Tensor, rho_core: Tensor, r: Tensor,
                electrons_on_a: bool) -> Tensor:
    t = slot_tables()[kind]
    dtype, device = r.dtype, r.device
    q, u = config_tensors(t["config"], dtype, device)
    q0, u0 = config_tensors(["Q"] * len(t["config"]), dtype, device)
    de = d_e[:, t["d_src"]]
    rho = r_e[:, t["rho_src"]] + rho_core.unsqueeze(-1)
    zero = torch.zeros_like(de)
    if electrons_on_a:
        K = config_interaction(q, u, de, q0, u0, zero, r, rho)
    else:
        K = config_interaction(q0, u0, zero, q, u, de, r, rho)
    coef = torch.as_tensor(t["coef"][:, :k, :k], dtype=dtype, device=device)
    return torch.einsum("ps,sij->pij", K, coef)


def _overlap_scheme(qa: Dict, qb: Dict, ia: Tensor, ib: Tensor) -> Tensor:
    """Series truncation scheme of the overlaps of each pair (see
    :func:`~.integrals.series_order`): pairs of ``sp`` elements H-Cl (except He
    and Ne) use the scheme of their row combination, all others 0."""
    def row_class(q):
        out = []
        for z, no in zip(q["z"], q["n_orb"]):
            if z > 17 or no > 4 or z in (2, 10):
                out.append(0)
            else:
                out.append(1 if z == 1 else (2 if z <= 10 else 3))
        return torch.as_tensor(out, device=ia.device)
    ra, rb = row_class(qa)[ia], row_class(qb)[ib]
    lo, hi = torch.minimum(ra, rb), torch.maximum(ra, rb)
    # highest auxiliary index of the row combination: H-H 2, H-2nd 3, H-3rd 4,
    # 2nd-2nd 4, 2nd-3rd 5, 3rd-3rd 6
    table = torch.as_tensor([[0, 0, 0, 0], [0, 2, 3, 4], [0, 3, 4, 5], [0, 4, 5, 6]], device=ia.device)
    return torch.where(lo > 0, table[lo, hi], torch.zeros_like(lo))


def _overlaps_local(qa: Dict, qb: Dict, ia: Tensor, ib: Tensor, r: Tensor, k: int) -> Tensor:
    """Local-frame overlaps ``(P, k, k)`` of the pairs (element indices ``ia``, ``ib``)."""
    P = r.shape[0]
    S = r.new_zeros((P, k, k))
    def zeta(q):
        z = torch.stack([q["zs"], q["zp"], q["zd_overlap"]], dim=-1)
        # shells an element does not have get a harmless exponent (masked later)
        return torch.where(z > 0, z, torch.ones_like(z))
    za, zb = zeta(qa)[ia], zeta(qb)[ib]
    scheme = _overlap_scheme(qa, qb, ia, ib)
    pqn_a = torch.as_tensor(qa["pqn"], device=r.device)[ia]
    pqn_b = torch.as_tensor(qb["pqn"], device=r.device)[ib]
    labels = LOCAL[:k]
    for i, la_lab in enumerate(labels):
        for j, lb_lab in enumerate(labels):
            if M_OF[la_lab] != M_OF[lb_lab]:
                continue
            la, lb, m = L_OF[la_lab], L_OF[lb_lab], abs(M_OF[la_lab])
            combos = torch.stack([pqn_a[:, la], pqn_b[:, lb]], dim=-1)
            for na, nb in torch.unique(combos, dim=0).tolist():
                sel = (combos[:, 0] == na) & (combos[:, 1] == nb)
                idx = sel.nonzero().squeeze(-1)
                val = overlap_local(na, la, za[idx, la], nb, lb, zb[idx, lb], r[idx], m, scheme[idx])
                S = S.index_put((idx, torch.full_like(idx, i), torch.full_like(idx, j)), val)
    return S


def pair_integrals(q: Dict[str, Tensor], ea: Tensor, eb: Tensor, vec: Tensor, k: int,
                   need_general: bool) -> Dict[str, Tensor]:
    """Molecular-frame integrals of atom pairs.

    Parameters
    ----------
    q : dict
        Derived element quantities (:func:`~.derived.element_quantities`).
    ea, eb : Tensor
        Element index (into ``q``) of atoms A and B of every pair ``(P,)``.
    vec : Tensor
        ``R_B - R_A`` in Angstrom ``(P, 3)``.
    k : int
        Orbitals per atom of the batch (1, 4 or 9; padded).
    need_general : bool
        Whether any atom has ``d`` orbitals.

    Returns
    -------
    dict
        ``S`` overlaps ``(P, k, k)``; ``W`` electron repulsion ``(mu nu | la si)``
        ``(P, k, k, k, k)`` with ``mu nu`` on A; ``VA`` / ``VB`` attraction of
        the electrons of A by the core of B (and vice versa) ``(P, k, k)``;
        ``gab`` the ``(s_A s_A | s_B s_B)`` core integral ``(P,)``; ``r``
        the distances in Angstrom. Energies in eV.
    """
    r_ang = vec.norm(dim=-1)
    r = r_ang / BOHR
    d_src, rho_src = atom_sources(q)
    dA, rA, dB, rB = d_src[ea], rho_src[ea], d_src[eb], rho_src[eb]
    T = local_frames(vec, k)

    W = _ee_block("ee_classic", "ee_classic_pairs", k, dA, rA, dB, rB, r)
    if k >= 4:
        hpp = 0.5 * (W[:, 2, 2, 2, 2] - W[:, 2, 2, 3, 3])
        for idx in ((2, 3, 2, 3), (2, 3, 3, 2), (3, 2, 2, 3), (3, 2, 3, 2)):
            W[(slice(None),) + idx] = hpp
    if need_general and k == 9:
        W = W + _ee_block("ee_general", "ee_general_pairs", k, dA, rA, dB, rB, r)

    po9A, po9B = q["po"][ea, 9], q["po"][eb, 9]
    tore_a, tore_b = q["tore"][ea], q["tore"][eb]
    VA = _core_block("core_classic", k, dA, rA, po9B, r, True)
    VB = _core_block("core_classic", k, dB, rB, po9A, r, False)
    if need_general and k == 9:
        VA = VA + _core_block("core_general", k, dA, rA, po9B, r, True)
        VB = VB + _core_block("core_general", k, dB, rB, po9A, r, False)
    VA = -tore_b[:, None, None] * VA
    VB = -tore_a[:, None, None] * VB
    gab = HARTREE / torch.sqrt(r * r + (po9A + po9B) ** 2)

    S = _overlaps_local(q, q, ea, eb, r, k)
    S = torch.where((r_ang * r_ang < 225.0)[:, None, None], S, torch.zeros_like(S))

    # rotate to the molecular frame
    S = torch.einsum("pai,pab,pbj->pij", T, S, T)
    VA = torch.einsum("pai,pab,pbj->pij", T, VA, T)
    VB = torch.einsum("pai,pab,pbj->pij", T, VB, T)
    W = torch.einsum("pai,pabcd->pibcd", T, W)
    W = torch.einsum("pbj,pibcd->pijcd", T, W)
    W = torch.einsum("pck,pijcd->pijkd", T, W)
    W = torch.einsum("pdl,pijkd->pijkl", T, W)
    return dict(S=S, W=W, VA=VA, VB=VB, gab=gab, r=r_ang)
