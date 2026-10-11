"""Molecular-mechanics and post-SCF corrections of the NDDO methods (kcal/mol).

* :func:`bond_topology` -- which atoms are bonded (covalent radii with pair
  safety factors), the connectivity the corrections are defined on;
* :func:`peptide_torsion` -- the ``k sin^2(phi)`` barrier added to every
  ``X-N-C=O`` torsion of an amide (all PM methods except PM6-ORG);
* :func:`yne_correction` -- 12 kcal/mol per acetylenic C-C bond, switched
  off smoothly between 1.21 and 1.33 Angstrom (PM6, PM7, PM6-ORG);
* :func:`nsp2_correction` -- stabilization of planar secondary and tertiary
  amine nitrogens, ``-0.5 exp(-10 (2 pi - sum of angles))`` (PM6, eq 12 of
  the PM6 paper);
* :func:`si_o_h_correction` -- the Si-O-H bending term of PM7.

Discrete choices (bonds, groups) are made on the detached geometry; the
energies are differentiable functions of the positions.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import torch
from torch import Tensor

from ..common import elements as el


def bond_topology(z: Sequence[int], pos: Tensor) -> List[List[int]]:
    """Bonded neighbors of every atom of one molecule.

    Two atoms are bonded when closer than ``f (r_i + r_j)`` with the covalent
    radii of :data:`~xnn.sqmnn.common.elements.COVALENT_RADIUS` (alkali and
    alkaline-earth metals count with radius -1) and ``f = 1.1``, except 1.25
    for C-H, 1.0 for B-N, 1.2 for C-C, C-N and S-S; at most 15 bonds per
    atom, and a hydrogen with several bonds keeps only those to heavy atoms.
    """
    n = len(z)
    xyz = pos.detach().tolist()
    radius = [(-1.0 if v in el.METALS else (el.COVALENT_RADIUS[v] or 999.0)) for v in z]
    bonds: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i):
            lo, hi = min(z[i], z[j]), max(z[i], z[j])
            f = 1.1
            if lo == 1 and hi == 6:
                f = 1.25
            elif lo == 5 and hi == 7:
                f = 1.0
            elif lo == 6 and hi < 8:
                f = 1.2
            elif lo == 16 and hi == 16:
                f = 1.2
            d2 = sum((xyz[i][c] - xyz[j][c]) ** 2 for c in range(3))
            if d2 < (f * (radius[i] + radius[j])) ** 2 and len(bonds[i]) < 15 and len(bonds[j]) < 15:
                bonds[i].append(j)
                bonds[j].append(i)
    for i in range(n):
        if z[i] == 1 and len(bonds[i]) >= 2:
            bonds[i] = [j for j in bonds[i] if z[j] != 1]
    return bonds


def _distance(pos: Tensor, i: int, j: int) -> Tensor:
    return (pos[i] - pos[j]).norm()


def _angle(pos: Tensor, i: int, j: int, k: int) -> Tensor:
    """Angle ``i-j-k`` (radians) from the three distances, as MOPAC's bangle."""
    d2ij = (pos[i] - pos[j]).pow(2).sum()
    d2jk = (pos[j] - pos[k]).pow(2).sum()
    d2ik = (pos[i] - pos[k]).pow(2).sum()
    c = 0.5 * (d2ij + d2jk - d2ik) / torch.sqrt(d2ij * d2jk)
    return torch.acos(c.clamp(-1.0, 1.0))


def _sin2_dihedral(pos: Tensor, i: int, j: int, k: int, l: int) -> Tensor:
    b1, b2, b3 = pos[j] - pos[i], pos[k] - pos[j], pos[l] - pos[k]
    n1, n2 = torch.cross(b1, b2, dim=0), torch.cross(b2, b3, dim=0)
    cos = (n1 * n2).sum() / (n1.norm() * n2.norm())
    return 1.0 - cos * cos


def peptide_torsion(z: Sequence[int], pos: Tensor, barrier: float) -> Tensor:
    """Per-atom ``barrier * sin^2`` terms of the amide torsions (kcal/mol).

    Each H-N-C=O group found by distance (C=O <= 1.3, C-N <= 1.6, N-H <= 1.3,
    N-X <= 1.7 Angstrom) contributes the torsions O=C-N-X and O=C-N-H; the
    energy is assigned to the nitrogen.
    """
    out = pos.new_zeros(len(z))
    if barrier == 0.0:
        return out
    xyz = pos.detach()
    n = len(z)

    def d(a, b):
        return float((xyz[a] - xyz[b]).norm())

    used_n = set()
    for j in range(n):
        if z[j] != 6:
            continue
        found = False
        for i in range(n):
            if found or z[i] != 8 or d(i, j) > 1.3:
                continue
            for kk in range(n):
                if found or z[kk] != 7 or d(kk, j) > 1.6:
                    continue
                for ll in range(n):
                    if found or z[ll] != 1 or d(kk, ll) > 1.3:
                        continue
                    for m in range(n):
                        if m in (kk, ll, j) or d(m, kk) > 1.7:
                            continue
                        if kk in used_n:
                            continue
                        used_n.add(kk)
                        out = out.index_add(0, torch.tensor([kk], device=pos.device), barrier * (
                            _sin2_dihedral(pos, i, j, kk, m) + _sin2_dihedral(pos, i, j, kk, ll)).reshape(1))
                        found = True
                        break
    return out


def yne_correction(z: Sequence[int], pos: Tensor, bonds: List[List[int]]) -> Tensor:
    """12 kcal/mol per C-C bond shorter than 1.21 Angstrom, smoothly to zero at 1.33."""
    out = pos.new_zeros(len(z))
    rmin, rmax = 1.21, 1.33
    for i in range(len(z)):
        if z[i] != 6:
            continue
        for j in bonds[i]:
            if j > i or z[j] != 6:
                continue
            r = _distance(pos, i, j)
            rv = float(r)
            if rv < rmin:
                out = out.index_add(0, torch.tensor([i], device=pos.device),
                                    torch.full((1,), 12.0, dtype=pos.dtype, device=pos.device))
            elif rv < rmax:
                x = (r - rmin) / (rmax - rmin)
                f = (1.0 - 10.0 * x ** 3 + 15.0 * x ** 4 - 6.0 * x ** 5
                     + (-5.0 + 25.0 * x) * (x ** 3 - 3.0 * x ** 4 + 3.0 * x ** 5 - x ** 6))
                out = out.index_add(0, torch.tensor([i], device=pos.device), (12.0 * f).reshape(1))
    return out


def nsp2_correction(z: Sequence[int], pos: Tensor, bonds: List[List[int]]) -> Tensor:
    """Planarity stabilization of nitrogens with three ligands, at most one of them H."""
    out = pos.new_zeros(len(z))
    for i in range(len(z)):
        if z[i] != 7 or len(bonds[i]) != 3:
            continue
        a, b, c = bonds[i]
        if sum(z[x] == 1 for x in (a, b, c)) >= 2:
            continue
        tot = 2.0 * math.pi - (_angle(pos, b, i, c) + _angle(pos, a, i, c) + _angle(pos, b, i, a))
        out = out.index_add(0, torch.tensor([i], device=pos.device),
                            (-0.5 * torch.exp(-10.0 * tot)).reshape(1))
    return out


def si_o_h_correction(z: Sequence[int], pos: Tensor, bonds: List[List[int]]) -> Tensor:
    """PM7 bending term ``15 (theta - 125 deg)^2`` of Si-O-H, damped beyond Si-O 1.7 / O-H 1.0."""
    out = pos.new_zeros(len(z))
    ref = 125.0 * math.pi / 180.0
    for o in range(len(z)):
        if z[o] != 8:
            continue
        si = h = None
        for k in bonds[o]:
            if z[k] == 14:
                si = k
            if z[k] == 1:
                h = k
        if si is None or h is None:
            continue
        r_si = (pos[o] - pos[si]).pow(2).sum() - 1.7 ** 2
        r_h = (pos[o] - pos[h]).pow(2).sum() - 1.0
        theta = _angle(pos, si, o, h)
        e = (15.0 * (theta - ref) ** 2 * torch.exp(-33.0 * torch.clamp(r_si, min=0.0))
             * torch.exp(-68.0 * torch.clamp(r_h, min=0.0)))
        out = out.index_add(0, torch.tensor([o], device=pos.device), e.reshape(1))
    return out


def _dihedral(pos: Tensor, i: int, j: int, k: int, l: int) -> Tensor:
    """Dihedral ``i-j-k-l`` in ``[0, 2 pi)`` (IUPAC sense); 0 for degenerate geometries."""
    b0, b1, b2 = pos[i] - pos[j], pos[k] - pos[j], pos[l] - pos[k]
    axis = b1 / b1.norm()
    v = b0 - (b0 * axis).sum() * axis
    w = b2 - (b2 * axis).sum() * axis
    if float(v.norm()) < 1e-6 or float(w.norm()) < 1e-6:
        return pos.new_zeros(())
    x = (v * w).sum()
    y = (torch.cross(axis, v, dim=0) * w).sum()
    phi = torch.atan2(y, x)
    if abs(float(phi)) < 4e-5:
        return pos.new_zeros(())
    return torch.where(phi < 0, phi + 2.0 * math.pi, phi)


def _wrap(a: Tensor) -> Tensor:
    """Angle into ``(-pi, pi]``."""
    if float(a) <= -math.pi:
        return a + 2.0 * math.pi
    if float(a) > math.pi:
        return a - 2.0 * math.pi
    return a


# Jurecka-type dispersion of PM7: C6 (J nm^6 / mol), van der Waals radii (pm) and
# Slater-Kirkwood effective electron numbers
_DISP_C6 = {1: 0.16, 2: 0.084, 5: 5.79, 6: 1.65, 7: 1.11, 8: 0.70, 9: 0.57, 10: 0.45, 15: 3.25,
            16: 5.79, 17: 5.97, 18: 3.71, 27: 0.04, 35: 11.60, 36: 4.47, 53: 25.80, 54: 16.50}
_DISP_R = {1: 156.0, 2: 140.0, 5: 180.0, 6: 170.0, 7: 155.0, 8: 152.0, 9: 147.0, 10: 154.0,
           15: 180.0, 16: 180.0, 17: 175.0, 18: 188.0, 27: 140.0, 35: 185.0, 36: 202.0,
           53: 198.0, 54: 216.0}
_DISP_N = {1: 0.80, 2: 1.42, 5: 2.16, 6: 2.50, 7: 2.82, 8: 3.15, 9: 3.48, 10: 3.81, 15: 4.50,
           16: 4.80, 17: 5.10, 18: 5.40, 27: 2.90, 35: 6.00, 36: 6.30, 53: 6.95, 54: 7.25}


def pm7_dispersion(z: Sequence[int], pos: Tensor, bonds: List[List[int]], alpha: float = 15.450118,
                   s: float = 1.226593, scale: float = 2.286419) -> Tensor:
    """PM7 dispersion (Jurecka-type damped C6/R^6, Stewart 2013), per atom (kcal/mol).

    ``E = -scale sum_ij C6_ij / R^6 / (1 + exp(-alpha (R / (s R0) - 1)))`` over
    the pairs of elements with data; carbon's C6 is 0.95 with four bonds and
    1.65 otherwise; Slater-Kirkwood combination of the C6 values.
    """
    out = pos.new_zeros(len(z))
    have = [i for i, v in enumerate(z) if v in _DISP_C6]
    if len(have) < 2:
        return out
    idx = torch.as_tensor(have, device=pos.device)
    c6 = [(0.95 if len(bonds[i]) == 4 else 1.65) if z[i] == 6 else _DISP_C6[z[i]] for i in have]
    c6 = torch.as_tensor(c6, dtype=pos.dtype, device=pos.device)
    nn_ = torch.as_tensor([_DISP_N[z[i]] for i in have], dtype=pos.dtype, device=pos.device)
    rr = torch.as_tensor([_DISP_R[z[i]] for i in have], dtype=pos.dtype, device=pos.device)
    a, b = torch.triu_indices(len(have), len(have), offset=1, device=pos.device)
    c6ab = (2.0 * (c6[a] ** 2 * c6[b] ** 2 * nn_[a] * nn_[b]) ** (1.0 / 3.0)
            / ((c6[a] * nn_[b] ** 2) ** (1.0 / 3.0) + (c6[b] * nn_[a] ** 2) ** (1.0 / 3.0)))
    r0 = (rr[a] ** 3 + rr[b] ** 3) / (rr[a] ** 2 + rr[b] ** 2) / 1000.0 * 2.0
    rij = (pos[idx[a]] - pos[idx[b]]).norm(dim=-1)
    keep = rij < 100.0
    rnm = rij * 0.1
    damp = 1.0 / (1.0 + torch.exp(-alpha * (rnm / (s * r0) - 1.0)))
    e = -scale * c6ab / rnm ** 6 * damp / (1000.0 * 4.184)
    e = torch.where(keep, e, torch.zeros_like(e))
    return out.index_add(0, idx[a], 0.5 * e).index_add(0, idx[b], 0.5 * e)


# covalent radii (Angstrom) of the hydrogen-bond topology, scaled by 4/3 in use
_HB_RADII = (0.0,
             0.32, 0.46, 1.20, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67, 1.40, 1.25, 1.13, 1.04,
             1.10, 1.02, 0.99, 0.96, 1.76, 1.54, 1.33, 1.22, 1.21, 1.10, 1.07, 1.04, 1.00, 0.99,
             1.01, 1.09, 1.12, 1.09, 1.15, 1.10, 1.14, 1.17, 1.89, 1.67, 1.47, 1.39, 1.32, 1.24,
             1.15, 1.13, 1.13, 1.08, 1.15, 1.23, 1.28, 1.26, 1.26, 1.23, 1.32, 1.31, 2.09, 1.76,
             1.62, 1.47, 1.58, 1.57, 1.56, 1.55, 1.51, 1.52, 1.51, 1.50, 1.49, 1.49, 1.48, 1.53,
             1.46, 1.37, 1.31, 1.23, 1.18, 1.16, 1.11, 1.12, 1.13, 1.32, 1.30, 1.30, 1.36, 1.31,
             1.38, 1.42)


def _hbond_triples(z: Sequence[int], xyz, r_xh: float, cutoff: float, heavy=(7, 8)):
    """Candidate hydrogen bonds ``(donor, H, acceptor)`` found by distance and angle."""
    n = len(z)

    def d(a, b):
        return math.sqrt(sum((xyz[a][c] - xyz[b][c]) ** 2 for c in range(3)))

    acc = [i for i in range(n) if z[i] in heavy]
    used = [False] * n
    hyd = []
    for i in acc:
        for j in range(n):
            if z[j] == 1 and not used[j] and d(i, j) < r_xh:
                hyd.append(j)
                used[j] = True
    triples = []
    for i in acc:
        for j in hyd:
            if d(i, j) >= r_xh:
                continue
            for k in acc:
                if k == i or d(k, i) >= cutoff:
                    continue
                # angle k-j-i (at the hydrogen)
                v1 = [xyz[k][c] - xyz[j][c] for c in range(3)]
                v2 = [xyz[i][c] - xyz[j][c] for c in range(3)]
                cosang = sum(a * b for a, b in zip(v1, v2)) / (d(k, j) * d(i, j))
                if math.acos(max(-1.0, min(1.0, cosang))) <= 0.5 * math.pi:
                    continue
                if (k, j, i) in triples or (i, j, k) in triples:
                    continue
                triples.append((i, j, k))
    return triples


def _hb_neighbors(z, xyz, center: int, h: int, radius):
    """Up to four atoms bonded to ``center`` (dropping the longest of a fifth)."""
    n = len(z)

    def d(a, b):
        return math.sqrt(sum((xyz[a][c] - xyz[b][c]) ** 2 for c in range(3)))

    out = []
    for j in range(n):
        if j != center and d(j, center) < radius[z[j]] + radius[z[center]]:
            out.append(j)
            if len(out) == 5:
                far, best = 0.0, 0
                for k in range(5):
                    dk = d(center, out[k])
                    if far < dk:
                        far, best = dk, k
                del out[best]
    return out


def _hb_frame(z, xyz, center: int, h: int, bonded: List[int], radius):
    """The three reference atoms of one side of a hydrogen bond (MOPAC's DH+ choice)."""
    def d(a, b):
        return math.sqrt(sum((xyz[a][c] - xyz[b][c]) ** 2 for c in range(3)))

    def farthest(cands, exclude=()):
        best, far = None, -1.0
        for k in cands:
            dk = d(k, h)
            if dk > far and k not in exclude:
                far, best = dk, k
        return best

    close = d(center, h) < radius[z[center]] + radius[z[h]]
    nb = len(bonded)
    if nb in (3, 4):
        a = farthest(bonded)
        b = farthest(bonded, (a,))
        c = farthest(bonded, (a, b))
        return a, b, c
    if nb == 2:
        a = farthest(bonded)
        b = [k for k in bonded if k != a][-1] if any(k != a for k in bonded) else None
        return a, b, (h if close else center)
    if nb == 1:
        a = bonded[0]
        second = [k for k in range(len(z)) if k != a and d(k, a) < radius[z[k]] + radius[z[a]]]
        b = farthest(second)
        return a, b, (h if close else center)
    if nb == 0:
        x = h if close else center
        return x, x, x
    return None


def pm7_hbond(z: Sequence[int], pos: Tensor) -> Tensor:
    """PM7 hydrogen-bond correction, per hydrogen atom (kcal/mol).

    The third-generation correction of Korth (J. Chem. Theory Comput. 6, 3808,
    2010) with the PM7 scales and damping (Stewart 2013): for every
    X-H...Y (X, Y = N, O) found by distance (X-H < 1.4, X...Y < 7 Angstrom)
    and angle,

        E = s / R_XY^2 cos^2(theta) [1 - (1 - c_X t_X c_Y t_Y)^2] f_damp

    with ``theta`` the deviation of X-H...Y from linearity and ``c``, ``t``
    the cosines of the deviations of the lone-pair angle and torsion on each
    side from their hybridization targets, plus an extra short-range O...O
    term.
    """
    out = pos.new_zeros(len(z))
    xyz = pos.detach().tolist()
    radius = [4.0 / 3.0 * r for r in _HB_RADII]
    triples = _hbond_triples(z, xyz, 1.4, 7.0)
    if not triples:
        return out
    hartree_kcal = 27.211386245988 * 23.060547830619029
    a0sq = 0.529177210903 ** 2
    scale = {7: -0.171271 * a0sq, 8: -0.098822 * a0sq}
    deg = math.pi / 180.0

    def d(a, b):
        return (pos[a] - pos[b]).norm()

    def dv(a, b):
        return math.sqrt(sum((xyz[a][c] - xyz[b][c]) ** 2 for c in range(3)))

    for (A, H, B) in triples:
        na = _hb_neighbors(z, xyz, A, H, radius)
        nb = _hb_neighbors(z, xyz, B, H, radius)
        if B in na or any(x == y and y != H for x in na for y in nb):
            continue
        fa = _hb_frame(z, xyz, A, H, na, radius)
        fb = _hb_frame(z, xyz, B, H, nb, radius)
        if fa is None or fb is None or None in fa or None in fb:
            continue
        cos_hb = -torch.cos(_angle(pos, A, H, B))
        if float(cos_hb) <= 0.0:
            continue
        sides = []
        ok = True
        for center, (r1, r2, r3), nbonds, last in ((A, fa, len(na), False), (B, fb, len(nb), True)):
            tcs = tcs2 = False
            shift = shift2 = tshift = 0.0
            if z[center] == 8:
                if nbonds == 1:
                    shift, shift2, tshift, tcs2 = math.pi, 120.0 * deg, 0.0, True
                else:
                    shift = shift2 = 109.48 * deg
                    tshift = 54.74 * deg
            elif z[center] == 7:
                if nbonds == 2:
                    shift = shift2 = 120.0 * deg
                else:
                    shift = shift2 = 109.48 * deg
                    tshift = 54.74 * deg
                    tcs = True
            check = 0.0
            if tcs:
                tc = _wrap(_dihedral(pos, r2, r1, center, r3))
                tc = -math.pi - tc if float(tc) < 0 else math.pi - tc
                signed = float(tc)
                tcd = tc.abs() / deg
                tshift = tshift + ((54.74 - tcd) / 54.74 * 35.26) * deg
                shift = shift - ((54.74 - tcd) / 54.74 * 19.48) * deg
                shift2 = shift
                check = signed
            ang2 = _angle(pos, r1, center, H)
            c2 = torch.maximum(torch.cos(shift - ang2), torch.cos(shift2 - ang2))
            if float(c2) <= 0.0:
                ok = False
                break
            tcorr = _wrap(_dihedral(pos, r2, r1, center, H))
            if (not tcs2) or abs(float(tcorr)) / deg > 90.0:
                tcorr = -math.pi - tcorr if float(tcorr) < 0 else math.pi - tcorr
            if check < 0:
                tcos = torch.cos(_wrap(tshift - tcorr))
            elif check > 0:
                tcos = torch.cos(_wrap(-tshift - tcorr))
            else:
                tcos = torch.maximum(torch.cos(_wrap(tshift - tcorr)), torch.cos(_wrap(-tshift - tcorr)))
            if dv(H, center) > dv(H, r1) and tcs2:
                tcos = torch.zeros_like(tcos)
            if r2 == r3 or H == r3:
                tcos = torch.ones_like(tcos)
            if last:
                tcos = tcos.abs()
            elif float(tcos) < 0.0:
                ok = False
                break
            sides.append((c2, tcos))
        if not ok:
            continue
        s_c = 0.5 * (scale[z[A]] + scale[z[B]])
        ha, hb = d(H, A), d(H, B)
        xc = torch.minimum(ha, hb)
        if float(torch.maximum(ha, hb) - xc) > 0.5:
            damping = 1.0 - 1.0 / (1.0 + torch.exp(-60.0 * (xc / 1.2 - 1.0)))
        else:
            damping = torch.ones_like(xc)
        rab = d(A, B)
        damping = damping / (1.0 + torch.exp(-100.0 * (rab / 2.4 - 1.0)))
        damping = damping * (1.0 - 1.0 / (1.0 + torch.exp(-10.0 * (rab / 7.0 - 1.0))))
        prod = sides[0][0] * sides[0][1] * sides[1][0] * sides[1][1]
        e = s_c / rab ** 2 * cos_hb ** 2 * (1.0 - (1.0 - prod) ** 2) * hartree_kcal * damping
        if z[A] == 8 and z[B] == 8:
            e = e - 2.5 * torch.exp(-80.0 * torch.clamp(rab - 2.67, min=0.0) ** 2) * cos_hb ** 4
        out = out.index_add(0, torch.tensor([H], device=pos.device), e.reshape(1))
    return out


def _smoothstep7(x: Tensor) -> Tensor:
    return -20.0 * x ** 7 + 70.0 * x ** 6 - 84.0 * x ** 5 + 35.0 * x ** 4


def _valence(z: Sequence[int], pos: Tensor, a: int, b: int) -> Tensor:
    """Smooth bond count of atoms ``a`` and ``b`` (1 within the sum of the
    covalent radii, 0 beyond 1.6 times it)."""
    r0 = (el.COVALENT_RADIUS[z[a]] or 0.0) + (el.COVALENT_RADIUS[z[b]] or 0.0)
    r1 = 1.6 * r0
    if a == b:
        return pos.new_zeros(())
    r = (pos[a] - pos[b]).norm()
    rv = float(r)
    if rv == 0.0 or rv >= r1:
        return pos.new_zeros(())
    if rv <= r0:
        return pos.new_ones(())
    return 1.0 - _smoothstep7((r - r0) / (r1 - r0))


def h4_hbond(z: Sequence[int], pos: Tensor) -> Tensor:
    """Hydrogen-bond correction H4 (Rezac & Hobza, J. Chem. Theory Comput. 8,
    141, 2012) as used by PM6-ORG, per hydrogen atom (kcal/mol).

    For every X-H...Y (X, Y = N, O; X-H < 1.15, X...Y < 5.5 Angstrom) the
    energy ``c_XY E_rad(R_XY) E_ang(theta) f_switch f_water f_NR4+ f_COO-``
    with a seventh-order radial polynomial, the angular switch of the
    deviation from linearity, and the scalings of water, ammonium and
    carboxylate groups.
    """
    out = pos.new_zeros(len(z))
    xyz = pos.detach().tolist()
    triples = _hbond_triples(z, xyz, 1.15, 5.5, heavy=(7, 8, 16))
    para = {(8, 8): 2.32, (8, 7): 3.10, (7, 8): 1.07, (7, 7): 2.01}
    n = len(z)
    for (i0, h, k0) in triples:
        a_, b_ = k0, i0           # the order of the reference implementation's arguments
        rda = (pos[a_] - pos[b_]).norm()
        rih, rjh = (pos[a_] - pos[h]).norm(), (pos[b_] - pos[h]).norm()
        ang = math.pi - _angle(pos, a_, h, b_)
        if float(rih) < float(rjh):
            d, acc, rdh, rah = a_, b_, rih, rjh
        else:
            d, acc, rdh, rah = b_, a_, rjh, rih
        e_rad = (-0.00303407407407313510 * rda ** 7 + 0.07357629629627092382 * rda ** 6
                 - 0.70087111111082800452 * rda ** 5 + 3.25309629629461749545 * rda ** 4
                 - 7.20687407406838786983 * rda ** 3 + 5.31754666665572184314 * rda ** 2
                 + 3.40736000001102778967 * rda - 4.68512000000450434811)
        e_ang = 1.0 - _smoothstep7(ang / (math.pi / 2.0)) ** 2
        c = para.get((z[d], z[acc]), 0.0)
        if float(rdh) > 1.15:
            x = (rdh - 1.15) / (0.5 * rdh + 0.5 * rah - 1.15)
            e_sw = 1.0 - _smoothstep7(x)
        else:
            e_sw = torch.ones_like(rda)
        e_w = torch.ones_like(rda)
        if z[d] == 8 and z[acc] == 8:
            hyd = sum((_valence(z, pos, d, m) for m in range(n) if z[m] == 1), pos.new_zeros(()))
            oth = sum((_valence(z, pos, d, m) for m in range(n) if z[m] != 1), pos.new_zeros(()))
            if float(hyd) >= 1.0:
                v = float(hyd)
                fv = hyd - 1.0 if 1.0 < v <= 2.0 else (3.0 - hyd if 2.0 < v < 3.0 else torch.zeros_like(hyd))
                e_w = 1.0 + (0.42 - 1.0) * fv * torch.clamp(1.0 - oth, min=0.0)
        e_chd = torch.ones_like(rda)
        if z[d] == 7:
            v = sum((_valence(z, pos, d, m) for m in range(n)), pos.new_zeros(()))
            e_chd = 1.0 + (3.61 - 1.0) * torch.clamp(v - 3.0, min=0.0) if float(v) > 3.0 else e_chd
        e_cha = torch.ones_like(rda)
        if z[acc] == 8:
            o1 = acc
            cv_o1 = pos.new_zeros(())
            cc, cdist = None, 9.9e9
            for m in range(n):
                v = _valence(z, pos, o1, m)
                cv_o1 = cv_o1 + v
                dm = float((pos[o1] - pos[m]).norm())
                if float(v) > 0.0 and z[m] == 6 and dm < cdist:
                    cdist, cc = dm, m
            if cc is not None:
                o2, odist = None, 9.9e9
                cv_cc = pos.new_zeros(())
                for m in range(n):
                    v = _valence(z, pos, cc, m)
                    cv_cc = cv_cc + v
                    dm = float((pos[cc] - pos[m]).norm())
                    if float(v) > 0.0 and m != o1 and z[m] == 8 and dm < odist:
                        odist, o2 = dm, m
                if o2 is not None:
                    cv_o2 = sum((_valence(z, pos, o2, m) for m in range(n)), pos.new_zeros(()))
                    f1 = torch.clamp(1.0 - (1.0 - cv_o1).abs(), min=0.0)
                    f2 = torch.clamp(1.0 - (1.0 - cv_o2).abs(), min=0.0)
                    f3 = torch.clamp(1.0 - (3.0 - cv_cc).abs(), min=0.0)
                    e_cha = 1.0 + (1.41 - 1.0) * f1 * f2 * f3
        e = c * e_rad * e_ang * e_sw * e_w * e_chd * e_cha
        out = out.index_add(0, torch.tensor([h], device=pos.device), e.reshape(1))
    return out


def hh_repulsion(z: Sequence[int], pos: Tensor) -> Tensor:
    """Hydrogen-hydrogen repulsion of D3H4 (Rezac & Hobza 2012), per atom (kcal/mol)."""
    out = pos.new_zeros(len(z))
    hs = [i for i, v in enumerate(z) if v == 1]
    if len(hs) < 2:
        return out
    idx = torch.as_tensor(hs, device=pos.device)
    a, b = torch.triu_indices(len(hs), len(hs), offset=1, device=pos.device)
    r = (pos[idx[a]] - pos[idx[b]]).norm(dim=-1)
    rc = r.clamp(min=1e-6)
    mid = (-2714.952351603469651 * rc ** 5 + 17103.650110591705015 * rc ** 4
           - 42511.857982217959943 * rc ** 3 + 52063.196799138342612 * rc ** 2
           - 31430.658335972289933 * rc + 7516.084696095140316)
    far = 118.7326 * torch.exp(-1.53965 * rc ** 1.72905)
    e = torch.where(r <= 1.0, torch.full_like(r, 25.46293603147693), torch.where(r < 1.5, mid, far))
    return out.index_add(0, idx[a], 0.5 * e).index_add(0, idx[b], 0.5 * e)
