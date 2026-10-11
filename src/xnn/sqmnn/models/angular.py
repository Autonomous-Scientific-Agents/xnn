"""Angular algebra of the ``spd`` Slater basis: real harmonics, Gaunt coefficients.

Orbital orders
--------------
* molecular frame: ``s, px, py, pz, dx2-y2, dxz, dz2, dyz, dxy`` (MOPAC's order);
* diatomic (local) frame, ``z`` along the bond: ``s, pz, px, py, dz2, dxz,
  dyz, dx2-y2, dxy`` (sigma, pi, pi, delta), the frame of the two-center
  integrals.

Everything here is a constant computed once with numpy: Gaunt coefficients
``int Y_i Y_j Y_LM dOmega`` by exact product quadrature, the multipole
coefficients of the orbital products used by the two-center integrals, and
the angular factors of the one-center Slater-Condon integrals.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

#: Labels of the molecular-frame orbitals.
MOLECULAR = ("s", "x", "y", "z", "x2y2", "xz", "z2", "yz", "xy")
#: Labels of the diatomic-frame orbitals.
LOCAL = ("s", "z", "x", "y", "z2", "xz", "yz", "x2y2", "xy")
#: Angular momentum of each label.
L_OF = {"s": 0, "x": 1, "y": 1, "z": 1, "z2": 2, "xz": 2, "yz": 2, "x2y2": 2, "xy": 2}
#: Magnetic quantum number of each label (cosine type > 0, sine type < 0).
M_OF = {"s": 0, "z": 0, "x": 1, "y": -1, "z2": 0, "xz": 1, "yz": -1, "x2y2": 2, "xy": -2}
#: Labels of the multipole components of order L.
MULTIPOLES = {0: ("s",), 1: ("z", "x", "y"), 2: ("z2", "xz", "yz", "x2y2", "xy")}

_PI = math.pi


def harmonic(label: str, v: np.ndarray) -> np.ndarray:
    """Real normalized spherical harmonic ``label`` at unit vectors ``v`` ``(..., 3)``."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    c1 = math.sqrt(3.0 / (4.0 * _PI))
    c20 = math.sqrt(5.0 / (16.0 * _PI))
    c21 = math.sqrt(15.0 / (4.0 * _PI))
    c22 = math.sqrt(15.0 / (16.0 * _PI))
    table = {
        "s": lambda: np.full_like(x, 1.0 / math.sqrt(4.0 * _PI)),
        "x": lambda: c1 * x, "y": lambda: c1 * y, "z": lambda: c1 * z,
        "z2": lambda: c20 * (3.0 * z * z - 1.0),
        "xz": lambda: c21 * x * z, "yz": lambda: c21 * y * z,
        "x2y2": lambda: c22 * (x * x - y * y), "xy": lambda: c21 * x * y,
    }
    return table[label]()


def _legendre_real(l: int, m: int, v: np.ndarray) -> np.ndarray:
    """Real spherical harmonic of any order (for the one-center angular factors)."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    am = abs(m)
    # associated Legendre P_l^|m|(z) by the standard recursion, without the
    # Condon-Shortley phase (the factors below use squares of these functions)
    s = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    pmm = np.ones_like(z)
    for i in range(1, am + 1):
        pmm = pmm * (2 * i - 1) * s
    if l == am:
        plm = pmm
    else:
        pm1 = z * (2 * am + 1) * pmm
        if l == am + 1:
            plm = pm1
        else:
            for ll in range(am + 2, l + 1):
                plm = (z * (2 * ll - 1) * pm1 - (ll + am - 1) * pmm) / (ll - am)
                pmm, pm1 = pm1, plm
            plm = pm1
    norm = math.sqrt((2 * l + 1) / (4 * _PI) * math.factorial(l - am) / math.factorial(l + am))
    phi = np.arctan2(y, x)
    if m > 0:
        return math.sqrt(2.0) * norm * plm * np.cos(m * phi)
    if m < 0:
        return math.sqrt(2.0) * norm * plm * np.sin(am * phi)
    return norm * plm


@lru_cache(maxsize=None)
def _quadrature(n_theta: int = 12, n_phi: int = 24):
    u, w = np.polynomial.legendre.leggauss(n_theta)
    phi = 2.0 * _PI * np.arange(n_phi) / n_phi
    st = np.sqrt(1.0 - u * u)
    v = np.stack([np.outer(st, np.cos(phi)), np.outer(st, np.sin(phi)),
                  np.outer(u, np.ones_like(phi))], axis=-1).reshape(-1, 3)
    weights = np.outer(w, np.full(n_phi, 2.0 * _PI / n_phi)).reshape(-1)
    return v, weights


def gaunt(a: str, b: str, c: str) -> float:
    """``int Y_a Y_b Y_c dOmega`` of three real harmonics (labels up to ``l = 2``)."""
    v, w = _quadrature()
    return float(np.sum(w * harmonic(a, v) * harmonic(b, v) * harmonic(c, v)))


def pair_type(la: int, lb: int) -> str:
    """Name of the charge-distribution type of a pair of shells: ss, sp, pp, sd, pd, dd."""
    names = "spd"
    lo, hi = sorted((la, lb))
    return names[lo] + names[hi]


# reference orbital pair whose multipole coefficient is 1, per (pair type, L)
_REFERENCE = {("ss", 0): ("s", "s", "s"), ("sp", 1): ("s", "z", "z"),
              ("pp", 0): ("z", "z", "s"), ("pp", 2): ("x", "z", "xz"),
              ("sd", 2): ("xz", "s", "xz"), ("pd", 1): ("xz", "x", "z"),
              ("dd", 0): ("z2", "z2", "s"), ("dd", 2): ("xz", "xz", "x2y2")}
#: Multipole orders carried by each charge-distribution type (truncated at L = 2).
ORDERS = {"ss": (0,), "sp": (1,), "pp": (0, 2), "sd": (2,), "pd": (1,), "dd": (0, 2)}


@lru_cache(maxsize=None)
def multipole_coefficient(a: str, b: str, comp: str) -> float:
    """Weight of the point-charge multipole ``comp`` in the orbital product ``a b``.

    The product ``Y_a Y_b`` of two real orbitals is expanded in multipoles of
    order ``L <= 2`` represented by point-charge configurations whose
    separations are fixed per charge-distribution type; the weight is the
    Gaunt coefficient of the component relative to the reference product of
    the same type and order, with the factor ``2 / sqrt(3)`` of the
    ``L = 2, M = 0`` configuration (Thiel & Voityuk, Theor. Chim. Acta 81,
    391, 1992).
    """
    L = L_OF[comp]
    ptype = pair_type(L_OF[a], L_OF[b])
    if L not in ORDERS[ptype]:
        return 0.0
    value = gaunt(a, b, comp)
    if abs(value) < 1e-14:
        return 0.0
    ra, rb, rc = _REFERENCE[(ptype, L)]
    scale = 2.0 / math.sqrt(3.0) if (L == 2 and comp == "z2") else 1.0
    return value * scale / gaunt(ra, rb, rc)


@lru_cache(maxsize=None)
def slater_condon_factors() -> np.ndarray:
    """Angular factors ``A^K_{ijkl}`` of the one-center integrals, molecular order.

    ``(ij|kl) = sum_K R^K A^K_{ijkl}`` with ``A^K = 4 pi / (2K + 1) sum_Q
    <i|Y_KQ|j> <k|Y_KQ|l>``; shape ``(5, 9, 9, 9, 9)`` for ``K = 0..4``.
    """
    v, w = _quadrature(14, 28)
    orb = np.stack([harmonic(lab, v) for lab in MOLECULAR])          # (9, P)
    out = np.zeros((5, 9, 9, 9, 9))
    for K in range(5):
        g = np.stack([np.einsum("p,ip,jp,p->ij", _legendre_real(K, Q, v), orb, orb, w)
                      for Q in range(-K, K + 1)])                   # (2K+1, 9, 9)
        out[K] = 4.0 * _PI / (2 * K + 1) * np.einsum("qij,qkl->ijkl", g, g)
    out[np.abs(out) < 1e-13] = 0.0
    return out


@lru_cache(maxsize=None)
def dipole_factors() -> np.ndarray:
    """``<i| r_hat_j |k>`` angular weights of the one-center hybrid dipoles.

    Shape ``(3, 9, 9)`` in molecular order: the multipole coefficients of the
    ``sp`` and ``pd`` products on the ``x``, ``y``, ``z`` dipole components.
    """
    out = np.zeros((3, 9, 9))
    for c, comp in enumerate(("x", "y", "z")):
        for i, a in enumerate(MOLECULAR):
            for k, b in enumerate(MOLECULAR):
                if pair_type(L_OF[a], L_OF[b]) in ("sp", "pd"):
                    out[c, i, k] = multipole_coefficient(a, b, comp)
    return out
