"""NDDO parameter sets in the MOPAC ``EXTERNAL`` file format.

The format is the one MOPAC reads with its ``EXTERNAL=`` keyword: one
parameter per line, ``NAME  Element  value``; the diatomic core-core
parameters of a pair as ``ALPB_X  Y  value`` and ``XFAC_X  Y  value``; the
method's global parameters as ``PARn  value``. Lines starting with ``*`` are
comments, ``END`` (or a blank line) ends the list. A file written by
:func:`write_parameters` is read back exactly and can be passed to MOPAC
itself, so parameters refit with xnn run unchanged in MOPAC.

Per-element parameter names (eV, Angstrom, bohr):

=========================  =====================================================
``USS UPP UDD``            one-center one-electron energies
``ZS ZP ZD``               Slater exponents (overlaps, multipole moments)
``BETAS BETAP BETAD``      resonance parameters
``GSS GSP GPP GP2 HSP``    one-center two-electron integrals (main group)
``F0SD G2SD``              one-center ``sd`` integrals overriding the computed ones
``ZSN ZPN ZDN``            internal exponents of the one-center integrals
``POC``                    additive term of the core charge distribution
``ALP``                    core-core exponent (PM3)
``FNij``                   core-core Gaussian ``j``: ``i`` = 1 amplitude, 2 width,
                           3 center
=========================  =====================================================

The published sets are listed by :func:`list_parameter_sets` and selected by
name (``"pm3"``, ``"pm6"``, ``"pm6-org"``, ``"pm7"``); their values are read
from the MOPAC program on first use (:mod:`xnn.sqmnn.common.published`). Any
other set is read from a path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Union

from .elements import CHEMICAL_SYMBOLS, atomic_number

#: Per-element parameter names in the order the writer uses.
ELEMENT_PARAMETERS = ("USS", "UPP", "UDD", "ZS", "ZP", "ZD", "BETAS", "BETAP", "BETAD",
                      "GSS", "GSP", "GPP", "GP2", "HSP", "F0SD", "G2SD", "POC", "ALP",
                      "ZSN", "ZPN", "ZDN")
#: The core-core Gaussians: ``FN1k`` amplitude, ``FN2k`` exponent, ``FN3k`` center.
GAUSSIAN_PARAMETERS = tuple(f"FN{i}{k}" for k in range(1, 5) for i in range(1, 4))
#: Diatomic core-core parameters.
PAIR_PARAMETERS = ("ALPB", "XFAC")

_ALIASES = {"pm3": "pm3", "pm6": "pm6", "pm7": "pm7", "pm6-org": "pm6-org",
            "pm6_org": "pm6-org", "pm6org": "pm6-org"}


@dataclass
class ParameterSet:
    """The parameters of one NDDO method.

    Attributes
    ----------
    name : str
        A label (the published set's name or the file stem).
    elements : dict
        ``{Z: {NAME: value}}``; a parameter that is absent is zero.
    pairs : dict
        ``{(Z1, Z2): {"ALPB": value, "XFAC": value}}`` with ``Z1 >= Z2``.
    globals : dict
        ``{n: value}``, the method's ``PARn`` parameters.
    """

    name: str = ""
    elements: Dict[int, Dict[str, float]] = field(default_factory=dict)
    pairs: Dict[Tuple[int, int], Dict[str, float]] = field(default_factory=dict)
    globals: Dict[int, float] = field(default_factory=dict)

    def get(self, z: int, name: str) -> float:
        """Value of a per-element parameter (0 when the set does not define it)."""
        return float(self.elements.get(int(z), {}).get(name, 0.0))

    def pair(self, z1: int, z2: int, name: str) -> float:
        """Value of a diatomic parameter (0 when the pair has none)."""
        key = (max(int(z1), int(z2)), min(int(z1), int(z2)))
        return float(self.pairs.get(key, {}).get(name, 0.0))

    def par(self, n: int) -> float:
        """Value of the global parameter ``PARn`` (0 when undefined)."""
        return float(self.globals.get(int(n), 0.0))

    @property
    def atomic_numbers(self) -> tuple:
        """The elements with orbitals, in increasing order."""
        return tuple(sorted(z for z, p in self.elements.items()
                            if p.get("ZS", 0.0) > 0.0 or p.get("GSS", 0.0) > 0.0))


def list_parameter_sets() -> tuple:
    """Names of the published parameter sets."""
    from .published import PUBLISHED
    return tuple(sorted(PUBLISHED))


def read_parameters(spec: Union[str, Path]) -> ParameterSet:
    """Read a parameter set: a published method's name or a MOPAC ``EXTERNAL`` file.

    Parameters
    ----------
    spec : str or Path
        ``"pm3"``, ``"pm6"``, ``"pm6-org"``, ``"pm7"`` or a path.

    Returns
    -------
    ParameterSet

    Raises
    ------
    FileNotFoundError
        For a name that is neither a published set nor an existing file.
    ValueError
        For a line that is not a known parameter.
    """
    key = str(spec).strip().lower()
    if key in _ALIASES:
        from .published import published_parameters
        return published_parameters(_ALIASES[key])
    path = Path(spec).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"no parameter set {spec!r}; published sets: {list_parameter_sets()}")
    out = ParameterSet(name=path.stem)
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if line.startswith("*"):
            continue
        if not line or line.upper().startswith("END"):
            break
        tokens = line.split()
        key = tokens[0].upper()
        try:
            m = re.fullmatch(r"PAR(\d+)", key)
            if m:
                out.globals[int(m.group(1))] = float(tokens[1])
                continue
            m = re.fullmatch(r"(ALPB|XFAC)_([A-Z]{1,2})", key)
            if m:
                z2 = atomic_number(m.group(2))
                z1 = atomic_number(tokens[1])
                pk = (max(z1, z2), min(z1, z2))
                out.pairs.setdefault(pk, {})[m.group(1)] = float(tokens[2])
                continue
            if key in ELEMENT_PARAMETERS or key in GAUSSIAN_PARAMETERS:
                out.elements.setdefault(atomic_number(tokens[1]), {})[key] = float(tokens[2])
                continue
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(f"{path}:{lineno}: cannot read {raw!r}") from exc
        raise ValueError(f"{path}:{lineno}: unknown parameter {tokens[0]!r}")
    return out


def write_parameters(params: ParameterSet, path: Union[str, Path], header: str = "") -> Path:
    """Write a parameter set as a MOPAC ``EXTERNAL`` file.

    Parameters
    ----------
    params : ParameterSet
        The set (e.g. from :meth:`~xnn.sqmnn.models.nddo.NDDO.parameter_set`).
    path : str or Path
        Output file.
    header : str, optional
        Comment lines written first (each prefixed with ``*``).

    Returns
    -------
    Path
        The written file.
    """
    lines = [f"* {h}" for h in header.splitlines()] if header else []
    for z in sorted(params.elements):
        values = params.elements[z]
        sym = CHEMICAL_SYMBOLS[z]
        for name in ELEMENT_PARAMETERS + GAUSSIAN_PARAMETERS:
            v = values.get(name, 0.0)
            if v != 0.0:
                lines.append(f"{name:<8s}{sym:<4s}{v!r:>24s}")
    for (z1, z2) in sorted(params.pairs, key=lambda p: (p[0], p[1])):
        for name in PAIR_PARAMETERS:
            v = params.pairs[(z1, z2)].get(name, 0.0)
            if v != 0.0:
                lines.append(f"{name + '_' + CHEMICAL_SYMBOLS[z2]:<8s}{CHEMICAL_SYMBOLS[z1]:<4s}"
                             f"{v!r:>24s}")
    for n in sorted(params.globals):
        if params.globals[n] != 0.0:
            lines.append(f"PAR{n:<5d}{'':4s}{params.globals[n]!r:>24s}")
    lines.append("END")
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path
