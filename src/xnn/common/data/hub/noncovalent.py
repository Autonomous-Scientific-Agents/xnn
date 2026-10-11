"""Builders for the S22 and S66 benchmark sets of noncovalent interactions.

* ``s22``: 22 dimers (hydrogen-bonded, dispersion-bound, mixed) of Jurecka,
  Sponer, Cerny & Hobza, *Phys. Chem. Chem. Phys.* 8, 1985 (2006);
* ``s66``: 66 dimers of Rezac, Riley & Hobza, *J. Chem. Theory Comput.* 7,
  2427 (2011).

Each entry is one dimer at its reference geometry with ``fragment`` (0 for
the first monomer, 1 for the second, per atom), ``label`` and the
CCSD(T)/CBS ``interaction_energy`` (kcal/mol). ``reference="original"``
(default) gives the values of the original publications, ``"revised"`` the
revised S22 values of Takatani *et al.*, *J. Chem. Phys.* 132, 144104 (2010)
(S66 has one set). The monomers of an interaction energy are the fragments of
the dimer geometry.

Notes
-----
* Source: the database files of the Psi4 program (github.com/psi4/psi4,
  ``psi4/share/psi4/databases``, commit 66507f4), which tabulate the
  published geometries and reference energies; read as text, nothing is
  executed.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

from ...data.elements import atomic_number
from ._download import download_file
from .base import DatasetBuilder, register_dataset

_URL = ("https://raw.githubusercontent.com/psi4/psi4/66507f43f5c324bcf595be35f6d4f49773804103/"
        "psi4/share/psi4/databases/{name}.py")
_FILES = {"s22": ("S22.py", "718f6f7bc46e885eb83cb9585208d976"),
          "s66": ("S66.py", "2f0d0cd1994570804fc78e865dae1543")}
_REFERENCE = {"s22": {"original": "BIND_S220", "revised": "BIND_S22A"},
              "s66": {"original": "BIND", "revised": "BIND"}}


def _parse(text: str, table: str) -> list[dict]:
    """Dimers, labels and reference energies of a database file."""
    labels = dict((int(m.group(1)), m.group(2).strip()) for m in re.finditer(
        r"TAGL\['%s-%s'\s*%\s*\(dbse,\s*'?(\d+)'?\s*\)\]\s*=\s*(?:\"\"\"|')(.*?)(?:\"\"\"|')", text))
    energies = dict((int(m.group(1)), float(m.group(2))) for m in re.finditer(
        re.escape(table) + r"\['%s-%s'\s*%\s*\(dbse,\s*'?(\d+)'?\s*\)\]\s*=\s*([-+0-9.]+)", text))
    out = []
    for m in re.finditer(r"GEOS\['%s-%s-dimer'\s*%\s*\(dbse,\s*'?(\d+)'?\s*\)\]\s*=\s*"
                         r"qcdb\.Molecule\(\"\"\"(.*?)\"\"\"\)", text, re.S):
        index = int(m.group(1))
        symbols, xyz, frag = [], [], []
        part = 0
        for line in m.group(2).strip().splitlines():
            tokens = line.split()
            if not tokens or tokens[0].lower() == "units":
                continue
            if tokens[0] == "--":
                part += 1
                continue
            if len(tokens) == 4:
                symbols.append(tokens[0])
                xyz.append([float(v) for v in tokens[1:]])
                frag.append(part)
        out.append({
            "atomic_numbers": np.array([atomic_number(s) for s in symbols], dtype=np.int64),
            "pos": np.array(xyz, dtype=np.float64),
            "fragment": np.array(frag, dtype=np.int64),
            "label": labels.get(index, str(index)),
            "index": index,
            "interaction_energy": energies.get(index, float("nan")),
        })
    return sorted(out, key=lambda s: s["index"])


class _NoncovalentBuilder(DatasetBuilder):

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def load(self, *, split: Optional[str] = None, cache_dir: Path,
             reference: str = "original", **kwargs) -> list[dict]:
        """The dimers (one split, ``"all"``).

        Parameters
        ----------
        split : str, optional
            ``None`` or ``"all"``.
        cache_dir : Path
            Download cache.
        reference : str, optional
            ``"original"`` or ``"revised"`` reference interaction energies.
        """
        if split not in (None, "all"):
            raise ValueError(f"{self.name} has one split, 'all'")
        if reference not in _REFERENCE[self.name]:
            raise ValueError(f"reference must be one of {sorted(_REFERENCE[self.name])}")
        fname, md5 = _FILES[self.name]
        path = download_file(_URL.format(name=fname[:-3]), Path(cache_dir) / self.name / fname, md5,
                             quiet=True)
        data = _parse(Path(path).read_text(), _REFERENCE[self.name][reference])
        return data if split == "all" else {"all": data}


register_dataset(_NoncovalentBuilder(
    "s22", "S22: 22 noncovalent dimers with CCSD(T)/CBS interaction energies (kcal/mol)"))
register_dataset(_NoncovalentBuilder(
    "s66", "S66: 66 noncovalent dimers with CCSD(T)/CBS interaction energies (kcal/mol)"))
