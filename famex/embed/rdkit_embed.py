"""RDKit ETKDGv3 embedding.

RDKit is an optional extra. This module imports it only when
``embedder='rdkit'`` is selected.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from ase.data import atomic_numbers

from famex.embed.parse import _guess_multiplicity
from famex.io.geometry import Geometry

ToGeometry = Callable[[str, int, int, list[str], np.ndarray], Geometry]


def _import_rdkit() -> tuple[Any, Any, Any]:
    """Import RDKit on demand.

    The embed package must stay importable when only ``famex[smiles]`` is
    installed, so RDKit is not a module-level import.
    """
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for embedder='rdkit'. Install it with: pip install 'famex[rdkit]'"
        ) from exc
    return Chem, AllChem, RDLogger


def _charge_and_multiplicity(mol: Any) -> tuple[int, int]:
    numbers = [int(atomic_numbers[atom.GetSymbol()]) for atom in mol.GetAtoms()]
    implicit_h = [int(atom.GetNumImplicitHs()) for atom in mol.GetAtoms()]
    charge = int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
    return charge, _guess_multiplicity(numbers, implicit_h, charge)


def _conformer_coordinates(mol: Any, conf_id: int) -> np.ndarray:
    conf = mol.GetConformer(int(conf_id))
    return np.array(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=float,
    )


def embed_with_rdkit(
    smiles: str,
    *,
    n_conf: int,
    seed: int,
    add_h: bool,
    to_geometry: ToGeometry,
) -> list[Geometry]:
    """Embed ``n_conf`` conformers with RDKit ETKDGv3.

    ``to_geometry`` is the shared constructor from :mod:`famex.embed.api`.
    It is passed in so this module can stay free of a circular import.
    """
    chem, allchem, rdlogger = _import_rdkit()
    rdlogger.DisableLog("rdApp.*")

    text = smiles.strip()
    if not text:
        raise ValueError("SMILES string is empty")
    mol = chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError(f"Invalid SMILES {text!r}")
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        raise ValueError(f"SMILES {text!r} contains a wildcard atom; every atom needs an element")
    if add_h:
        mol = chem.AddHs(mol)

    params = allchem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 1
    params.enableSequentialRandomSeeds = True
    conf_ids = list(allchem.EmbedMultipleConfs(mol, numConfs=int(n_conf), params=params))
    if not conf_ids:
        raise RuntimeError(
            f"RDKit ETKDGv3 failed to embed SMILES {text!r}. "
            "Some metals and nonstandard valences have no RDKit distance-geometry parameters."
        )
    if len(conf_ids) < n_conf:
        raise RuntimeError(
            f"RDKit ETKDGv3 produced {len(conf_ids)} conformer(s) for {text!r}; requested {n_conf}"
        )

    charge, multiplicity = _charge_and_multiplicity(mol)
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return [
        to_geometry(text, charge, multiplicity, elements, _conformer_coordinates(mol, conf_id))
        for conf_id in conf_ids
    ]
