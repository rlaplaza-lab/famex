"""SMILES to 3D embedding.

``embedder='pysmiles'`` (default) is distance geometry from UFF bounds. Install
it with ``pip install famex[smiles]``. ``embedder='rdkit'`` is RDKit ETKDGv3.
Install it with ``pip install famex[rdkit]``.
"""

from famex.embed.api import (
    DEFAULT_EMBEDDER,
    EMBEDDERS,
    embed_smiles,
    normalize_embedder,
    smiles_to_atoms,
)

__all__ = [
    "DEFAULT_EMBEDDER",
    "EMBEDDERS",
    "embed_smiles",
    "normalize_embedder",
    "smiles_to_atoms",
]
