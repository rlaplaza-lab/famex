"""Public SMILES to 3D entry points."""

from __future__ import annotations

from typing import Literal, overload

import numpy as np

from famex.embed.dgeom import embed_coordinates, geometry_score
from famex.embed.parse import parse_smiles
from famex.embed.typing import assign_parameters
from famex.io.geometry import Geometry
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)

EmbedderName = Literal["pysmiles", "rdkit"]
EMBEDDERS: tuple[EmbedderName, ...] = ("pysmiles", "rdkit")
DEFAULT_EMBEDDER: EmbedderName = "pysmiles"


def normalize_embedder(embedder: str) -> EmbedderName:
    """Return a known embedder name, or raise ``ValueError``."""
    key = embedder.strip().lower()
    for name in EMBEDDERS:
        if key == name:
            return name
    choices = ", ".join(EMBEDDERS)
    raise ValueError(f"Unknown SMILES embedder {embedder!r}. Choose from: {choices}")


def _to_geometry(
    smiles: str,
    graph_charge: int,
    graph_mult: int,
    elements: list[str],
    coords: np.ndarray,
    embedder: EmbedderName,
) -> Geometry:
    geom = Geometry(
        atoms=elements,
        positions=np.asarray(coords, dtype=float),
        charge=graph_charge,
        mult=graph_mult,
    )
    geom.info["charge"] = graph_charge
    geom.info["spin"] = graph_mult
    geom.info["smiles"] = smiles
    geom.info["embedder"] = embedder
    return geom


def _embed_pysmiles(
    smiles: str,
    *,
    n_conf: int,
    seed: int,
    add_h: bool,
) -> list[Geometry]:
    graph = parse_smiles(smiles, add_h=add_h)
    params = assign_parameters(graph)
    n_trials = max(4, n_conf * 3)
    ranked: list[tuple[float, np.ndarray]] = []
    for attempt in range(n_trials):
        rng = np.random.default_rng(seed + attempt)
        try:
            coords = embed_coordinates(
                graph,
                params,
                rng,
                deterministic=(attempt == 0),
            )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
            logger.debug("SMILES embed attempt %s failed: %s", attempt, exc)
            continue
        if not np.isfinite(coords).all():
            continue
        ranked.append((geometry_score(coords, graph, params), coords))
    if not ranked:
        raise RuntimeError(f"Distance geometry failed for SMILES {smiles!r}")
    if len(ranked) < n_conf:
        raise RuntimeError(
            f"Distance geometry produced {len(ranked)} conformer(s) for {smiles!r}; "
            f"requested {n_conf}"
        )
    ranked.sort(key=lambda item: item[0])
    best_score = ranked[0][0]
    if best_score > 0.5 * max(graph.n_atoms, 1):
        logger.warning(
            "Distance-geometry embedding for %s is a rough guess (score %.3f). "
            "Relax it with UMA or PET before production use.",
            smiles,
            best_score,
        )
    chosen = ranked[:n_conf]
    return [
        _to_geometry(
            smiles.strip(),
            graph.total_charge,
            graph.multiplicity,
            graph.elements,
            coords,
            "pysmiles",
        )
        for _score, coords in chosen
    ]


def embed_smiles(
    smiles: str,
    *,
    n_conf: int = 1,
    seed: int = 0,
    add_h: bool = True,
    embedder: str = DEFAULT_EMBEDDER,
) -> list[Geometry]:
    """Embed ``n_conf`` 3D guesses from a SMILES string.

    ``embedder='pysmiles'`` (the default) builds bounds from UFF, embeds them
    with classical distance geometry, and runs a short UFF-like cleanup.
    Install that path with ``pip install famex[smiles]``.

    ``embedder='rdkit'`` uses RDKit ETKDGv3. Install it with
    ``pip install famex[rdkit]``.

    Parameters
    ----------
    smiles : str
        OpenSMILES string.
    n_conf : int, default 1
        How many conformers to return.
    seed : int, default 0
        Seed for distance sampling. The first pysmiles trial is deterministic.
    add_h : bool, default True
        Add implicit hydrogens before embedding.
    embedder : {'pysmiles', 'rdkit'}, default 'pysmiles'
        Which optional embedding backend to use.

    Returns
    -------
    list of Geometry
        For ``pysmiles``, best-first. For ``rdkit``, ETKDGv3 conformer order.
    """
    if n_conf < 1:
        raise ValueError(f"n_conf must be >= 1, got {n_conf}")
    chosen = normalize_embedder(embedder)
    if chosen == "pysmiles":
        return _embed_pysmiles(smiles, n_conf=n_conf, seed=seed, add_h=add_h)
    # RDKit is an optional extra. Import it only when this backend is selected.
    from famex.embed.rdkit_embed import embed_with_rdkit

    return embed_with_rdkit(
        smiles,
        n_conf=n_conf,
        seed=seed,
        add_h=add_h,
        to_geometry=lambda text, charge, mult, elements, coords: _to_geometry(
            text, charge, mult, elements, coords, "rdkit"
        ),
    )


@overload
def smiles_to_atoms(
    smiles: str,
    *,
    n_conf: Literal[1] = 1,
    seed: int = 0,
    add_h: bool = True,
    embedder: str = DEFAULT_EMBEDDER,
) -> Geometry: ...


@overload
def smiles_to_atoms(
    smiles: str,
    *,
    n_conf: int,
    seed: int = 0,
    add_h: bool = True,
    embedder: str = DEFAULT_EMBEDDER,
) -> Geometry | list[Geometry]: ...


def smiles_to_atoms(
    smiles: str,
    *,
    n_conf: int = 1,
    seed: int = 0,
    add_h: bool = True,
    embedder: str = DEFAULT_EMBEDDER,
) -> Geometry | list[Geometry]:
    """Build a 3D geometry from SMILES.

    With ``n_conf=1`` (the default) this returns one
    :class:`~famex.io.geometry.Geometry`. With ``n_conf>1`` it returns that
    many geometries. See :func:`embed_smiles` for ``embedder``.

    ``pysmiles`` requires ``pip install famex[smiles]``. ``rdkit`` requires
    ``pip install famex[rdkit]``.
    """
    geoms = embed_smiles(smiles, n_conf=n_conf, seed=seed, add_h=add_h, embedder=embedder)
    if n_conf == 1:
        return geoms[0]
    return geoms
