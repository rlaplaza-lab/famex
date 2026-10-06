"""Parse OpenSMILES into a small internal molecular graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ase.data import atomic_numbers


@dataclass(frozen=True)
class ChiralCenter:
    """Tetrahedral stereo as an ordered neighbor tuple.

    ``neighbors`` follows the pysmiles ``@`` convention (``@@`` is already
    swapped into that order). The signed volume of the first three
    neighbors around ``center`` should be positive.
    """

    center: int
    neighbors: tuple[int, int, int, int]


@dataclass(frozen=True)
class EZBond:
    """Cis/trans label on one ligand pair of a double bond."""

    ligand_a: int
    anchor_a: int
    anchor_b: int
    ligand_b: int
    kind: str


@dataclass
class MolGraph:
    """Elements, bonds, charge, and stereo extracted from a SMILES string."""

    elements: list[str]
    numbers: list[int]
    charges: list[int]
    aromatic: list[bool]
    bond_orders: dict[tuple[int, int], float]
    chiral: list[ChiralCenter]
    ez: list[EZBond]
    total_charge: int
    multiplicity: int
    implicit_h: list[int]

    @property
    def n_atoms(self) -> int:
        return len(self.elements)

    @property
    def bonds(self) -> list[tuple[int, int, float]]:
        return [(i, j, order) for (i, j), order in sorted(self.bond_orders.items())]

    def neighbor_lists(self) -> list[list[int]]:
        """Sorted neighbor indices for each atom. Zero-order contacts are absent."""
        neigh: list[list[int]] = [[] for _ in range(self.n_atoms)]
        for i, j, _order in self.bonds:
            neigh[i].append(j)
            neigh[j].append(i)
        for nbs in neigh:
            nbs.sort()
        return neigh

    def angle_indices(self) -> list[tuple[int, int, int]]:
        """Return (i, j, k) angles with ``j`` as the central atom."""
        angles: list[tuple[int, int, int]] = []
        for center, nbs in enumerate(self.neighbor_lists()):
            for left, atom_i in enumerate(nbs):
                for atom_k in nbs[left + 1 :]:
                    angles.append((atom_i, center, atom_k))
        return angles

    def connected_components(self) -> list[list[int]]:
        """Atom indices of each bonded component, lowest index first."""
        neigh = self.neighbor_lists()
        seen = [False] * self.n_atoms
        components: list[list[int]] = []
        for start in range(self.n_atoms):
            if seen[start]:
                continue
            seen[start] = True
            stack = [start]
            component: list[int] = []
            while stack:
                node = stack.pop()
                component.append(node)
                for nb in neigh[node]:
                    if not seen[nb]:
                        seen[nb] = True
                        stack.append(nb)
            components.append(sorted(component))
        return components


def _read_smiles(smiles: str, *, explicit_hydrogen: bool, strict: bool) -> Any:
    """Parse SMILES with pysmiles.

    Imported on demand so the core FAMEX install does not require the
    optional ``famex[smiles]`` extra.
    """
    try:
        from pysmiles import read_smiles
    except ImportError as exc:
        raise ImportError(
            "pysmiles is required for embedder='pysmiles'. "
            "Install it with: pip install 'famex[smiles]'"
        ) from exc
    # Nonstandard valence (radicals, carbonyl ligands) is accepted on purpose.
    logging.getLogger("pysmiles").setLevel(logging.ERROR)
    return read_smiles(
        smiles,
        explicit_hydrogen=explicit_hydrogen,
        zero_order_bonds=False,
        reinterpret_aromatic=True,
        strict=strict,
    )


def _load_mol(smiles: str, *, explicit_hydrogen: bool) -> Any:
    try:
        return _read_smiles(smiles, explicit_hydrogen=explicit_hydrogen, strict=True)
    except ImportError:
        raise
    except KeyError:
        # Nonstandard valence (radicals, some ligands) is a warning under strict=False.
        try:
            return _read_smiles(smiles, explicit_hydrogen=explicit_hydrogen, strict=False)
        except ImportError:
            raise
        except Exception as exc:
            raise ValueError(f"Invalid SMILES {smiles!r}: {exc}") from exc
    except (SyntaxError, ValueError, IndexError) as exc:
        raise ValueError(f"Invalid SMILES {smiles!r}: {exc}") from exc


def _bond_key(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def _chiral_neighbors(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, tuple) or len(value) != 4:
        return None
    try:
        return (int(value[0]), int(value[1]), int(value[2]), int(value[3]))
    except (TypeError, ValueError):
        return None


def _guess_multiplicity(numbers: list[int], implicit_h: list[int], charge: int) -> int:
    n_electrons = int(sum(numbers) + sum(implicit_h) - charge)
    if n_electrons % 2:
        return 2
    return 1


def parse_smiles(smiles: str, *, add_h: bool = True) -> MolGraph:
    """Parse an OpenSMILES string into a :class:`MolGraph`.

    Parameters
    ----------
    smiles : str
        OpenSMILES, including bracket atoms for metals.
    add_h : bool, default True
        Materialize implicit hydrogens as atoms. Metals do not receive
        implicit hydrogens unless the SMILES writes them.

    Returns
    -------
    MolGraph

    Raises
    ------
    ImportError
        If ``pysmiles`` is not installed.
    ValueError
        If ``smiles`` is empty, contains a wildcard, or is not valid SMILES.
    """
    text = smiles.strip()
    if not text:
        raise ValueError("SMILES string is empty")

    mol = _load_mol(text, explicit_hydrogen=add_h)
    raw_ids = sorted(int(node) for node in mol.nodes)
    if not raw_ids:
        raise ValueError(f"Invalid SMILES {text!r}: no atoms")
    id_map = {old: new for new, old in enumerate(raw_ids)}

    elements: list[str] = []
    numbers: list[int] = []
    charges: list[int] = []
    aromatic: list[bool] = []
    implicit_h: list[int] = []

    for old in raw_ids:
        data = mol.nodes[old]
        element = data.get("element")
        if not element or element == "*":
            raise ValueError(
                f"SMILES {text!r} contains a wildcard atom; every atom needs an element"
            )
        symbol = str(element)
        if symbol not in atomic_numbers:
            raise ValueError(f"Unknown element {symbol!r} in SMILES {text!r}")
        elements.append(symbol)
        numbers.append(int(atomic_numbers[symbol]))
        charges.append(int(data.get("charge") or 0))
        aromatic.append(bool(data.get("aromatic", False)))
        implicit_h.append(int(data.get("hcount") or 0))

    bond_orders: dict[tuple[int, int], float] = {}
    for u, v, data in mol.edges(data=True):
        raw_order = data.get("order", 1)
        order = 1.0 if raw_order is None else float(raw_order)
        if order <= 0.0:
            continue
        i = id_map[int(u)]
        j = id_map[int(v)]
        if i == j:
            continue
        bond_orders[_bond_key(i, j)] = order

    chiral: list[ChiralCenter] = []
    for old in raw_ids:
        neighs = _chiral_neighbors(mol.nodes[old].get("rs_isomer"))
        if neighs is None:
            continue
        center = id_map[old]
        try:
            mapped = tuple(id_map[n] for n in neighs)
        except KeyError:
            continue
        if len(set(mapped)) != 4 or center in mapped:
            continue
        chiral.append(
            ChiralCenter(center=center, neighbors=(mapped[0], mapped[1], mapped[2], mapped[3]))
        )
    chiral.sort(key=lambda spec: spec.center)

    ez: list[EZBond] = []
    seen_ez: set[tuple[int, int, int, int, str]] = set()
    for old in raw_ids:
        raw = mol.nodes[old].get("ez_isomer")
        if not raw:
            continue
        records = [raw] if isinstance(raw, tuple) else list(raw)
        for record in records:
            if not isinstance(record, tuple) or len(record) < 5:
                continue
            kind = str(record[4])
            if kind not in {"cis", "trans"}:
                continue
            try:
                lig_a = id_map[int(record[0])]
                anchor_a = id_map[int(record[1])]
                anchor_b = id_map[int(record[2])]
                lig_b = id_map[int(record[3])]
            except (KeyError, TypeError, ValueError):
                continue
            key = (lig_a, anchor_a, anchor_b, lig_b, kind)
            reverse = (lig_b, anchor_b, anchor_a, lig_a, kind)
            if key in seen_ez or reverse in seen_ez:
                continue
            seen_ez.add(key)
            ez.append(
                EZBond(
                    ligand_a=lig_a,
                    anchor_a=anchor_a,
                    anchor_b=anchor_b,
                    ligand_b=lig_b,
                    kind=kind,
                )
            )
    ez.sort(
        key=lambda spec: (spec.ligand_a, spec.anchor_a, spec.anchor_b, spec.ligand_b, spec.kind)
    )

    total_charge = int(sum(charges))
    return MolGraph(
        elements=elements,
        numbers=numbers,
        charges=charges,
        aromatic=aromatic,
        bond_orders=bond_orders,
        chiral=chiral,
        ez=ez,
        total_charge=total_charge,
        multiplicity=_guess_multiplicity(numbers, implicit_h, total_charge),
        implicit_h=implicit_h,
    )
