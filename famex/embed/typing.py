"""UFF atom typing and ideal bond lengths.

Tabulated types are transcribed from Rappé et al., J. Am. Chem. Soc. 1992,
114, 10024. Elements without a matching type use ASE covalent radii and a
coordination-number angle. Four-coordinate centers default to tetrahedral;
ordinary SMILES does not distinguish square-planar metals.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ase.data import covalent_radii, vdw_radii

from famex.embed.parse import MolGraph


@dataclass(frozen=True)
class AtomParams:
    """Parameters used for bounds and the short UFF-like cleanup."""

    name: str
    z: int
    r: float
    theta: float
    x: float
    d: float
    zstar: float
    chi: float
    cn: int


def _count_orders(orders: list[float]) -> tuple[int, int]:
    n_double = 0
    n_triple = 0
    for order in orders:
        if order >= 2.5:
            n_triple += 1
        elif order >= 1.75:
            n_double += 1
    return n_double, n_triple


def uff_type_name(element: str, aromatic: bool, orders: list[float], cn: int) -> str | None:
    """Map an atom to a tabulated UFF type, or ``None`` to synthesize one.

    The digit in names such as ``C_3`` is the UFF geometry label (sp3, sp2,
    resonant, sp), not the oxidation state.
    """
    n_double, n_triple = _count_orders(orders)
    if element == "H":
        return "H_"
    if element == "C":
        if aromatic:
            return "C_R"
        if n_triple or n_double >= 2:
            return "C_1"
        if n_double or cn == 3:
            return "C_2"
        return "C_3"
    if element == "N":
        if aromatic:
            return "N_R"
        if n_triple or cn <= 1:
            return "N_1"
        if n_double or cn == 2:
            return "N_2"
        return "N_3"
    if element == "O":
        if aromatic:
            return "O_R"
        if n_triple:
            return "O_1"
        if n_double:
            return "O_2"
        return "O_3"
    if element == "S":
        if cn > 3:
            return None
        return "S_3"
    if element == "P":
        if cn > 3:
            return None
        return "P_3"
    if element == "B":
        return "B_2" if cn <= 3 else "B_3"
    if element == "Si":
        return "Si3" if cn >= 4 else None
    if element == "F":
        return "F_"
    if element == "Cl":
        return "Cl"
    if element == "Br":
        return "Br"
    if element == "I":
        return "I_"
    return None


def _fallback_theta(cn: int) -> float:
    """Angle (degrees) for an element that has no UFF atom type.

    Two-coordinate metals default to linear, four-coordinate centers to
    tetrahedral, and five-or-higher coordination to 90 degrees. The cleanup
    potential also accepts 180 degree trans angles when the coordination
    number is 5 or 6.
    """
    if cn <= 2:
        return 180.0
    if cn == 3:
        return 120.0
    if cn == 4:
        return 109.471
    return 90.0


@lru_cache(maxsize=1)
def _load_uff() -> dict[str, Any]:
    path = Path(__file__).with_name("uff_params.json")
    with path.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


@lru_cache(maxsize=1)
def _type_table() -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    raw_types = _load_uff()["types"]
    for name, row in raw_types.items():
        table[str(name)] = {
            "r": float(row["r"]),
            "theta": float(row["theta"]),
            "x": float(row["x"]),
            "d": float(row["d"]),
            "zstar": float(row["zstar"]),
            "chi": max(float(row["chi"]), 0.1),
        }
    return table


def _constants() -> tuple[float, float]:
    data = _load_uff()
    return float(data["lambda_bo"]), float(data["k_scale"])


def _from_row(name: str, row: dict[str, float], *, z: int, cn: int) -> AtomParams:
    return AtomParams(
        name=name,
        z=z,
        r=row["r"],
        theta=row["theta"],
        x=row["x"],
        d=row["d"],
        zstar=row["zstar"],
        chi=row["chi"],
        cn=cn,
    )


def _synthesize(element: str, z: int, cn: int) -> AtomParams:
    if 0 <= z < len(covalent_radii):
        r_cov = float(covalent_radii[z])
    else:
        r_cov = float("nan")
    if not math.isfinite(r_cov) or r_cov <= 0.2:
        r_cov = 1.2
    if 0 <= z < len(vdw_radii):
        vdw = float(vdw_radii[z])
    else:
        vdw = float("nan")
    if not math.isfinite(vdw) or vdw <= 0.2:
        x_value = max(2.5, 2.0 * r_cov + 0.8)
    else:
        x_value = max(2.5, 2.0 * vdw)
    return AtomParams(
        name=f"{element}{cn}",
        z=z,
        r=r_cov,
        theta=_fallback_theta(cn),
        x=x_value,
        d=0.05,
        zstar=1.5,
        chi=2.2,
        cn=cn,
    )


def assign_parameters(graph: MolGraph) -> list[AtomParams]:
    """Assign one :class:`AtomParams` record per atom."""
    table = _type_table()
    neigh = graph.neighbor_lists()
    params: list[AtomParams] = []
    for index, element in enumerate(graph.elements):
        orders = [
            graph.bond_orders[(index, nb) if index < nb else (nb, index)] for nb in neigh[index]
        ]
        cn = len(orders)
        name = uff_type_name(element, graph.aromatic[index], orders, cn)
        if name is not None and name in table:
            params.append(_from_row(name, table[name], z=graph.numbers[index], cn=cn))
        else:
            params.append(_synthesize(element, graph.numbers[index], cn))
    return params


def bond_length(a: AtomParams, b: AtomParams, order: float) -> float:
    """UFF equilibrium distance (Angstrom), including bond-order and electronegativity corrections."""
    lam, _k_scale = _constants()
    rij = a.r + b.r
    if order > 0.0 and abs(order - 1.0) > 1e-8:
        rij -= lam * (a.r + b.r) * math.log(order)
    denom = a.chi * a.r + b.chi * b.r
    if denom > 1e-8:
        ren = a.r * b.r * (math.sqrt(a.chi) - math.sqrt(b.chi)) ** 2 / denom
        rij -= ren
    return max(rij, 0.4)


def bond_force_constant(a: AtomParams, b: AtomParams, length: float) -> float:
    """Harmonic bond constant in kcal/mol/Angstrom^2."""
    _lam, k_scale = _constants()
    k = k_scale * a.zstar * b.zstar / max(length, 0.4) ** 3
    return float(min(max(k, 50.0), 5000.0))


def angle_force_constant(
    z_i: float,
    z_k: float,
    r_ij: float,
    r_jk: float,
    theta_deg: float,
) -> float:
    """Harmonic angle constant in kcal/mol/radian^2 for the end atoms ``i`` and ``k``."""
    _lam, k_scale = _constants()
    theta = math.radians(theta_deg)
    cos_t = math.cos(theta)
    rik2 = r_ij * r_ij + r_jk * r_jk - 2.0 * r_ij * r_jk * cos_t
    if rik2 < 1e-4 or abs(theta_deg - 180.0) < 2.0:
        k = 200.0 * math.sqrt(max(z_i * z_k, 1e-6))
    else:
        rik = math.sqrt(rik2)
        k = (
            k_scale
            * z_i
            * z_k
            / rik**5
            * r_ij
            * r_jk
            * (3.0 * r_ij * r_jk * (1.0 - cos_t * cos_t) - rik2 * cos_t)
        )
    if not math.isfinite(k) or k <= 0.0:
        k = 100.0 * math.sqrt(max(z_i * z_k, 1e-6))
    return float(min(max(k, 10.0), 2000.0))


def preferred_angle(theta: float, theta0: float, cn: int) -> float:
    """Return the reference angle in radians.

    Coordination numbers of 5 and 6 use basins at 90, 120 (CN 5), and 180
    degrees so trans contacts are not pulled down to the cis angle.
    """
    targets: tuple[float, ...]
    if cn >= 6:
        targets = (0.5 * math.pi, math.pi)
    elif cn == 5:
        targets = (0.5 * math.pi, 2.0 * math.pi / 3.0, math.pi)
    else:
        return theta0
    return min(targets, key=lambda target: abs(theta - target))
