"""Distance bounds from bonds, valence angles, generic 1-4 windows, and van der Waals floors."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from famex.embed.parse import EZBond, MolGraph
from famex.embed.typing import AtomParams, bond_length


def _pair_distance(r_12: float, r_23: float, angle_deg: float) -> float:
    theta = math.radians(angle_deg)
    value = r_12 * r_12 + r_23 * r_23 - 2.0 * r_12 * r_23 * math.cos(theta)
    return math.sqrt(max(value, 1e-8))


def torsion_distance(
    r_ab: float,
    r_bc: float,
    r_cd: float,
    ang_abc: float,
    ang_bcd: float,
    phi: float,
) -> float:
    """Distance between the outer atoms of a four-atom chain.

    Parameters
    ----------
    r_ab, r_bc, r_cd : float
        Bond lengths in Angstrom.
    ang_abc, ang_bcd : float
        Bond angles in degrees.
    phi : float
        Dihedral angle in radians. ``0`` is cis and ``pi`` is trans.
    """
    a_ang = math.radians(ang_abc)
    b_ang = math.radians(ang_bcd)
    atom_a = np.array([r_ab * math.sin(a_ang), 0.0, r_ab * math.cos(a_ang)])
    atom_c = np.array([0.0, 0.0, r_bc])
    offset = np.array(
        [
            r_cd * math.sin(b_ang) * math.cos(phi),
            r_cd * math.sin(b_ang) * math.sin(phi),
            -r_cd * math.cos(b_ang),
        ]
    )
    return float(np.linalg.norm(atom_c + offset - atom_a))


def _angle_window(theta: float, cn: int) -> tuple[float, float]:
    if cn >= 5:
        return 80.0, 180.0
    if theta >= 170.0:
        return max(150.0, theta - 8.0), 180.0
    pad = 12.0
    return max(40.0, theta - pad), min(180.0, theta + pad)


def _torsion_window(
    r_ab: float,
    r_bc: float,
    r_cd: float,
    ang_b: float,
    ang_c: float,
    stereo: str | None,
) -> tuple[float, float]:
    if stereo == "cis":
        phis = (0.0, 25.0)
    elif stereo == "trans":
        phis = (155.0, 180.0)
    else:
        phis = (0.0, 180.0)
    dists = [torsion_distance(r_ab, r_bc, r_cd, ang_b, ang_c, math.radians(phi)) for phi in phis]
    lo = min(dists) * 0.97
    hi = max(dists) * 1.03
    return lo, max(hi, lo + 1e-3)


def _ez_lookup(ez: list[EZBond]) -> dict[tuple[int, int, int, int], str]:
    lookup: dict[tuple[int, int, int, int], str] = {}
    for spec in ez:
        lookup[(spec.ligand_a, spec.anchor_a, spec.anchor_b, spec.ligand_b)] = spec.kind
        lookup[(spec.ligand_b, spec.anchor_b, spec.anchor_a, spec.ligand_a)] = spec.kind
    return lookup


def _smooth(lower: np.ndarray, upper: np.ndarray) -> None:
    """Triangle-inequality smoothing. Upper bounds use Floyd-Warshall."""
    n = int(lower.shape[0])
    for k in range(n):
        via = upper[:, k][:, None] + upper[k, :][None, :]
        np.minimum(upper, via, out=upper)
    np.fill_diagonal(upper, 0.0)

    for _pass in range(4):
        changed = False
        for k in range(n):
            for i in range(n):
                if i == k:
                    continue
                lik = float(lower[i, k])
                uik = float(upper[i, k])
                for j in range(i + 1, n):
                    if j == k:
                        continue
                    cand = lik - float(upper[k, j])
                    if cand > float(lower[i, j]):
                        lower[i, j] = lower[j, i] = cand
                        changed = True
                    cand = float(lower[j, k]) - uik
                    if cand > float(lower[i, j]):
                        lower[i, j] = lower[j, i] = cand
                        changed = True
        if not changed:
            break
    np.fill_diagonal(lower, 0.0)


def distance_bounds(
    graph: MolGraph,
    params: Sequence[AtomParams],
    indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper distance bounds for ``indices`` (default: every atom).

    Bounds are in the local order of ``indices``. Bond windows are restored
    after triangle smoothing so a long-range constraint cannot erase a bond.
    """
    if indices is None:
        indices = list(range(graph.n_atoms))
    g2l = {global_i: local_i for local_i, global_i in enumerate(indices)}
    m = len(indices)
    lower = np.zeros((m, m), dtype=float)
    span = max(10.0, 2.0 * m)
    upper = np.full((m, m), span, dtype=float)
    np.fill_diagonal(upper, 0.0)
    kind = np.zeros((m, m), dtype=np.int8)

    ideal: dict[tuple[int, int], float] = {}
    bond_windows: dict[tuple[int, int], tuple[float, float]] = {}
    for i, j, order in graph.bonds:
        if i not in g2l or j not in g2l:
            continue
        length = bond_length(params[i], params[j], order)
        ideal[(i, j)] = length
        lo = length * 0.98
        hi = length * 1.02
        a = g2l[i]
        b = g2l[j]
        lower[a, b] = lower[b, a] = lo
        upper[a, b] = upper[b, a] = hi
        kind[a, b] = kind[b, a] = 1
        bond_windows[(a, b) if a < b else (b, a)] = (lo, hi)

    neigh = graph.neighbor_lists()
    for center in indices:
        nbs = [nb for nb in neigh[center] if nb in g2l]
        theta = params[center].theta
        cn = params[center].cn
        ang_lo, ang_hi = _angle_window(theta, cn)
        for left, atom_i in enumerate(nbs):
            key_ic = tuple(sorted((atom_i, center)))
            if key_ic not in ideal:
                continue
            r_ic = ideal[key_ic]
            for atom_k in nbs[left + 1 :]:
                key_ck = tuple(sorted((center, atom_k)))
                if key_ck not in ideal:
                    continue
                a = g2l[atom_i]
                b = g2l[atom_k]
                if kind[a, b] == 1:
                    continue
                d_lo = _pair_distance(r_ic, ideal[key_ck], ang_lo)
                d_hi = _pair_distance(r_ic, ideal[key_ck], ang_hi)
                new_lo, new_hi = min(d_lo, d_hi), max(d_lo, d_hi)
                if kind[a, b] == 2:
                    new_lo = max(float(lower[a, b]), new_lo)
                    new_hi = min(float(upper[a, b]), new_hi)
                    if new_hi < new_lo:
                        new_lo = new_hi
                lower[a, b] = lower[b, a] = new_lo
                upper[a, b] = upper[b, a] = new_hi
                kind[a, b] = kind[b, a] = 2

    stereo = _ez_lookup(graph.ez)
    for j, k, _order in graph.bonds:
        if j not in g2l or k not in g2l:
            continue
        bond_key = (j, k)
        r_jk = ideal[bond_key]
        for atom_a in neigh[j]:
            if atom_a == k or atom_a not in g2l:
                continue
            key_aj = tuple(sorted((atom_a, j)))
            if key_aj not in ideal:
                continue
            for atom_b in neigh[k]:
                if atom_b in {j, atom_a} or atom_b not in g2l:
                    continue
                key_kb = tuple(sorted((k, atom_b)))
                if key_kb not in ideal:
                    continue
                a = g2l[atom_a]
                b = g2l[atom_b]
                if kind[a, b] in {1, 2}:
                    continue
                label = stereo.get((atom_a, j, k, atom_b))
                new_lo, new_hi = _torsion_window(
                    ideal[key_aj],
                    r_jk,
                    ideal[key_kb],
                    params[j].theta,
                    params[k].theta,
                    label,
                )
                if kind[a, b] == 3:
                    new_lo = max(float(lower[a, b]), new_lo)
                    new_hi = min(float(upper[a, b]), new_hi)
                    if new_hi < new_lo:
                        new_lo = new_hi
                lower[a, b] = lower[b, a] = new_lo
                upper[a, b] = upper[b, a] = new_hi
                kind[a, b] = kind[b, a] = 3

    for local_i, global_i in enumerate(indices):
        for local_j in range(local_i + 1, m):
            if kind[local_i, local_j] != 0:
                continue
            global_j = indices[local_j]
            x_ij = math.sqrt(max(params[global_i].x * params[global_j].x, 1e-8))
            lower[local_i, local_j] = lower[local_j, local_i] = 0.8 * x_ij

    _smooth(lower, upper)
    for (a, b), (lo, hi) in bond_windows.items():
        lower[a, b] = lower[b, a] = lo
        upper[a, b] = upper[b, a] = hi

    for i in range(m):
        for j in range(i + 1, m):
            if float(lower[i, j]) > float(upper[i, j]):
                if kind[i, j] == 1:
                    mid = 0.5 * (float(lower[i, j]) + float(upper[i, j]))
                    lower[i, j] = lower[j, i] = mid
                    upper[i, j] = upper[j, i] = mid
                else:
                    lower[i, j] = lower[j, i] = float(upper[i, j])
    np.maximum(lower, 0.0, out=lower)
    np.maximum(upper, 0.0, out=upper)
    np.fill_diagonal(lower, 0.0)
    np.fill_diagonal(upper, 0.0)
    return lower, upper
