"""Short UFF-like relaxation: harmonic bonds and angles plus a Lennard-Jones wall.

This is a coordinate cleanup, not a replacement for a UMA or PET minimization.
Torsions are not parameterized per element; cis/trans restraints are applied
geometrically after this step.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from famex.embed.parse import MolGraph
from famex.embed.typing import (
    AtomParams,
    angle_force_constant,
    bond_force_constant,
    bond_length,
    preferred_angle,
)


@dataclass(frozen=True)
class _BondTerm:
    i: int
    j: int
    r0: float
    k: float


@dataclass(frozen=True)
class _AngleTerm:
    i: int
    j: int
    k: int
    theta0: float
    k_ang: float
    cn: int


@dataclass(frozen=True)
class _PairTerm:
    i: int
    j: int
    x: float
    d: float
    weight: float


def _topo_distance(neigh: list[list[int]], start: int) -> list[int]:
    dist = [-1] * len(neigh)
    dist[start] = 0
    queue = [start]
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        for nb in neigh[node]:
            if dist[nb] < 0:
                dist[nb] = dist[node] + 1
                queue.append(nb)
    return dist


def _lj(r: float, x_ij: float, depth: float) -> tuple[float, float]:
    """Lennard-Jones energy and dE/dr, with a smooth wall inside ``0.7 * x``."""
    r_floor = 0.7 * x_ij

    def at(rv: float) -> tuple[float, float]:
        sr6 = (x_ij / rv) ** 6
        sr12 = sr6 * sr6
        energy = depth * (sr12 - 2.0 * sr6)
        deriv = 12.0 * depth * (sr6 - sr12) / rv
        return energy, deriv

    if r >= r_floor:
        return at(r)
    e_floor, d_floor = at(r_floor)
    delta = r - r_floor
    k_wall = 200.0
    energy = e_floor + d_floor * delta + 0.5 * k_wall * delta * delta
    deriv = d_floor + k_wall * delta
    return energy, deriv


class _UFFModel:
    """Cached bond, angle, and nonbonded terms for one graph."""

    def __init__(self, graph: MolGraph, params: Sequence[AtomParams]) -> None:
        self.n_atoms = graph.n_atoms
        self.bonds: list[_BondTerm] = []
        for i, j, order in graph.bonds:
            length = bond_length(params[i], params[j], order)
            self.bonds.append(
                _BondTerm(
                    i=i,
                    j=j,
                    r0=length,
                    k=bond_force_constant(params[i], params[j], length),
                )
            )

        self.angles: list[_AngleTerm] = []
        ideal = {(term.i, term.j): term.r0 for term in self.bonds}
        ideal.update({(term.j, term.i): term.r0 for term in self.bonds})
        for i, j, k in graph.angle_indices():
            r_ij = ideal.get((i, j))
            r_jk = ideal.get((j, k))
            if r_ij is None or r_jk is None:
                continue
            self.angles.append(
                _AngleTerm(
                    i=i,
                    j=j,
                    k=k,
                    theta0=math.radians(params[j].theta),
                    k_ang=angle_force_constant(
                        params[i].zstar,
                        params[k].zstar,
                        r_ij,
                        r_jk,
                        params[j].theta,
                    ),
                    cn=params[j].cn,
                )
            )

        neigh = graph.neighbor_lists()
        self.pairs: list[_PairTerm] = []
        n = graph.n_atoms
        topo = [_topo_distance(neigh, i) for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                separation = topo[i][j]
                if 0 <= separation < 4:
                    continue
                x_ij = math.sqrt(max(params[i].x * params[j].x, 1e-8))
                depth = math.sqrt(max(params[i].d * params[j].d, 0.0))
                weight = 0.5 if separation == 4 else 1.0
                if x_ij <= 0.0 or depth <= 0.0:
                    continue
                self.pairs.append(_PairTerm(i=i, j=j, x=x_ij, d=depth, weight=weight))

    def energy_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        coords = np.asarray(x, dtype=float).reshape(self.n_atoms, 3)
        grad = np.zeros_like(coords)
        energy = 0.0

        for bond in self.bonds:
            vec = coords[bond.j] - coords[bond.i]
            dist = float(np.linalg.norm(vec))
            if dist < 1e-8:
                hat = np.array([1.0, 0.0, 0.0])
                dist = 1e-8
            else:
                hat = vec / dist
            diff = dist - bond.r0
            energy += 0.5 * bond.k * diff * diff
            d_e = bond.k * diff
            grad[bond.j] += d_e * hat
            grad[bond.i] -= d_e * hat

        for angle in self.angles:
            v1 = coords[angle.i] - coords[angle.j]
            v2 = coords[angle.k] - coords[angle.j]
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1e-8 or n2 < 1e-8:
                continue
            u1 = v1 / n1
            u2 = v2 / n2
            cos_t = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
            theta = math.acos(cos_t)
            target = preferred_angle(theta, angle.theta0, angle.cn)
            delta = theta - target
            energy += 0.5 * angle.k_ang * delta * delta
            sin_t = math.sin(theta)
            sin_use = math.copysign(max(abs(sin_t), 0.05), sin_t if sin_t != 0.0 else 1.0)
            d_theta = angle.k_ang * delta
            g1 = d_theta * (cos_t * u1 - u2) / (n1 * sin_use)
            g2 = d_theta * (cos_t * u2 - u1) / (n2 * sin_use)
            grad[angle.i] += g1
            grad[angle.k] += g2
            grad[angle.j] -= g1 + g2

        for pair in self.pairs:
            vec = coords[pair.j] - coords[pair.i]
            dist = float(np.linalg.norm(vec))
            if dist < 1e-6:
                hat = np.array([1.0, 0.0, 0.0])
                dist = 1e-6
            else:
                hat = vec / dist
            pair_e, pair_d = _lj(dist, pair.x, pair.d)
            energy += pair.weight * pair_e
            d_e = pair.weight * pair_d
            grad[pair.j] += d_e * hat
            grad[pair.i] -= d_e * hat

        flat = np.ascontiguousarray(grad.reshape(-1))
        return float(energy), flat


def relax_coordinates(
    coords: np.ndarray,
    graph: MolGraph,
    params: Sequence[AtomParams],
) -> np.ndarray:
    """Minimize the UFF-like energy. Returns the input if the step is not finite or higher."""
    if graph.n_atoms <= 1:
        return np.array(coords, dtype=float, copy=True)
    model = _UFFModel(graph, params)
    x0 = np.asarray(coords, dtype=float).reshape(-1)
    e0, _grad0 = model.energy_grad(x0)
    result = minimize(
        model.energy_grad,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 80, "maxfun": 200, "ftol": 1e-8, "gtol": 1e-4},
    )
    candidate = np.asarray(result.x, dtype=float).reshape(graph.n_atoms, 3)
    if not np.isfinite(candidate).all() or not math.isfinite(float(result.fun)):
        return x0.reshape(graph.n_atoms, 3)
    if float(result.fun) > e0 + 1e-6:
        return x0.reshape(graph.n_atoms, 3)
    return candidate
