"""Classical metric distance geometry, plus cis/trans and tetrahedral stereo fixes."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from famex.embed.bounds import distance_bounds
from famex.embed.cleanup import relax_coordinates
from famex.embed.parse import MolGraph
from famex.embed.typing import AtomParams, bond_length, preferred_angle


def _sample_distances(
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    *,
    deterministic: bool,
) -> np.ndarray:
    n = int(lower.shape[0])
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            lo = float(lower[i, j])
            hi = float(upper[i, j])
            if hi < lo:
                lo, hi = hi, lo
            if deterministic or hi - lo < 0.05:
                value = 0.5 * (lo + hi)
            else:
                value = float(rng.triangular(lo, 0.5 * (lo + hi), hi))
            dist[i, j] = dist[j, i] = value
    return dist


def _classical_mds(dist: np.ndarray) -> np.ndarray:
    n = int(dist.shape[0])
    if n == 1:
        return np.zeros((1, 3), dtype=float)
    if n == 2:
        return np.array([[0.0, 0.0, 0.0], [float(dist[0, 1]), 0.0, 0.0]], dtype=float)
    d2 = dist * dist
    gram = np.eye(n) - np.ones((n, n)) / n
    b_mat = -0.5 * gram @ d2 @ gram
    evals, evecs = np.linalg.eigh(b_mat)
    order = np.argsort(evals)[::-1][:3]
    lam = np.clip(evals[order], 0.0, None)
    coords = evecs[:, order] * np.sqrt(lam)
    return np.asarray(coords, dtype=float)


def _line_fallback(
    indices: Sequence[int],
    graph: MolGraph,
    params: Sequence[AtomParams],
) -> np.ndarray:
    g2l = {global_i: local_i for local_i, global_i in enumerate(indices)}
    m = len(indices)
    coords = np.zeros((m, 3), dtype=float)
    neigh: list[list[tuple[int, float]]] = [[] for _ in range(m)]
    for i, j, order in graph.bonds:
        if i not in g2l or j not in g2l:
            continue
        length = bond_length(params[i], params[j], order)
        a = g2l[i]
        b = g2l[j]
        neigh[a].append((b, length))
        neigh[b].append((a, length))
    placed = {0}
    queue = [0]
    while queue:
        node = queue.pop()
        for nb, length in neigh[node]:
            if nb in placed:
                continue
            direction = np.array(
                [1.0, 0.15 * ((nb % 5) - 2), 0.1 * ((node % 3) - 1)],
                dtype=float,
            )
            direction /= float(np.linalg.norm(direction))
            coords[nb] = coords[node] + direction * length
            placed.add(nb)
            queue.append(nb)
    for local_i in range(m):
        if local_i not in placed:
            coords[local_i, 0] = local_i * 1.5
    return coords


def _polish(coords: np.ndarray, targets: list[tuple[int, int, float]]) -> np.ndarray:
    """Pull bonded pairs toward their ideal lengths after the MDS projection."""
    coords = np.array(coords, dtype=float, copy=True)
    if not targets:
        return coords
    for _ in range(12):
        forces = np.zeros_like(coords)
        for i, j, r0 in targets:
            vec = coords[j] - coords[i]
            dist = float(np.linalg.norm(vec))
            if dist < 1e-8:
                continue
            direction = vec / dist
            err = r0 - dist
            forces[i] -= direction * err
            forces[j] += direction * err
        disp = 0.25 * forces
        scale = float(np.linalg.norm(disp))
        if scale > 0.5:
            disp *= 0.5 / scale
        coords += disp
    return coords


def _separate_clashes(coords: np.ndarray, min_dist: float = 0.35) -> None:
    n = len(coords)
    for _ in range(3):
        for i in range(n):
            for j in range(i + 1, n):
                vec = coords[j] - coords[i]
                dist = float(np.linalg.norm(vec))
                if dist >= min_dist:
                    continue
                if dist < 1e-8:
                    direction = np.array([1.0, 0.0, 0.0])
                else:
                    direction = vec / dist
                shift = 0.5 * (min_dist - dist) * direction
                coords[i] -= shift
                coords[j] += shift


def _component_gap(
    params: Sequence[AtomParams], comp_a: Sequence[int], comp_b: Sequence[int]
) -> float:
    x_a = max(params[i].x for i in comp_a)
    x_b = max(params[i].x for i in comp_b)
    raw = 0.8 * math.sqrt(max(x_a * x_b, 1e-8))
    return float(min(4.5, max(2.0, raw)))


def _embed_component(
    indices: Sequence[int],
    graph: MolGraph,
    params: Sequence[AtomParams],
    rng: np.random.Generator,
    *,
    deterministic: bool,
) -> np.ndarray:
    m = len(indices)
    if m == 1:
        return np.zeros((1, 3), dtype=float)
    lower, upper = distance_bounds(graph, params, indices)
    dist = _sample_distances(lower, upper, rng, deterministic=deterministic)
    try:
        coords = _classical_mds(dist)
        if not np.isfinite(coords).all():
            raise ValueError("non-finite MDS coordinates")
    except (np.linalg.LinAlgError, ValueError):
        coords = _line_fallback(indices, graph, params)

    g2l = {global_i: local_i for local_i, global_i in enumerate(indices)}
    targets: list[tuple[int, int, float]] = []
    for i, j, order in graph.bonds:
        if i in g2l and j in g2l:
            targets.append((g2l[i], g2l[j], bond_length(params[i], params[j], order)))
    coords = _polish(coords, targets)
    _separate_clashes(coords)
    return coords


def _pack(
    graph: MolGraph,
    params: Sequence[AtomParams],
    pieces: list[tuple[list[int], np.ndarray]],
) -> np.ndarray:
    coords = np.zeros((graph.n_atoms, 3), dtype=float)
    cursor = 0.0
    for comp_i, (indices, local) in enumerate(pieces):
        placed = np.array(local, dtype=float, copy=True)
        placed -= placed.mean(axis=0)
        rad = float(np.linalg.norm(placed, axis=1).max()) if len(indices) else 0.0
        placed[:, 0] += cursor + rad
        for local_i, global_i in enumerate(indices):
            coords[global_i] = placed[local_i]
        if comp_i + 1 < len(pieces):
            gap = _component_gap(params, indices, pieces[comp_i + 1][0])
        else:
            gap = 0.0
        cursor += 2.0 * rad + gap
    coords -= coords.mean(axis=0)
    return coords


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _dihedral(coords: np.ndarray, i: int, j: int, k: int, atom_l: int) -> float:
    b0 = coords[j] - coords[i]
    b1 = coords[k] - coords[j]
    b2 = coords[atom_l] - coords[k]
    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2)
    n1_norm = float(np.linalg.norm(n1))
    n2_norm = float(np.linalg.norm(n2))
    b1_norm = float(np.linalg.norm(b1))
    if n1_norm < 1e-8 or n2_norm < 1e-8 or b1_norm < 1e-8:
        return 0.0
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    m1 = np.cross(n1, b1 / b1_norm)
    return math.atan2(float(np.dot(m1, n2)), float(np.dot(n1, n2)))


def _angle(coords: np.ndarray, i: int, j: int, k: int) -> float:
    v1 = coords[i] - coords[j]
    v2 = coords[k] - coords[j]
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_t = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.acos(cos_t)


def _rotate(
    coords: np.ndarray,
    indices: Sequence[int],
    origin: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> None:
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8 or abs(angle) < 1e-8:
        return
    unit = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    for idx in indices:
        v = coords[idx] - origin
        coords[idx] = (
            origin + v * c + np.cross(unit, v) * s + unit * float(np.dot(unit, v)) * (1.0 - c)
        )


def _is_bridge(neigh: list[list[int]], a: int, b: int) -> bool:
    seen = {a}
    stack = [nb for nb in neigh[a] if nb != b]
    while stack:
        node = stack.pop()
        if node == b:
            return False
        if node in seen:
            continue
        seen.add(node)
        for nb in neigh[node]:
            if nb not in seen:
                stack.append(nb)
    return True


def _side(neigh: list[list[int]], origin: int, blocked: int) -> list[int]:
    seen = {blocked}
    stack = [origin]
    out: list[int] = []
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        out.append(node)
        for nb in neigh[node]:
            if nb not in seen:
                stack.append(nb)
    return out


def _enforce_ez(coords: np.ndarray, graph: MolGraph) -> None:
    if not graph.ez:
        return
    neigh = graph.neighbor_lists()
    for spec in graph.ez:
        if not _is_bridge(neigh, spec.anchor_a, spec.anchor_b):
            continue
        phi = _dihedral(coords, spec.ligand_a, spec.anchor_a, spec.anchor_b, spec.ligand_b)
        target = 0.0 if spec.kind == "cis" else math.pi
        delta = _wrap(target - phi)
        if abs(delta) < 1e-3:
            continue
        probe = np.array(coords, copy=True)
        axis = coords[spec.anchor_b] - coords[spec.anchor_a]
        atoms = _side(neigh, spec.anchor_b, spec.anchor_a)
        _rotate(probe, atoms, coords[spec.anchor_a], axis, 0.02)
        phi2 = _dihedral(probe, spec.ligand_a, spec.anchor_a, spec.anchor_b, spec.ligand_b)
        direction = _wrap(phi2 - phi) / 0.02
        if abs(direction) < 1e-6:
            continue
        _rotate(coords, atoms, coords[spec.anchor_a], axis, delta / direction)


def _chiral_volume(coords: np.ndarray, center: int, a: int, b: int, c: int) -> float:
    v1 = coords[a] - coords[center]
    v2 = coords[b] - coords[center]
    v3 = coords[c] - coords[center]
    return float(np.dot(v1, np.cross(v2, v3)))


def _chiral_violations(coords: np.ndarray, graph: MolGraph) -> int:
    bad = 0
    for spec in graph.chiral:
        volume = _chiral_volume(
            coords,
            spec.center,
            spec.neighbors[0],
            spec.neighbors[1],
            spec.neighbors[2],
        )
        if volume <= 0.02:
            bad += 1
    return bad


def _fix_chirality(coords: np.ndarray, graph: MolGraph) -> None:
    if not graph.chiral:
        return
    bad = _chiral_violations(coords, graph)
    if bad == 0:
        return
    flipped = np.array(coords, copy=True)
    flipped[:, 0] *= -1.0
    if _chiral_violations(flipped, graph) < bad:
        coords[:, 0] *= -1.0


def _ez_penalty(coords: np.ndarray, graph: MolGraph) -> float:
    penalty = 0.0
    for spec in graph.ez:
        phi = _dihedral(coords, spec.ligand_a, spec.anchor_a, spec.anchor_b, spec.ligand_b)
        target = 0.0 if spec.kind == "cis" else math.pi
        penalty += (_wrap(phi - target) / math.pi) ** 2
    return penalty


def geometry_score(coords: np.ndarray, graph: MolGraph, params: Sequence[AtomParams]) -> float:
    """Bond, angle, clash, and stereo penalty used to rank embeddings. Lower is better."""
    score = 0.0
    for i, j, order in graph.bonds:
        r0 = bond_length(params[i], params[j], order)
        dist = float(np.linalg.norm(coords[j] - coords[i]))
        score += ((dist - r0) / r0) ** 2
        if dist < 0.4:
            score += 50.0
    for i, j, k in graph.angle_indices():
        theta = _angle(coords, i, j, k)
        target = preferred_angle(theta, math.radians(params[j].theta), params[j].cn)
        score += ((theta - target) / math.pi) ** 2
    if graph.n_atoms >= 2:
        delta = coords[:, None, :] - coords[None, :, :]
        dmat = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(dmat, np.inf)
        min_d = float(dmat.min())
        if min_d < 0.5:
            score += (0.5 - min_d) * 100.0
    score += 8.0 * _chiral_violations(coords, graph)
    score += 3.0 * _ez_penalty(coords, graph)
    return float(score)


def embed_coordinates(
    graph: MolGraph,
    params: Sequence[AtomParams],
    rng: np.random.Generator,
    *,
    deterministic: bool,
) -> np.ndarray:
    """Embed every component, clean up, then enforce annotated stereo."""
    if graph.n_atoms == 0:
        raise ValueError("Cannot embed an empty molecule")
    pieces: list[tuple[list[int], np.ndarray]] = []
    for component in graph.connected_components():
        local = _embed_component(component, graph, params, rng, deterministic=deterministic)
        pieces.append((component, local))
    coords = _pack(graph, params, pieces)
    _separate_clashes(coords)
    coords = relax_coordinates(coords, graph, params)
    if not np.isfinite(coords).all():
        coords = _pack(graph, params, pieces)
    _enforce_ez(coords, graph)
    _fix_chirality(coords, graph)
    coords -= coords.mean(axis=0)
    return coords
