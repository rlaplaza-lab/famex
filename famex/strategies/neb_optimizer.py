"""NEB / CI-NEB optimizer with ASE improved-tangent forces.

Force law matches ``ase.mep.NEB(..., method='improvedtangent')``.
CI-NEB uses ORCA energy-weighted spring constants; interior images step with FIRE.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from ase import Atoms

from famex.strategies.utils import StrategyUtils
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)


def energy_weighted_spring_constants(
    energies: Sequence[float],
    k_u: float,
    k_l: float | None = None,
) -> np.ndarray:
    """ORCA energy-weighted spring constants (length ``nimages - 1``)."""
    energies_arr = np.asarray(energies, dtype=float)
    n_springs = len(energies_arr) - 1
    if n_springs < 1:
        return np.array([], dtype=float)

    if k_l is None:
        k_l = 0.1 * k_u

    e_max = float(np.max(energies_arr))
    e_ref = float(max(energies_arr[0], energies_arr[-1]))
    denom = e_max - e_ref
    e_pair = np.maximum(energies_arr[:-1], energies_arr[1:])

    k_values = np.full(n_springs, float(k_l), dtype=float)
    if denom > 1e-12:
        above = e_pair > e_ref
        alpha = (e_max - e_pair[above]) / denom
        k_values[above] = (1.0 - alpha) * k_u + alpha * k_l
    return k_values


def improved_tangent(
    positions: Sequence[np.ndarray],
    energies: Sequence[float],
    i: int,
) -> np.ndarray:
    """Return normalized improved NEB tangent at interior image ``i`` (ASE)."""
    energies_arr = np.asarray(energies, dtype=float)
    spring1 = np.asarray(positions[i] - positions[i - 1], dtype=float).ravel()
    spring2 = np.asarray(positions[i + 1] - positions[i], dtype=float).ravel()

    e_im1, e_i, e_ip1 = energies_arr[i - 1], energies_arr[i], energies_arr[i + 1]

    if e_ip1 > e_i > e_im1:
        tangent = spring2.copy()
    elif e_ip1 < e_i < e_im1:
        tangent = spring1.copy()
    else:
        deltavmax = max(abs(e_ip1 - e_i), abs(e_im1 - e_i))
        deltavmin = min(abs(e_ip1 - e_i), abs(e_im1 - e_i))
        if e_ip1 > e_im1:
            tangent = spring2 * deltavmax + spring1 * deltavmin
        else:
            tangent = spring2 * deltavmin + spring1 * deltavmax

    norm = np.linalg.norm(tangent)
    if norm < 1e-12:
        return np.zeros_like(tangent)
    return cast(np.ndarray, tangent / norm)


def compute_neb_forces(
    positions: Sequence[np.ndarray],
    energies: Sequence[float],
    true_forces: Sequence[np.ndarray],
    k: float | Sequence[float] | np.ndarray,
    climb: bool = False,
) -> tuple[list[np.ndarray], int | None]:
    """Improved-tangent NEB forces; endpoints are zero (fixed).

    Matches ASE ``NEB(..., method='improvedtangent').get_forces()`` for the
    interior images. ``k`` is a scalar or one value per spring.
    """
    nimages = len(positions)
    if nimages < 3:
        msg = "NEB requires at least 3 images"
        raise ValueError(msg)

    if isinstance(k, (float, int)):
        k_arr = np.full(nimages - 1, float(k), dtype=float)
    else:
        k_arr = np.asarray(k, dtype=float).reshape(-1)
        if len(k_arr) != nimages - 1:
            msg = f"Expected {nimages - 1} spring constants, got {len(k_arr)}"
            raise ValueError(msg)

    spring_lengths = [
        float(np.linalg.norm(np.asarray(positions[i + 1] - positions[i], dtype=float)))
        for i in range(nimages - 1)
    ]

    climbing_image: int | None = None
    if climb:
        climbing_image = 1 + int(np.argmax(np.asarray(energies[1:-1], dtype=float)))

    neb_forces: list[np.ndarray] = [
        np.zeros_like(np.asarray(true_forces[0], dtype=float)),
    ]

    for i in range(1, nimages - 1):
        force = np.asarray(true_forces[i], dtype=float).copy()
        force_flat = np.asarray(force.ravel(), dtype=float)
        tangent = improved_tangent(positions, energies, i)
        tangential_force = float(np.vdot(force_flat, tangent))

        if climb and i == climbing_image:
            updated = force_flat - 2.0 * tangential_force * tangent
        else:
            updated = force_flat - tangential_force * tangent
            spring_force_mag = k_arr[i] * spring_lengths[i] - k_arr[i - 1] * spring_lengths[i - 1]
            updated = updated + spring_force_mag * tangent

        neb_forces.append(np.asarray(updated, dtype=float).reshape(force.shape))

    neb_forces.append(np.zeros_like(np.asarray(true_forces[-1], dtype=float)))
    return neb_forces, climbing_image


class _FIREBandStepper:
    """FIRE on concatenated interior-image coordinates (ASE FIRE defaults)."""

    def __init__(
        self,
        *,
        dt: float = 0.1,
        maxstep: float = 0.2,
        dtmax: float = 1.0,
        nmin: int = 5,
        finc: float = 1.1,
        fdec: float = 0.5,
        astart: float = 0.1,
        fa: float = 0.99,
    ) -> None:
        self.dt = dt
        self.maxstep = maxstep
        self.dtmax = dtmax
        self.nmin = nmin
        self.finc = finc
        self.fdec = fdec
        self.astart = astart
        self.fa = fa
        self.a = astart
        self.nsteps = 0
        self.v: np.ndarray | None = None

    def step(self, positions: np.ndarray, forces: np.ndarray) -> np.ndarray:
        f = np.asarray(forces, dtype=float).reshape(-1, 3)
        r = np.asarray(positions, dtype=float).reshape(-1, 3)

        if self.v is None:
            self.v = np.zeros_like(f)
        else:
            vf = float(np.vdot(f, self.v))
            if vf > 0.0:
                v_norm = np.sqrt(float(np.vdot(self.v, self.v)))
                f_norm = np.sqrt(float(np.vdot(f, f)))
                if f_norm > 1e-16 and v_norm > 1e-16:
                    self.v = (1.0 - self.a) * self.v + self.a * f / f_norm * v_norm
                if self.nsteps > self.nmin:
                    self.dt = min(self.dt * self.finc, self.dtmax)
                    self.a *= self.fa
                self.nsteps += 1
            else:
                self.v[:] = 0.0
                self.a = self.astart
                self.dt *= self.fdec
                self.nsteps = 0

        self.v = self.v + self.dt * f
        dr = self.dt * self.v
        normdr = float(np.sqrt(np.vdot(dr, dr)))
        if normdr > self.maxstep:
            dr = self.maxstep * dr / normdr
        return (r + dr).ravel()


class NEBOptimizer:
    """NEB / CI-NEB with ASE improved-tangent forces and FIRE steps."""

    def __init__(
        self,
        images: Sequence[Atoms],
        spring_constant: float = 5.0,
        climb: bool = False,
        fmax: float = 0.05,
        steps: int = 1000,
        spring_constant_lower: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.images: list[Atoms] = []
        for atoms in images:
            copied = atoms.copy()
            if getattr(atoms, "calc", None) is not None:
                copied.calc = atoms.calc
            self.images.append(copied)

        self.spring_constant = float(spring_constant)
        self.spring_constant_lower = (
            float(spring_constant_lower)
            if spring_constant_lower is not None
            else 0.1 * self.spring_constant
        )
        self.climb = climb
        self.fmax = fmax
        self.steps = steps
        self.climbing_image: int | None = None

        for atoms in self.images:
            StrategyUtils.ensure_charge_spin_info(atoms)

        calculator = self.images[0].calc if self.images[0].calc is not None else None
        self.supports_batch = StrategyUtils.check_batch_support(calculator)

        self._fire = _FIREBandStepper(
            dt=float(kwargs.get("dt", 0.1)),
            maxstep=float(kwargs.get("maxstep", 0.2)),
            dtmax=float(kwargs.get("dtmax", 1.0)),
            nmin=int(kwargs.get("Nmin", 5)),
            finc=float(kwargs.get("finc", 1.1)),
            fdec=float(kwargs.get("fdec", 0.5)),
            astart=float(kwargs.get("astart", 0.1)),
            fa=float(kwargs.get("fa", 0.99)),
        )

        method_name = "CI-NEB" if self.climb else "NEB"
        if self.supports_batch:
            logger.info(
                "Using batch evaluation for %s optimization with %d images",
                method_name,
                len(self.images),
            )
        else:
            logger.info("Starting %s optimization with %d images", method_name, len(self.images))

    def get_spring_constants(self, energies: Sequence[float]) -> np.ndarray:
        if self.climb:
            return energy_weighted_spring_constants(
                energies,
                k_u=self.spring_constant,
                k_l=self.spring_constant_lower,
            )
        return np.full(len(self.images) - 1, self.spring_constant, dtype=float)

    def compute_forces(
        self,
        energies: Sequence[float],
        true_forces: Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        positions = [img.positions.copy() for img in self.images]
        neb_forces, climbing_image = compute_neb_forces(
            positions,
            energies,
            true_forces,
            k=self.get_spring_constants(energies),
            climb=self.climb,
        )
        self.climbing_image = climbing_image
        return neb_forces

    def optimize(self) -> list[Atoms]:
        method_name = "CI-NEB" if self.climb else "NEB"
        converged = False

        for step in range(self.steps):
            energies, forces_list = self._calculate_energies_forces()
            neb_forces = self.compute_forces(energies, forces_list)

            interior_forces = neb_forces[1:-1]
            if not interior_forces:
                break

            max_force = max(float(np.max(np.abs(force))) for force in interior_forces)
            if max_force < self.fmax:
                logger.info(
                    "%s converged after %d steps (max force: %.6f)",
                    method_name,
                    step + 1,
                    max_force,
                )
                if self.climb and self.climbing_image is not None:
                    logger.info("Climbing image was image %d", self.climbing_image)
                converged = True
                break

            self._fire_update(neb_forces)

        if not converged:
            logger.warning("%s did not converge within %d steps", method_name, self.steps)
        return self.images

    def _calculate_energies_forces(self) -> tuple[list[float], list[np.ndarray]]:
        calculator = self.images[0].calc if self.images[0].calc is not None else None
        return StrategyUtils.calculate_batch_energies_forces(
            self.images,
            calculator,
            self.supports_batch,
        )

    def _fire_update(self, neb_forces: list[np.ndarray]) -> None:
        if len(self.images) < 3:
            return

        interior_pos = np.concatenate([img.positions.ravel() for img in self.images[1:-1]])
        interior_f = np.concatenate([f.ravel() for f in neb_forces[1:-1]])
        new_pos = self._fire.step(interior_pos, interior_f)

        natoms = len(self.images[0])
        ncoords = 3 * natoms
        for idx, img in enumerate(self.images[1:-1]):
            start = idx * ncoords
            img.positions = new_pos[start : start + ncoords].reshape(natoms, 3)
