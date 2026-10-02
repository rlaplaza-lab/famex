"""Dewar–Healy–Stewart (DHS) two-image bracket TS strategy."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from ase import Atoms

from famex.core.base_strategy import BaseStrategy, StrategyMetadata
from famex.core.registry import REGISTRY
from famex.io.path_manager import PathManager
from famex.strategies.helpers import validate_ts_structure
from famex.strategies.ts import LocalTSStrategy
from famex.strategies.utils import StrategyUtils
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)


def _euclidean_distance(a: Atoms, b: Atoms) -> float:
    return float(np.linalg.norm(a.positions - b.positions))


def _project_orthogonal_to_vector(force: np.ndarray, vector: np.ndarray) -> np.ndarray:
    vec = vector.ravel()
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.asarray(force, dtype=float)
    unit = vec / norm
    f = np.asarray(force, dtype=float).ravel()
    return (f - float(np.dot(f, unit)) * unit).reshape(force.shape)


def distance_constrained_minimize(
    atoms: Atoms,
    pivot: Atoms,
    *,
    fmax: float = 0.05,
    max_steps: int = 50,
    trust_radius: float = 0.1,
) -> bool:
    """Minimize ``atoms`` at fixed Euclidean distance from ``pivot``."""
    target_dist = _euclidean_distance(atoms, pivot)
    if target_dist < 1e-8:
        return True

    pivot_pos = pivot.positions.copy()
    for _ in range(max_steps):
        tangential = _project_orthogonal_to_vector(
            np.asarray(atoms.get_forces(), dtype=float),
            atoms.positions - pivot_pos,
        )
        if float(np.max(np.abs(tangential))) < fmax:
            return True

        step = tangential
        step_norm = float(np.linalg.norm(step))
        if step_norm > trust_radius:
            step = step * (trust_radius / step_norm)

        new_pos = atoms.positions + step
        radial = new_pos - pivot_pos
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm > 1e-12:
            new_pos = pivot_pos + radial * (target_dist / radial_norm)
        atoms.positions = new_pos

    return False


class MultiStructureTSDHSStrategy(BaseStrategy):
    """Dewar–Healy–Stewart bracket method for TS search."""

    metadata = StrategyMetadata(
        name="ts:dhs",
        target="ts",
        strategy="dhs",
        description="Dewar-Healy-Stewart two-image bracket TS search",
        aliases=["dhs"],
        requires_multiple_structures=True,
    )

    def run(
        self,
        atoms_list: list[Atoms],
        fmax: float = 0.05,
        steps: int = 200,
        validate_ts: bool = False,
        calculate_frequencies: bool = False,
        dist_tol: float = 0.5,
        large_step: float = 0.2,
        small_step: float = 0.05,
        switch_thresh: float = 1.5,
        constrain_fmax: float = 0.05,
        constrain_steps: int = 50,
        trust_radius: float = 0.1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.validate_inputs(atoms_list)
        if len(atoms_list) != 2:
            msg = f"DHS requires exactly 2 structures, got {len(atoms_list)}"
            raise ValueError(msg)

        left = atoms_list[0].copy()
        right = atoms_list[1].copy()
        if self.explorer is not None:
            StrategyUtils.ensure_charge_spin_info(left)
            StrategyUtils.ensure_charge_spin_info(right)
            PathManager.attach_calculators(self.explorer, [left, right])

        logger.info(
            "DHS start: dist=%.4f Å",
            _euclidean_distance(left, right),
        )

        max_macro = int(kwargs.get("max_macro_steps", steps))
        converged_bracket = False

        for macro in range(max_macro):
            dist = _euclidean_distance(left, right)
            if dist <= dist_tol:
                converged_bracket = True
                logger.info("DHS converged at distance %.4f Å", dist)
                break

            step_size = large_step if dist > switch_thresh else small_step
            left_energy = float(left.get_potential_energy())
            right_energy = float(right.get_potential_energy())
            moving, pivot = (left, right) if left_energy <= right_energy else (right, left)

            direction = pivot.positions - moving.positions
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm < 1e-12:
                converged_bracket = True
                break
            moving.positions = moving.positions + direction * (step_size / direction_norm)

            distance_constrained_minimize(
                moving,
                pivot,
                fmax=constrain_fmax,
                max_steps=constrain_steps,
                trust_radius=min(step_size, trust_radius),
            )
            logger.info(
                "DHS macro %d: dist=%.4f Å",
                macro + 1,
                _euclidean_distance(left, right),
            )

        left_energy = float(left.get_potential_energy())
        right_energy = float(right.get_potential_energy())
        ts_guess = left if left_energy >= right_energy else right

        local_optimizer_name = kwargs.get("local_optimizer_name", "rfo")
        ts_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in {"local_optimizer_name", "max_macro_steps", "ts_refinement_steps"}
        }
        ts_result = LocalTSStrategy(self.explorer).run(
            [ts_guess],
            fmax=fmax,
            steps=kwargs.get("ts_refinement_steps", 500),
            local_optimizer_name=local_optimizer_name,
            calculate_frequencies=calculate_frequencies,
            **ts_kwargs,
        )

        optimized_atoms = cast(Atoms | list[Atoms], ts_result["optimized_atoms"])
        result = self.prepare_result(
            optimized_atoms,
            steps_taken=ts_result.get("steps_taken", 0),
            converged=bool(ts_result.get("converged", False)) and converged_bracket,
            dhs_distance=_euclidean_distance(left, right),
            dhs_converged=converged_bracket,
        )
        if validate_ts and isinstance(optimized_atoms, Atoms):
            result["ts_validation"] = validate_ts_structure(optimized_atoms, self.explorer)
        if "frequency_analysis" in ts_result:
            result["frequency_analysis"] = ts_result["frequency_analysis"]
            result["is_ts"] = ts_result.get("is_ts")
            result["free_energy_correction"] = ts_result.get("free_energy_correction")
        return result


REGISTRY.register(MultiStructureTSDHSStrategy)
