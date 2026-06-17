"""Post-optimization frequency cleanup using Hessian-based optimizers."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, Literal, cast

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

from famex.optimizers.sella_utils import is_sella_optimizer
from famex.strategies.helpers import _get_local_optimizer_class
from famex.strategies.utils import StrategyUtils
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)

StationaryPointTarget = Literal["minima", "ts"]

# Frequencies below this magnitude (cm^-1) are treated as imaginary; mirrors the
# default threshold used by FrequencyAnalysis.is_minima / is_transition_state.
_IMAGINARY_THRESHOLD_CM = 50.0
# Base geometric perturbation amplitude (Å) applied before a retry. The amplitude
# grows with each subsequent attempt to escape stubborn stationary points.
_BASE_PERTURBATION_AMPLITUDE = 0.1


def default_cleanup_optimizer_name(target: StationaryPointTarget) -> str:
    if target == "ts":
        return "rfo"
    return "trust-krylov"


def relevant_imaginary_count(
    frequency_result: dict[str, Any],
    target: StationaryPointTarget,
) -> int:
    if target == "minima":
        analysis = frequency_result.get("minima_analysis") or {}
        return int(analysis.get("n_significant_imaginary_frequencies", 0))
    analysis = frequency_result.get("ts_analysis") or {}
    return int(analysis.get("n_imaginary_frequencies", 0))


def stationary_point_is_clean(
    frequency_result: dict[str, Any],
    target: StationaryPointTarget,
) -> bool:
    if target == "minima":
        return bool(frequency_result.get("is_minimum", False))
    return bool(frequency_result.get("is_ts", False))


def target_imaginary_count(target: StationaryPointTarget) -> int:
    return 1 if target == "ts" else 0


def prepare_minima_hessian_optimizer_kwargs(optimizer_name: str, explorer: Any) -> dict[str, Any]:
    opt_kwargs = dict(getattr(explorer, "optimizer_kwargs", {}) or {})
    normalized_name = optimizer_name.lower()

    if is_sella_optimizer(normalized_name):
        opt_kwargs.setdefault("internal", True)
        opt_kwargs.setdefault("order", 0)
    elif normalized_name in (
        "trust-krylov",
        "trustkrylov",
        "trust_krylov",
        "trust-ncg",
        "trustncg",
        "trust_ncg",
        "trust-exact",
        "trustexact",
        "trust_exact",
        "newton-cg",
        "newtoncg",
        "newton_cg",
    ):
        opt_kwargs.setdefault("hessian_method", "auto")
        opt_kwargs.setdefault("hessian_update_freq", 1)
        opt_kwargs.setdefault("use_bfgs_update", False)
        if normalized_name in (
            "trust-krylov",
            "trustkrylov",
            "trust_krylov",
            "trust-ncg",
            "trustncg",
            "trust_ncg",
            "trust-exact",
            "trustexact",
            "trust_exact",
        ):
            opt_kwargs.setdefault("trust_radius", 0.1)
            opt_kwargs.setdefault("max_trust_radius", 0.3)

    if getattr(explorer, "force_finite_diff_hessian", False):
        if normalized_name in (
            "trust-krylov",
            "trustkrylov",
            "trust_krylov",
            "trust-ncg",
            "trustncg",
            "trust_ncg",
            "trust-exact",
            "trustexact",
            "trust_exact",
            "newton-cg",
            "newtoncg",
            "newton_cg",
        ):
            opt_kwargs["hessian_method"] = "finite_differences"

    return opt_kwargs


def _resolve_cleanup_settings(
    explorer: Any,
    target: StationaryPointTarget,
) -> tuple[str, int, int]:
    kwargs_source = getattr(explorer, "ts_kwargs", {}) or {}
    if target == "minima":
        kwargs_source = {
            **kwargs_source,
            **(getattr(explorer, "optimizer_kwargs", {}) or {}),
        }

    cleanup_optimizer = kwargs_source.get("cleanup_frequency_optimizer")
    if cleanup_optimizer is None:
        cleanup_optimizer = default_cleanup_optimizer_name(target)

    max_attempts = int(kwargs_source.get("cleanup_frequency_max_attempts", 3))
    cleanup_steps = int(kwargs_source.get("cleanup_frequency_steps", 200))
    return str(cleanup_optimizer), max(1, max_attempts), max(1, cleanup_steps)


def _fixed_atom_indices(atoms: Atoms) -> np.ndarray:
    """Atom indices held fixed by any FixAtoms constraint on ``atoms``."""
    indices: list[int] = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            indices.extend(int(index) for index in constraint.get_indices())
    return np.array(sorted(set(indices)), dtype=int)


def _zero_fixed_displacements(displacement: np.ndarray, fixed_indices: np.ndarray) -> None:
    if fixed_indices.size:
        displacement[fixed_indices] = 0.0


def _apply_constrained_displacement(atoms: Atoms, displacement: np.ndarray) -> None:
    """Apply a Cartesian displacement while honoring all ASE position constraints."""
    displacement = np.asarray(displacement, dtype=float).copy()
    if displacement.shape != atoms.positions.shape:
        msg = (
            f"Displacement shape {displacement.shape} does not match "
            f"atomic positions {atoms.positions.shape}"
        )
        raise ValueError(msg)

    _zero_fixed_displacements(displacement, _fixed_atom_indices(atoms))

    new_positions = atoms.positions + displacement
    for constraint in atoms.constraints:
        adjust_positions = getattr(constraint, "adjust_positions", None)
        if adjust_positions is not None:
            adjust_positions(atoms, new_positions)

    atoms.set_positions(new_positions)


def _mode_atom_indices(
    n_atoms: int,
    normal_modes: np.ndarray,
    frequency_result: dict[str, Any],
) -> list[int] | None:
    """Map normal-mode rows to atom indices, including partial Hessian analyses."""
    if normal_modes.shape[0] == 3 * n_atoms:
        return list(range(n_atoms))

    indices = frequency_result.get("indices")
    if indices is not None and normal_modes.shape[0] == 3 * len(indices):
        return [int(index) for index in indices]

    return None


def _select_modes_to_follow(
    frequencies: np.ndarray,
    target: StationaryPointTarget,
) -> list[int]:
    """Return column indices of normal modes to displace along for the given target."""
    imaginary_idx = list(np.where(frequencies < -_IMAGINARY_THRESHOLD_CM)[0])

    if target == "minima":
        # Every imaginary mode is spurious; displacing along it slides off the saddle.
        return imaginary_idx

    if len(imaginary_idx) >= 2:
        # Keep the most negative mode (reaction coordinate); remove the extras.
        reaction_global = int(imaginary_idx[int(frequencies[imaginary_idx].argmin())])
        return [idx for idx in imaginary_idx if idx != reaction_global]

    if len(imaginary_idx) == 0:
        # Collapsed to a minimum: follow the softest real mode to regain curvature.
        real_idx = np.where(frequencies > _IMAGINARY_THRESHOLD_CM)[0]
        if real_idx.size == 0:
            return []
        return [int(real_idx[int(frequencies[real_idx].argmin())])]

    # Exactly one imaginary mode means the TS is already clean.
    return []


def _perturb_along_modes(
    atoms: Atoms,
    frequency_result: dict[str, Any],
    target: StationaryPointTarget,
    amplitude: float,
    rng: np.random.Generator,
) -> bool:
    """Displace along selected normal modes. Returns True if a displacement was applied."""
    frequencies = np.asarray(frequency_result.get("frequencies", []), dtype=float)
    normal_modes = np.asarray(frequency_result.get("normal_modes", []), dtype=float)

    if frequencies.ndim != 1 or frequencies.size == 0 or normal_modes.ndim != 2:
        return False
    if normal_modes.shape[1] != frequencies.shape[0]:
        return False

    atom_indices = _mode_atom_indices(len(atoms), normal_modes, frequency_result)
    if atom_indices is None:
        return False

    modes_to_follow = _select_modes_to_follow(frequencies, target)
    if not modes_to_follow:
        return False

    n_mode_atoms = len(atom_indices)
    displacement = np.zeros((len(atoms), 3))
    for col in modes_to_follow:
        mode_vec = normal_modes[:, col].reshape(n_mode_atoms, 3)
        norm = float(np.linalg.norm(mode_vec))
        if norm < 1e-12:
            continue
        sign = 1.0 if rng.random() < 0.5 else -1.0
        scaled_mode = sign * amplitude * (mode_vec / norm)
        for local_index, atom_index in enumerate(atom_indices):
            displacement[atom_index] += scaled_mode[local_index]

    if not np.any(displacement):
        return False

    _apply_constrained_displacement(atoms, displacement)
    return True


def _perturb_random(atoms: Atoms, amplitude: float, rng: np.random.Generator) -> None:
    """Apply a small random Cartesian kick to escape a stationary point."""
    displacement = rng.normal(scale=amplitude, size=atoms.positions.shape)
    _apply_constrained_displacement(atoms, displacement)


def _apply_perturbation(
    atoms: Atoms,
    frequency_result: dict[str, Any],
    target: StationaryPointTarget,
    amplitude: float,
    rng: np.random.Generator,
) -> str:
    """Perturb the geometry, preferring mode-following with a random fallback."""
    if _perturb_along_modes(atoms, frequency_result, target, amplitude, rng):
        return "mode"
    _perturb_random(atoms, amplitude, rng)
    return "random"


def run_frequency_cleanup(
    atoms: Atoms,
    explorer: Any,
    *,
    target: StationaryPointTarget,
    fmax: float,
    temperature: float,
    verbose: int,
    prepare_optimizer_kwargs: Callable[[str, Any], dict[str, Any]],
    prepare_frequency_kwargs: Callable[[Atoms], dict[str, Any]] | None,
    profiler: Any | None = None,
) -> tuple[Atoms, dict[str, Any], int, dict[str, Any]]:
    """Refine a structure in place until imaginary-mode count matches the target.

    The first attempt optimizes from the post-optimization geometry. Because a
    Hessian-aware optimizer sitting on a stationary point will not move, each
    subsequent attempt first perturbs the geometry: it displaces along the
    spurious imaginary mode(s) when they can be identified, otherwise it applies
    a random Cartesian kick. The perturbation amplitude grows with each attempt.
    """
    cleanup_optimizer, max_attempts, cleanup_steps = _resolve_cleanup_settings(explorer, target)
    desired_count = target_imaginary_count(target)

    freq_kwargs: dict[str, Any] = {
        "atoms": atoms,
        "temperature": temperature,
        "save_hessian": False,
    }
    if prepare_frequency_kwargs is not None:
        freq_kwargs.update(prepare_frequency_kwargs(atoms))

    frequency_result = cast(dict[str, Any], explorer.calculate_frequencies(**freq_kwargs))
    total_cleanup_steps = 0
    attempts: list[dict[str, Any]] = []
    rng = np.random.default_rng()

    if stationary_point_is_clean(frequency_result, target):
        if verbose >= 1:
            logger.info(
                "Frequency cleanup skipped: structure already has %d relevant imaginary mode(s)",
                relevant_imaginary_count(frequency_result, target),
            )
        return atoms, frequency_result, 0, {"attempts": attempts, "skipped": True}

    if verbose >= 1:
        logger.info(
            "Starting frequency cleanup from current geometry with %s "
            "(found %d relevant imaginary mode(s), target %d)",
            cleanup_optimizer,
            relevant_imaginary_count(frequency_result, target),
            desired_count,
        )

    for attempt_idx in range(1, max_attempts + 1):
        n_imaginary_before = relevant_imaginary_count(frequency_result, target)
        if stationary_point_is_clean(frequency_result, target):
            break

        # The optimizer cannot escape an already-converged stationary point on its
        # own, so perturb the geometry before every retry after the first.
        perturbation = None
        if attempt_idx > 1:
            amplitude = _BASE_PERTURBATION_AMPLITUDE * (attempt_idx - 1)
            perturbation = _apply_perturbation(atoms, frequency_result, target, amplitude, rng)
            if verbose >= 1:
                logger.info(
                    "Frequency cleanup: applied %s perturbation (amplitude %.3f Å) before retry %d",
                    perturbation,
                    amplitude,
                    attempt_idx,
                )

        opt_class = _get_local_optimizer_class(cleanup_optimizer)
        opt_kwargs = prepare_optimizer_kwargs(cleanup_optimizer, explorer)
        opt_kwargs.setdefault("verbose", verbose)
        if profiler is not None:
            opt_kwargs["profiler"] = profiler

        opt = opt_class(atoms, **opt_kwargs)
        with (
            profiler.profile_section("frequency_cleanup") if profiler is not None else nullcontext()
        ):
            opt.run(fmax=fmax, steps=cleanup_steps)

        steps_taken = StrategyUtils.get_step_count(opt) or 0
        total_cleanup_steps += steps_taken

        freq_kwargs["atoms"] = atoms
        frequency_result = cast(dict[str, Any], explorer.calculate_frequencies(**freq_kwargs))
        n_imaginary_after = relevant_imaginary_count(frequency_result, target)
        attempts.append(
            {
                "attempt": attempt_idx,
                "optimizer": cleanup_optimizer,
                "perturbation": perturbation,
                "steps": steps_taken,
                "n_imaginary_before": n_imaginary_before,
                "n_imaginary_after": n_imaginary_after,
                "is_clean": stationary_point_is_clean(frequency_result, target),
            }
        )

        if verbose >= 1:
            logger.info(
                "Frequency cleanup attempt %d/%d: %d -> %d relevant imaginary mode(s)",
                attempt_idx,
                max_attempts,
                n_imaginary_before,
                n_imaginary_after,
            )

        if stationary_point_is_clean(frequency_result, target):
            if verbose >= 1:
                logger.info("Frequency cleanup succeeded after %d attempt(s)", attempt_idx)
            break
    else:
        if verbose >= 1:
            logger.warning(
                "Frequency cleanup finished without reaching target "
                "(%d relevant imaginary mode(s), wanted %d)",
                relevant_imaginary_count(frequency_result, target),
                desired_count,
            )

    cleanup_summary = {
        "attempts": attempts,
        "skipped": False,
        "optimizer": cleanup_optimizer,
        "max_attempts": max_attempts,
        "target_imaginary_count": desired_count,
        "final_imaginary_count": relevant_imaginary_count(frequency_result, target),
        "is_clean": stationary_point_is_clean(frequency_result, target),
    }
    return atoms, frequency_result, total_cleanup_steps, cleanup_summary
