"""IRC (Intrinsic Reaction Coordinate) path calculation strategy."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, cast

import numpy as np
from ase import Atoms
from numpy.typing import NDArray

from famex.analysis.frequency import FrequencyAnalysis
from famex.core.base_strategy import BaseStrategy, StrategyMetadata
from famex.core.registry import REGISTRY
from famex.strategies.helpers import _get_local_optimizer_class, validate_ts_structure
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)


class LocalIRCStrategy(BaseStrategy):
    """IRC (Intrinsic Reaction Coordinate) path calculation strategy."""

    metadata = StrategyMetadata(
        name="path:irc",
        target="path",
        strategy="irc",
        description="IRC (Intrinsic Reaction Coordinate) path from transition state",
        aliases=["irc", "local:irc", "local-irc"],
        requires_multiple_structures=False,
    )

    def run(
        self,
        atoms_list: Sequence[Atoms],
        fmax: float = 0.05,
        steps: int = 100,
        step_size: float = 0.1,
        direction: str = "both",
        validate_ts: bool = True,
        **kwargs: Any,
    ) -> dict[str, Atoms | list[Atoms] | bool | int | float | str]:
        atoms_list = list(atoms_list)
        self.validate_inputs(atoms_list)

        local_optimizer_name = kwargs.get("local_optimizer_name", "bfgs")
        verbose = kwargs.get("verbose", 1)
        imag_threshold = float(kwargs.get("imag_threshold", 50.0))

        if self.profiler is not None:
            self.profiler.snapshot_memory()

        if len(atoms_list) != 1:
            msg = (
                "IRC runner expects a single structure (transition state), "
                f"got {len(atoms_list)} structures"
            )
            raise ValueError(
                msg,
            )
        atoms_input = atoms_list[0]

        ts_atoms = atoms_input.copy()

        with self.profiler.profile_section("calculator_setup") if self.profiler else nullcontext():
            self.explorer._create_and_attach_calculator(ts_atoms)
            self.explorer._apply_constraints(ts_atoms)

        hessian: NDArray[np.float64] | None = None
        if validate_ts:
            with self.profiler.profile_section("ts_validation") if self.profiler else nullcontext():
                _validation_result, hessian_raw = validate_ts_structure(
                    ts_atoms,
                    self.explorer,
                    threshold=imag_threshold,
                    return_hessian=True,
                    verbose=verbose,
                )
                if hessian_raw is not None:
                    hessian = np.asarray(hessian_raw, dtype=np.float64)

        transition_mode, hessian_used = self._get_transition_vector(
            ts_atoms,
            hessian=hessian,
            imag_threshold=imag_threshold,
            verbose=verbose,
        )

        if verbose >= 1:
            logger.info("Starting IRC calculation from transition state")
            logger.info(f"Direction: {direction}, Max steps: {steps}, Step size: {step_size}")
            logger.info(f"Force threshold: {fmax} eV/Å")

        forward_path = []
        backward_path = []

        def displace_along_mode(sign: float) -> Atoms:
            # ASE stubs type Atoms.copy() as Any; cast restores Atoms for mypy.
            displaced = cast(Atoms, ts_atoms.copy())
            mode_3d = transition_mode.reshape((-1, 3))
            new_positions = displaced.get_positions() + sign * step_size * mode_3d
            displaced.set_positions(new_positions)
            self.explorer._create_and_attach_calculator(displaced)
            self.explorer._apply_constraints(displaced)
            return displaced

        def follow_irc_downhill(initial_atoms: Atoms, max_steps: int) -> list[Atoms]:
            path: list[Atoms] = []
            current = initial_atoms.copy()
            if current.calc is None:
                self.explorer._create_and_attach_calculator(current)
                self.explorer._apply_constraints(current)

            for _step in range(max_steps):
                current_forces = current.get_forces()
                current_masses = current.get_masses()
                max_force = float(np.max(np.abs(current_forces)))
                if max_force < fmax:
                    if verbose >= 2:
                        logger.debug(
                            "IRC converged to minimum (max force: %.6f eV/Å), optimizing endpoint",
                            max_force,
                        )
                    opt_class = _get_local_optimizer_class(local_optimizer_name)
                    opt_copy = current.copy()
                    self.explorer._create_and_attach_calculator(opt_copy)
                    self.explorer._apply_constraints(opt_copy)
                    opt = opt_class(opt_copy)
                    opt.run(fmax=fmax, steps=100)
                    path.append(opt_copy)
                    break

                mw_forces = current_forces / np.sqrt(current_masses[:, np.newaxis])
                mw_forces_norm = float(np.linalg.norm(mw_forces))
                if mw_forces_norm < 1e-10:
                    break

                step_direction = mw_forces / mw_forces_norm
                displacement = step_size * step_direction * np.sqrt(current_masses[:, np.newaxis])
                next_atoms = current.copy()
                next_atoms.set_positions(current.get_positions() + displacement)
                self.explorer._create_and_attach_calculator(next_atoms)
                self.explorer._apply_constraints(next_atoms)
                path.append(next_atoms.copy())
                current = next_atoms

            return path

        with self.profiler.profile_section("irc_calculation") if self.profiler else nullcontext():
            if direction.lower() in ("forward", "both"):
                if verbose >= 2:
                    logger.debug("Following IRC in forward direction")
                forward_path = follow_irc_downhill(displace_along_mode(1.0), steps)

            if direction.lower() in ("backward", "both"):
                if verbose >= 2:
                    logger.debug("Following IRC in backward direction")
                backward_path = follow_irc_downhill(displace_along_mode(-1.0), steps)

        if direction.lower() == "both":
            trajectory = [
                *list(reversed(backward_path)),
                ts_atoms.copy(),
                *forward_path,
            ]
        elif direction.lower() == "forward":
            trajectory = [ts_atoms.copy(), *forward_path]
        elif direction.lower() == "backward":
            trajectory = [*list(reversed(backward_path)), ts_atoms.copy()]
        else:
            msg = f"Invalid direction: {direction}. Must be 'forward', 'backward', or 'both'"
            raise ValueError(
                msg,
            )

        if verbose >= 1:
            logger.info(f"IRC calculation completed: {len(trajectory)} images generated")
            logger.info(
                f"Forward path: {len(forward_path)} images, Backward path: {len(backward_path)} images",
            )

        result = self.prepare_result(
            trajectory,
            converged=True,
            trajectory=trajectory,
            forward_path=forward_path,
            backward_path=backward_path,
        )

        result["hessian_computed"] = True
        result["hessian"] = hessian_used

        return self._merge_profiler_results(result)

    def _get_transition_vector(
        self,
        ts_atoms: Atoms,
        hessian: NDArray[np.float64] | None,
        imag_threshold: float,
        verbose: int,
    ) -> tuple[np.ndarray, NDArray[np.float64]]:
        """Return the imaginary-mode eigenvector and the Hessian used for it."""
        if ts_atoms.calc is None:
            self.explorer._create_and_attach_calculator(ts_atoms)

        freq = FrequencyAnalysis(atoms=ts_atoms, calculator=ts_atoms.calc, verbose=0)
        hessian_used = (
            freq.set_hessian(hessian)
            if hessian is not None
            else np.asarray(freq.calculate_hessian(method="auto"), dtype=np.float64)
        )

        freq.diagonalize_hessian()
        frequencies = freq.get_frequencies(unit="cm-1", imag_threshold=imag_threshold)
        modes = freq.get_normal_modes()

        imag = frequencies < -imag_threshold
        if not np.any(imag):
            msg = (
                "IRC requires a transition-state imaginary mode "
                f"(frequency < -{imag_threshold} cm^-1); none found."
            )
            raise RuntimeError(msg)

        mode_idx = int(np.argmin(np.where(imag, frequencies, np.inf)))
        mode = np.asarray(modes[:, mode_idx], dtype=np.float64)
        norm = float(np.linalg.norm(mode))
        if norm < 1e-12:
            msg = "IRC transition vector has zero norm"
            raise RuntimeError(msg)

        if verbose >= 2:
            logger.debug(
                "IRC transition vector from mode %.1f cm^-1",
                float(frequencies[mode_idx]),
            )
        return mode / norm, hessian_used


REGISTRY.register(LocalIRCStrategy)
