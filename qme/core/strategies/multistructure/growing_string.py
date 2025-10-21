"""Growing string method strategy for TS search."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from ase import Atoms

from qme.core.strategies.local.helpers import validate_ts_structure
from qme.core.strategies.local.minima import LocalMinimaStrategy
from qme.core.strategies.local.ts import LocalTSStrategy
from qme.core.strategy import REGISTRY, BaseStrategy, StrategyMetadata
from qme.core.strategy_utils import StrategyUtils
from qme.logging_utils import get_qme_logger

logger = get_qme_logger(__name__)


class MultiStructureGrowingStringStrategy(BaseStrategy):
    """Growing string method strategy for TS search."""

    metadata = StrategyMetadata(
        name="ts:growing_string",
        target="ts",
        strategy="growing_string",
        description="Growing string method for TS search (DE-GSM style)",
        aliases=["growing_string", "gsm"],
        requires_multiple_structures=True,
    )

    def run(self, atoms_list: list[Atoms], validate_ts: bool = False, **kwargs: Any) -> dict[str, Union[Atoms, list[Atoms], bool, int, float, str]]:
        """Run growing string method using in-module helpers (no runner delegation)."""
        self.validate_inputs(atoms_list)

        # Parse params with defaults for strategy execution
        npoints: int = kwargs.get("npoints", 15)
        fmax: float = kwargs.get("fmax", 0.05)
        steps: int = kwargs.get("steps", 100)
        step_size: float = kwargs.get("step_size", 0.1)
        distance_threshold: float = kwargs.get("distance_threshold", 0.5)
        local_optimizer_name: str = kwargs.get("local_optimizer_name", "sella")
        optimize_endpoints: bool = kwargs.get("optimize_endpoints", True)
        refine_ts: bool = kwargs.get("refine_ts", True)

        # Enforce exactly two structures (reactant, product)
        if isinstance(atoms_list, Atoms):
            raise ValueError(
                "Growing string method requires two Atoms objects (reactant and product)"
            )
        seq = list(atoms_list)
        if len(seq) != 2:
            raise ValueError(
                f"Growing string method requires exactly 2 Atoms objects, got {len(seq)}"
            )

        reactant, product = seq[0].copy(), seq[1].copy()

        # Attach calculators
        if self.explorer is not None:
            self.explorer._create_and_attach_calculator(reactant)
            self.explorer._apply_constraints(reactant)
            self.explorer._create_and_attach_calculator(product)
            self.explorer._apply_constraints(product)

        # Step 1: Optimize endpoints to local minima if requested
        if optimize_endpoints:
            logger.info("Growing String: Optimizing reactant to local minimum...")
            minima_strategy = LocalMinimaStrategy(explorer=self.explorer)
            r_result = minima_strategy.run(
                reactant, fmax=fmax, steps=200, local_optimizer_name="lbfgs"
            )
            reactant = r_result["optimized_atoms"]

            logger.info("Growing String: Optimizing product to local minimum...")
            p_result = minima_strategy.run(
                product, fmax=fmax, steps=200, local_optimizer_name="lbfgs"
            )
            product = p_result["optimized_atoms"]

            # Re-attach calculators after optimization (may have been lost)
            if self.explorer is not None:
                self.explorer._create_and_attach_calculator(reactant)
                self.explorer._apply_constraints(reactant)
                self.explorer._create_and_attach_calculator(product)
                self.explorer._apply_constraints(product)

        # Initialize strings
        forward_string = [reactant.copy()]  # Growing from reactant
        backward_string = [product.copy()]  # Growing from product

        # Ensure calculators are attached to initial string nodes
        if self.explorer is not None:
            for atoms in forward_string:
                self.explorer._create_and_attach_calculator(atoms)
                self.explorer._apply_constraints(atoms)
            for atoms in backward_string:
                self.explorer._create_and_attach_calculator(atoms)
                self.explorer._apply_constraints(atoms)

        logger.info(f"Growing String: Starting with reactant and product, max {npoints} images")

        # Step 2: Grow strings iteratively
        converged = False
        for iteration in range(steps):
            # Check if we've reached max images
            total_images = len(forward_string) + len(backward_string)
            if total_images >= npoints:
                logger.info(f"Growing String: Reached maximum images ({npoints})")
                break

            # Check if strings have met
            forward_tip = forward_string[-1].positions
            backward_tip = backward_string[-1].positions
            distance = np.linalg.norm(forward_tip - backward_tip)

            if distance < distance_threshold:
                logger.info(
                    f"Growing String: Strings met after {iteration} iterations "
                    f"(distance: {distance:.4f} Å)"
                )
                converged = True
                break

            # Grow forward string (from reactant)
            if total_images < npoints:
                new_forward = StrategyUtils.grow_string_node(
                    forward_string[-1],
                    direction="forward",
                    step_size=step_size,
                    fmax=fmax,
                    explorer=self.explorer,
                )
                if new_forward is not None:
                    forward_string.append(new_forward)

            # Grow backward string (from product)
            total_images = len(forward_string) + len(backward_string)
            if total_images < npoints:
                new_backward = StrategyUtils.grow_string_node(
                    backward_string[-1],
                    direction="backward",
                    step_size=step_size,
                    fmax=fmax,
                    explorer=self.explorer,
                )
                if new_backward is not None:
                    backward_string.append(new_backward)

            # Log progress every 10 iterations
            if (iteration + 1) % 10 == 0:
                logger.info(
                    f"Growing String: Iteration {iteration + 1}, "
                    f"forward={len(forward_string)}, backward={len(backward_string)}, "
                    f"distance={distance:.4f} Å"
                )

        # Step 3: Combine strings into full path
        # Reverse backward string so it goes from TS to product
        full_path = forward_string + backward_string[::-1]

        logger.info(
            f"Growing String: Complete with {len(full_path)} images "
            f"(forward: {len(forward_string)}, backward: {len(backward_string)})"
        )

        # Step 4: Find transition state as highest energy image
        energies = []
        for atoms in full_path:
            try:
                energy = atoms.get_potential_energy()
                energies.append(energy)
            except Exception:
                energies.append(float("-inf"))

        if not energies or all(e == float("-inf") for e in energies):
            # Fallback to middle structure
            ts_index = len(full_path) // 2
            logger.warning(
                "Growing String: Could not calculate energies, using middle image as TS guess"
            )
        else:
            ts_index = energies.index(max(energies))
            logger.info(
                f"Growing String: Highest energy at image {ts_index} "
                f"(E = {energies[ts_index]:.6f} eV)"
            )

        ts_guess = full_path[ts_index]

        # Step 5: Optionally refine TS with local optimization
        if refine_ts:
            logger.info("Growing String: Refining TS with local optimization...")

            # Avoid passing duplicate keys via kwargs when calling the strategy
            ts_call_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in {"fmax", "steps", "explorer", "local_optimizer_name"}
            }
            ts_strategy = LocalTSStrategy(explorer=self.explorer)
            ts_result = ts_strategy.run(
                ts_guess,
                fmax=fmax,
                steps=1000,
                local_optimizer_name=local_optimizer_name,
                **ts_call_kwargs,
            )
            ts_structure = ts_result["optimized_atoms"]
            ts_converged = ts_result["converged"]
        else:
            ts_structure = ts_guess
            ts_converged = converged

        # Validate TS structure if requested
        validation_result = None
        if validate_ts:
            validation_result = validate_ts_structure(ts_structure, self.explorer)

        result = self.prepare_result(
            ts_structure,
            converged=ts_converged,
            trajectory=full_path,
            forward_string=forward_string,
            backward_string=backward_string,
            strings_met=converged,
        )
        if validation_result is not None:
            result["ts_validation"] = validation_result
        return result


# Register the strategy
REGISTRY.register(MultiStructureGrowingStringStrategy)
