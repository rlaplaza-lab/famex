"""Two-ended strategy runners for Explorer.

This module provides small runners that operate on two or more ASE
Atoms objects. Input can be either exactly two Atoms (reactant, product)
or a sequence of Atoms interpreted as [reactant, intermediate_guess..., product].
When the sequence contains more than two states the module will interpolate
each consecutive pair and stitch the segments into a single path.

The focus is interpolation (linear or geodesic), optional simple path
optimization, and extracting a transition-state guess (middle structure
of the stitched interpolated path).
"""

import warnings
from typing import Any, List, Optional, Sequence, Union

import numpy as np
from ase import Atoms

from qme.core.local_strategies import _get_local_optimizer_class, _validate_ts_optimization_setup
from qme.core.reaction import Reaction


def _supports_batch_evaluation(calculator):
    """Check if calculator supports batch evaluation."""
    return hasattr(calculator, "supports_batch_evaluation") and calculator.supports_batch_evaluation


def path_generator(
    atoms_list: Union[Sequence[Atoms], Atoms],
    npoints: int = 11,
    method: str = "geodesic",
    optimize_path: bool = True,
    explorer: Optional[Any] = None,
    calculator=None,
    **kwargs,
):
    """Generate an interpolated path between two or more Atoms objects.

    This simplified helper focuses on path construction only. It accepts
    either two endpoints or a sequence of intermediate guesses. When more
    than two structures are provided, each consecutive pair is interpolated
    and the segments are stitched together to produce a single path of
    approximately `npoints` frames.
    """
    # Normalize input and validate
    if isinstance(atoms_list, Atoms):
        raise ValueError(
            "Two-ended strategies expect two or more Atoms objects, not a single Atoms"
        )

    seq = list(atoms_list)
    if len(seq) < 2:
        raise ValueError("Two-ended strategies require two or more Atoms objects")

    # Simple two-end case
    if len(seq) == 2:
        r, p = seq[0], seq[1]
        calc = calculator
        if explorer is not None and calc is None:
            calc = _attach_calculators_if_explorer(explorer, r, p)
        reaction = Reaction(r, p, calculator=calc)
        return reaction.interpolate(
            npoints=npoints, method=method, optimize_path=optimize_path, calculator=calc
        )

    # Multi-segment: distribute frames across segments and stitch
    segments = len(seq) - 1
    if npoints < 2:
        raise ValueError("Need at least 2 points for interpolation")

    total_intervals = npoints - 1
    base_intervals = total_intervals // segments
    remainder = total_intervals % segments

    per_segment_npoints = []
    for i in range(segments):
        intervals = base_intervals + (1 if i < remainder else 0)
        per_segment_npoints.append(intervals + 1)

    calc = calculator
    if explorer is not None and calc is None:
        calc = _attach_calculators_if_explorer(explorer, seq[0], seq[-1])

    stitched_path = []
    for i in range(segments):
        r, p = seq[i], seq[i + 1]
        nseg = per_segment_npoints[i]
        reaction = Reaction(r, p, calculator=calc)
        seg_path = reaction.interpolate(
            npoints=nseg, method=method, optimize_path=optimize_path, calculator=calc
        )
        if i == 0:
            stitched_path.extend(seg_path)
        else:
            stitched_path.extend(seg_path[1:])

    return stitched_path


def _attach_calculators_if_explorer(explorer: Any, reactant: Atoms, product: Atoms):
    """Use explorer helpers to create a calculator and attach to endpoints.

    Returns the calculator instance or None if creation failed.
    """
    try:
        # Only create and attach calculator if atoms doesn't already have one
        if getattr(reactant, "calc", None) is None:
            calc_r = explorer._create_and_attach_calculator(reactant)
        else:
            calc_r = reactant.calc
        if getattr(product, "calc", None) is None:
            calc_p = explorer._create_and_attach_calculator(product)
        else:
            calc_p = product.calc
        # Prefer the reactant calculator if both were created
        return calc_r or calc_p
    except Exception:
        warnings.warn("Failed to attach calculators via explorer")
        return None


def twoended_ts_guess_runner(
    atoms_list: Union[Sequence[Atoms], Atoms],
    npoints: int = 11,
    method: str = "geodesic",
    explorer: Optional[Any] = None,
    fmax: float = 0.05,
    steps: int = 1000,
    local_optimizer_name: str = "sella",
    **kwargs,
):
    """Interpolate and then run local TS optimization on the highest-energy frame(s).

    This convenience wrapper generates an interpolated path using the same
    logic as the interpolate runner, locates the highest-energy frame(s)
    and attempts a local TS optimization using the explorer's calculator
    helpers and the same optimizer selection logic as `_local_ts_runner`.
    """
    # Validate TS optimization setup - hardcoded restrictions
    if explorer is not None:
        _validate_ts_optimization_setup(explorer.backend, local_optimizer_name)

    # Generate stitched path (or single-segment path)
    path = path_generator(
        atoms_list,
        npoints=npoints,
        method=method,
        optimize_path=True,
        explorer=explorer,
        **kwargs,
    )

    # Reaction.calculate_path_energies expects Geometry objects; ensure we have energies
    # If path is a nested list (per-segment) flatten
    if path and isinstance(path[0], list):
        flat = []
        for seg in path:
            flat.extend(seg)
        path = flat

    # Try to compute energies using the Reaction helper by constructing a dummy Reaction
    if len(path) < 2:
        raise ValueError("Need at least two frames in interpolated path to locate TS guess")

    # Use the Reaction class between endpoints to access calculate_path_energies
    reaction = Reaction(path[0], path[-1], calculator=getattr(path[0], "calc", None))
    energies = reaction.calculate_path_energies(path)

    # Find index/indices of the maximum energy frames (could be multiple)
    import math

    if not energies:
        raise RuntimeError("Failed to calculate path energies for TS guess selection")

    max_e = max(energies)
    max_idxs = [i for i, e in enumerate(energies) if not math.isnan(e) and e == max_e]
    if not max_idxs:
        # Fall back to middle frame
        max_idxs = [len(path) // 2]

    ts_results = []
    # Select optimizer class
    opt_class = _get_local_optimizer_class(local_optimizer_name)

    for idx in max_idxs:
        geom = path[idx]
        # Ensure explorer attaches calculator
        if explorer is not None:
            try:
                # Only create and attach calculator if atoms doesn't already have one
                if getattr(geom, "calc", None) is None:
                    explorer._create_and_attach_calculator(geom)
            except Exception:
                warnings.warn("Failed to attach calculator to TS guess")
        # Apply constraints if any
        if explorer is not None:
            explorer._apply_constraints(geom)

        opt_kwargs = getattr(explorer, "ts_kwargs", {}) or {}
        if local_optimizer_name.lower() == "sella":
            opt_kwargs = dict(opt_kwargs)
            opt_kwargs.setdefault("internal", True)
            opt_kwargs.setdefault("order", 1)

        opt = opt_class(geom, **opt_kwargs)
        opt.run(fmax=fmax, steps=steps)
        ts_results.append(geom)

    # Return single geometry if single input expected shape
    # If original input was two Atoms, keep consistent and return single Geometry
    if isinstance(atoms_list, Atoms) or (
        isinstance(atoms_list, (list, tuple)) and len(atoms_list) == 2
    ):
        return ts_results[0] if ts_results else None
    return ts_results


__all__ = [
    "twoended_ts_guess_runner",
]


def twoended_minima_runner(
    atoms_list: Union[Sequence[Atoms], Atoms],
    npoints: int = 11,
    method: str = "geodesic",
    explorer: Optional[Any] = None,
    fmax: float = 0.05,
    steps: int = 1000,
    local_optimizer_name: str = "sella",
    **kwargs,
):
    """Interpolate path and attempt local minima optimizations on low-energy frames.

    This runner generates an interpolated path, computes approximate
    energies along it, finds local minima (or global min if none), and
    runs a local minima optimization on those frames. Returns optimized
    geometries (single geometry if input was two endpoints, otherwise a list).
    """
    path = path_generator(
        atoms_list,
        npoints=npoints,
        method=method,
        optimize_path=True,
        explorer=explorer,
        **kwargs,
    )

    # Flatten nested segments if needed
    if path and isinstance(path[0], list):
        flat = []
        for seg in path:
            flat.extend(seg)
        path = flat

    if len(path) < 1:
        raise ValueError("Interpolated path is empty; cannot locate minima")

    # Compute energies using Reaction helper
    reaction = Reaction(path[0], path[-1], calculator=getattr(path[0], "calc", None))
    energies = reaction.calculate_path_energies(path)

    import math

    # Find local minima indices: energy less than neighbours (handle endpoints)
    minima_idxs = []
    for i, e in enumerate(energies):
        if math.isnan(e):
            continue
        left = energies[i - 1] if i - 1 >= 0 else float("inf")
        right = energies[i + 1] if i + 1 < len(energies) else float("inf")
        if (not math.isnan(left) and e < left) and (not math.isnan(right) and e < right):
            minima_idxs.append(i)

    # If no strict local minima found, pick the global minimum
    if not minima_idxs:
        # choose argmin ignoring NaNs
        valid = [(i, e) for i, e in enumerate(energies) if not math.isnan(e)]
        if not valid:
            raise RuntimeError("No valid energies found along path to select minima")
        min_idx = min(valid, key=lambda ie: ie[1])[0]
        minima_idxs = [min_idx]

    results = []

    # Select optimizer class
    opt_class = _get_local_optimizer_class(local_optimizer_name)

    for idx in minima_idxs:
        geom = path[idx]
        # Attach calculator via explorer if available
        if explorer is not None:
            try:
                # Only create and attach calculator if atoms doesn't already have one
                if getattr(geom, "calc", None) is None:
                    explorer._create_and_attach_calculator(geom)
            except Exception:
                warnings.warn("Failed to attach calculator to minima guess")
            explorer._apply_constraints(geom)

        opt_kwargs = getattr(explorer, "optimizer_kwargs", {}) or {}
        if local_optimizer_name.lower() == "sella":
            opt_kwargs = dict(opt_kwargs)
            opt_kwargs.setdefault("internal", True)
            opt_kwargs.setdefault("order", 0)
        elif local_optimizer_name.lower() == "geometric":
            opt_kwargs = dict(opt_kwargs)
            opt_kwargs.setdefault("order", 0)

        opt = opt_class(geom, **opt_kwargs)
        opt.run(fmax=fmax, steps=steps)
        results.append(geom)

    # Return single geometry if original input was pair/endpoints
    if isinstance(atoms_list, Atoms) or (
        isinstance(atoms_list, (list, tuple)) and len(atoms_list) == 2
    ):
        return results[0] if results else None
    return results


class BatchNEBOptimizer:
    """
    NEB optimizer with batch evaluation support.

    This class implements a NEB optimization that leverages batch evaluation
    capabilities of calculators like TorchSim to calculate energies and forces
    for all NEB images simultaneously.
    """

    def __init__(self, path, calculator, fmax=0.05, steps=1000, spring_constant=5.0, **kwargs):
        """Initialize batch NEB optimizer."""
        self.path = [atoms.copy() for atoms in path]
        self.calculator = calculator
        self.fmax = fmax
        self.steps = steps
        self.spring_constant = spring_constant
        self.kwargs = kwargs

        # Attach calculator to all images
        for atoms in self.path:
            atoms.calc = calculator

    def optimize(self):
        """Optimize NEB path using batch evaluation."""
        print(f"Optimizing NEB path with {len(self.path)} images using " f"batch evaluation...")

        for step in range(self.steps):
            # Calculate forces for all images in one batch
            batch_results = self.calculator.calculate_batch(
                self.path, properties=["energy", "forces"]
            )

            # Extract forces from batch results
            forces = [result["forces"] for result in batch_results]

            # Apply NEB forces (spring + nudging)
            neb_forces = self._apply_neb_forces(forces)

            # Update positions
            self._update_positions(neb_forces)

            # Check convergence
            max_force = max(np.max(np.abs(force)) for force in neb_forces)
            if max_force < self.fmax:
                print(f"NEB converged after {step + 1} steps (max force: {max_force:.6f})")
                break

        return self.path

    def _apply_neb_forces(self, forces):
        """Apply NEB forces (spring + nudging)."""
        neb_forces = []

        for i in range(len(self.path)):
            if i == 0 or i == len(self.path) - 1:
                # Endpoints: use only spring forces
                neb_forces.append(self._spring_forces(i))
            else:
                # Middle images: spring forces + nudging
                spring_f = self._spring_forces(i)
                nudged_f = self._nudge_forces(forces[i], spring_f)
                neb_forces.append(nudged_f)

        return neb_forces

    def _spring_forces(self, i):
        """Calculate spring forces for image i."""
        if i == 0:
            # First image: spring to next
            return self.spring_constant * (self.path[i + 1].positions - self.path[i].positions)
        elif i == len(self.path) - 1:
            # Last image: spring to previous
            return self.spring_constant * (self.path[i - 1].positions - self.path[i].positions)
        else:
            # Middle images: spring to both neighbors
            f_prev = self.spring_constant * (self.path[i - 1].positions - self.path[i].positions)
            f_next = self.spring_constant * (self.path[i + 1].positions - self.path[i].positions)
            return f_prev + f_next

    def _nudge_forces(self, forces, spring_forces):
        """Apply nudging to forces (project out parallel component)."""
        # Calculate tangent vector (simplified)
        if len(self.path) > 1:
            tangent = self.path[1].positions - self.path[0].positions
            tangent = tangent / np.linalg.norm(tangent)
        else:
            tangent = np.zeros_like(forces[0])

        # Project forces perpendicular to tangent
        parallel_component = np.sum(forces * tangent) * tangent
        perpendicular_forces = forces - parallel_component

        return perpendicular_forces + spring_forces

    def _update_positions(self, forces, step_size=0.01):
        """Update positions using forces."""
        for i, (atoms, force) in enumerate(zip(self.path, forces)):
            atoms.positions += step_size * force


def _batch_neb_runner(
    atoms_list: Union[Sequence[Atoms], Atoms],
    calculator,
    npoints: int = 11,
    method: str = "geodesic",
    fmax: float = 0.05,
    steps: int = 1000,
    spring_constant: float = 5.0,
    **kwargs,
):
    """Batch NEB runner for calculators that support batch evaluation."""
    # Generate initial path using geodesic interpolation
    path = path_generator(
        atoms_list,
        npoints=npoints,
        method=method,
        optimize_path=False,  # Don't optimize initially, we'll do NEB
        calculator=calculator,
        **kwargs,
    )

    # Create batch NEB optimizer
    batch_neb = BatchNEBOptimizer(
        path,
        calculator=calculator,
        fmax=fmax,
        steps=steps,
        spring_constant=spring_constant,
        **kwargs,
    )

    # Optimize using batch evaluation
    optimized_path = batch_neb.optimize()

    return optimized_path


def twoended_neb_runner(
    atoms_list: Union[Sequence[Atoms], Atoms],
    npoints: int = 11,
    method: str = "geodesic",
    explorer: Optional[Any] = None,
    fmax: float = 0.05,
    steps: int = 1000,
    local_optimizer_name: str = "sella",
    spring_constant: float = 5.0,
    **kwargs,
):
    """Nudged Elastic Band (NEB) optimization using geodesic interpolation.

    This runner implements a simple NEB algorithm that:
    1. Generates an initial path using geodesic interpolation
    2. Applies spring forces between adjacent images
    3. Projects forces perpendicular to the path (nudging)
    4. Optimizes the entire path using the specified local optimizer

    Parameters
    ----------
    atoms_list : Union[Sequence[Atoms], Atoms]
        Two or more Atoms objects defining the path endpoints
    npoints : int, default=11
        Number of images in the NEB path
    method : str, default="geodesic"
        Interpolation method for initial path generation
    explorer : Any, optional
        Explorer instance for calculator and constraint management
    fmax : float, default=0.05
        Force convergence threshold
    steps : int, default=1000
        Maximum optimization steps
    local_optimizer_name : str, default="sella"
        Local optimizer to use for NEB optimization
    spring_constant : float, default=0.1
        Spring constant for NEB spring forces
    **kwargs
        Additional arguments passed to optimizer

    Returns
    -------
    List[Atoms]
        Optimized NEB path (list of Atoms objects)
    """
    # Check if we should use batch evaluation
    calculator = None
    if explorer is not None:
        # Create calculator using explorer's parameters
        from qme.core.calculator_setup import create_calculator

        calculator = create_calculator(
            backend=explorer.backend,
            model_name=explorer.model_name,
            model_path=explorer.model_path,
            device=explorer.device,
            default_charge=explorer.default_charge,
            default_spin=explorer.default_spin,
        )

    if calculator is not None and _supports_batch_evaluation(calculator):
        # Use batch NEB optimizer
        return _batch_neb_runner(
            atoms_list,
            calculator=calculator,
            npoints=npoints,
            method=method,
            fmax=fmax,
            steps=steps,
            spring_constant=spring_constant,
            **kwargs,
        )

    # Generate initial path using geodesic interpolation
    path = path_generator(
        atoms_list,
        npoints=npoints,
        method=method,
        optimize_path=False,  # Don't optimize initially, we'll do NEB
        explorer=explorer,
        **kwargs,
    )

    # Flatten nested segments if needed
    if path and isinstance(path[0], list):
        flat = []
        for seg in path:
            flat.extend(seg)
        path = flat

    if len(path) < 3:
        raise ValueError("NEB requires at least 3 images (npoints >= 3)")

    # Geometry objects inherit from Atoms, so no conversion needed

    # Attach calculators to all images
    if explorer is not None:
        for i, atoms in enumerate(path):
            try:
                # Only create and attach calculator if atoms doesn't already have one
                if getattr(atoms, "calc", None) is None:
                    explorer._create_and_attach_calculator(atoms)
                explorer._apply_constraints(atoms)
            except Exception as e:
                warnings.warn(f"Failed to attach calculator to image {i}: {e}")

    # Simple NEB implementation
    _run_simple_neb(
        path,
        spring_constant=spring_constant,
        fmax=fmax,
        steps=steps,
        local_optimizer_name=local_optimizer_name,
        explorer=explorer,
        **kwargs,
    )

    return path


def _run_simple_neb(
    path: List[Atoms],
    spring_constant: float = 5.0,
    fmax: float = 0.05,
    steps: int = 1000,
    local_optimizer_name: str = "sella",
    explorer: Optional[Any] = None,
    **kwargs,
):
    """Run a simple NEB optimization on the given path.

    This implements a basic NEB algorithm with spring forces and force projection.
    Uses batch evaluation when available for improved performance.
    """
    import numpy as np
    from ase.optimize import BFGS

    # Get optimizer class
    try:
        OptClass = _get_local_optimizer_class(local_optimizer_name)
    except Exception as e:
        warnings.warn(f"Could not select optimizer '{local_optimizer_name}': {e}")
        OptClass = BFGS  # Fallback to ASE BFGS

    # Check if we can use batch evaluation
    calculator = path[0].calc if path[0].calc is not None else None
    supports_batch = (
        calculator is not None
        and hasattr(calculator, "calculate_batch")
        and hasattr(calculator, "supports_batch_evaluation")
        and calculator.supports_batch_evaluation
    )

    if supports_batch:
        print(f"Using batch evaluation for NEB optimization with {len(path)} images")

    # NEB optimization loop
    for step in range(steps):
        # Calculate energies and forces for all images
        energies = []
        forces_list = []

        if supports_batch:
            # Use batch evaluation for better performance
            try:
                batch_results = calculator.calculate_batch(path, properties=["energy", "forces"])

                for result in batch_results:
                    energies.append(result.get("energy", float("inf")))
                    forces_list.append(result.get("forces", np.zeros((len(path[0]), 3))))

            except Exception as e:
                warnings.warn(
                    f"Batch evaluation failed, falling back to individual " f"calculations: {e}"
                )
                supports_batch = False  # Disable batch for future iterations

        if not supports_batch:
            # Fallback to individual calculations
            for atoms in path:
                try:
                    energy = atoms.get_potential_energy()
                    forces = atoms.get_forces()
                    energies.append(energy)
                    forces_list.append(forces)
                except Exception as e:
                    warnings.warn(f"Failed to calculate energy/forces: {e}")
                    energies.append(float("inf"))
                    forces_list.append(np.zeros((len(atoms), 3)))

        # Check convergence
        max_force = max(np.linalg.norm(forces, axis=1).max() for forces in forces_list)
        if max_force < fmax:
            print(f"NEB converged after {step + 1} steps (max force: {max_force:.6f})")
            break

        # Apply NEB forces
        for i in range(1, len(path) - 1):  # Skip endpoints
            atoms = path[i]
            forces = forces_list[i].copy()

            # Spring forces
            spring_force = _calculate_spring_force(path, i, spring_constant, energies)

            # Project forces perpendicular to path (nudging)
            tangent = _calculate_tangent(path, i, energies)
            if tangent is not None:
                # Remove component parallel to tangent
                parallel_component = np.dot(forces.flatten(), tangent) * tangent
                forces = forces.flatten() - parallel_component
                forces = forces.reshape(-1, 3)

            # Add spring forces
            forces += spring_force

            # Store forces in calculator results (ASE-compatible approach)
            if atoms.calc is not None:
                if not hasattr(atoms.calc, "results"):
                    atoms.calc.results = {}
                atoms.calc.results["forces"] = forces
            else:
                # Fallback: store in atoms info for compatibility
                atoms.info["forces"] = forces

        # Optimize each image (except endpoints)
        for i in range(1, len(path) - 1):
            atoms = path[i]
            try:
                opt = OptClass(atoms, **kwargs)
                opt.run(fmax=fmax, steps=1)  # Single step per iteration
            except Exception as e:
                warnings.warn(f"Optimization failed for image {i}: {e}")


def _calculate_spring_force(
    path: List[Atoms], index: int, spring_constant: float, energies: List[float]
) -> np.ndarray:
    """Calculate spring forces for NEB."""
    import numpy as np

    if index == 0 or index == len(path) - 1:
        return np.zeros((len(path[index]), 3))

    # Distance between adjacent images
    prev_pos = path[index - 1].get_positions()
    curr_pos = path[index].get_positions()
    next_pos = path[index + 1].get_positions()

    # Spring force towards previous image
    spring_prev = spring_constant * (prev_pos - curr_pos)
    # Spring force towards next image
    spring_next = spring_constant * (next_pos - curr_pos)

    return spring_prev + spring_next


def _calculate_tangent(
    path: List[Atoms], index: int, energies: List[float]
) -> Optional[np.ndarray]:
    """Calculate tangent vector for NEB force projection."""
    import numpy as np

    if index == 0 or index == len(path) - 1:
        return None

    # Use energy-weighted tangent calculation
    prev_pos = path[index - 1].get_positions()
    curr_pos = path[index].get_positions()
    next_pos = path[index + 1].get_positions()

    # Forward difference
    forward = next_pos - curr_pos
    # Backward difference
    backward = curr_pos - prev_pos

    # Energy-weighted tangent
    if energies[index + 1] > energies[index - 1]:
        tangent = forward
    else:
        tangent = backward

    # Normalize
    norm = np.linalg.norm(tangent)
    if norm > 1e-10:
        return tangent.flatten() / norm
    else:
        return None


__all__.append("twoended_minima_runner")
__all__.append("twoended_neb_runner")
