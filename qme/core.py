"""
Core optimization functionality combining ASE and SELLA optimizers with neural
network potentials. Supports multiple backends including UMA and SO3LR.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS, FIRE, LBFGS

from .calculator_registry import calculator_registry
from .config import config, get_default_model
from .dependencies import HAS_SELLA, deps


class QMEOptimizer:
    """Main optimizer class that combines ASE and SELLA optimizers with neural
    network potentials.

    Supports UMA, SO3LR, and AIMNET2 backends.

    This class provides a unified interface for molecular geometry optimization
    using machine learning potentials, supporting both minimum energy
    optimization and transition state searches.

    Attributes:
        calculator: The underlying energy/force calculator (UMA, SO3LR, AIMNET2,
            or mock)
        atoms: Currently loaded molecular structure
        results: Dictionary storing optimization results

    """

    AVAILABLE_OPTIMIZERS = {
        "BFGS": BFGS,
        "LBFGS": LBFGS,
        "FIRE": FIRE,
    }

    AVAILABLE_BACKENDS = {
        "so3lr": "SO3LR (SO(3) Invariant Neural Network)",
        "uma": "UMA (Universal Model for Atoms)",
        "aimnet2": "AIMNET2 (Accurate Neural Network Potential)",
        "mock": "Mock Calculator (for testing)",
    }

    if HAS_SELLA:
        AVAILABLE_OPTIMIZERS["Sella"] = deps.get("sella")

    def __init__(
        self,
        calculator=None,
        backend: Optional[str] = None,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_mock: bool = False,
    ):
        """Initialize QME optimizer.

        Parameters:
        -----------
        calculator : Calculator, optional
            Pre-configured calculator. If None, creates one based on backend.
        backend : str, optional
            Neural network backend to use ('uma', 'so3lr', 'aimnet2').
            Uses configuration default if None.
        model_name : str, optional
            Model name to use. Uses configuration defaults if None.
        model_path : str, optional
            Path to model file (SO3LR only)
        device : str, optional
            Device for computations ('cpu', 'cuda'). Auto-detected if None.
        use_mock : bool
            Use mock calculator for testing (default: False)
        """

        # Use configuration defaults if not provided
        if backend is None:
            backend = config.default_backend

        if backend not in QMEOptimizer.AVAILABLE_BACKENDS:
            available = list(QMEOptimizer.AVAILABLE_BACKENDS.keys())
            raise ValueError(f"Unknown backend: {backend}. Available: {available}")

        self.backend = backend

        if calculator is None:
            if backend == "mock" or use_mock:
                # Use mock backend explicitly or via use_mock flag
                # (for backwards compatibility)
                from .mock_calculator import MockCalculator

                # Determine which mock backend to simulate
                mock_backend = "generic"
                if backend in ["so3lr", "uma", "aimnet2"]:
                    mock_backend = backend
                self.calculator = MockCalculator(backend=mock_backend)
            else:
                self.calculator = self._create_calculator(
                    backend, model_name, model_path, device
                )
        else:
            self.calculator = calculator

        self.atoms: Optional[Atoms] = None
        self.results: Dict[str, Any] = {}

    def _create_calculator(
        self,
        backend: str,
        model_name: Optional[str],
        model_path: Optional[str],
        device: Optional[str],
    ):
        """Create calculator based on backend using the registry."""
        if model_name is None:
            model_name = get_default_model(backend)

        if device is None:
            device = config.get_device_preference()

        # Use the centralized calculator registry
        return calculator_registry.create_calculator(
            backend=backend, model_name=model_name, model_path=model_path, device=device
        )

    def load_structure(self, structure_file: Union[str, Path]) -> Atoms:
        """Load molecular structure from file.

        Args:
            structure_file: Path to structure file (xyz, cif, pdb, etc.).

        Returns:
            Loaded ASE Atoms object with calculator attached.

        Raises:
            FileNotFoundError: If structure file doesn't exist.
            ValueError: If structure file is invalid or empty.
            RuntimeError: If structure loading fails.
        """

        structure_file = Path(structure_file)

        # Validate file existence and format
        if not structure_file.exists():
            raise FileNotFoundError(f"Structure file not found: {structure_file}")

        if structure_file.stat().st_size == 0:
            raise ValueError(f"Structure file is empty: {structure_file}")

        # Validate file format
        valid_extensions = {".xyz", ".cif", ".pdb", ".mol", ".sdf", ".cml", ".traj"}
        if structure_file.suffix.lower() not in valid_extensions:
            if config.enable_warnings:
                print(
                    f"Warning: Unrecognized file extension '{structure_file.suffix}'. "
                    f"Supported formats: {', '.join(valid_extensions)}"
                )

        try:
            atoms_result = read(structure_file)

            # Handle case where read() returns a list of atoms objects
            if isinstance(atoms_result, list):
                if len(atoms_result) == 0:
                    raise ValueError(f"No structures found in file: {structure_file}")
                elif len(atoms_result) > 1:
                    if config.enable_warnings:
                        print(
                            f"Warning: Multiple structures found in {structure_file}. "
                            "Using the first one."
                        )
                atoms = atoms_result[0]
            else:
                atoms = atoms_result

            # Validate loaded structure
            if len(atoms) == 0:
                raise ValueError(f"No atoms found in structure file: {structure_file}")

            # Check for reasonable coordinates
            if hasattr(atoms, "positions") and atoms.positions is not None:
                pos = atoms.get_positions()
                if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                    raise ValueError(
                        f"Invalid coordinates (NaN or Inf) in structure: "
                        f"{structure_file}"
                    )

            atoms.calc = self.calculator
            self.atoms = atoms
            return atoms

        except Exception as e:
            if "No such file or directory" in str(e):
                raise FileNotFoundError(f"Structure file not found: {structure_file}")
            elif "Empty file" in str(e) or "No data" in str(e):
                raise ValueError(
                    f"Structure file appears to be empty or corrupted: {structure_file}"
                )
            else:
                raise RuntimeError(
                    f"Failed to load structure from {structure_file}: {e}"
                )

    def _run_single_optimization(
        self,
        atoms: Atoms,
        optimizer: str,
        fmax: float,
        steps: int,
        logfile: Optional[str] = None,
        trajectory: Optional[str] = None,
        **optimizer_kwargs,
    ) -> tuple[bool, int, str]:
        """Run the specified optimizer for geometry optimization.

        This method initializes and runs a single ASE optimizer instance
        with the provided settings. It does not implement any fallback
        mechanisms or pre-optimization strategies.

        Args:
            atoms: Structure to optimize
            optimizer: Name of the optimizer to use (e.g., 'BFGS', 'LBFGS', 'FIRE')
            fmax: Force convergence criterion
            steps: Maximum number of optimization steps
            logfile: Optional log file for optimization output
            trajectory: Optional trajectory file to save optimization steps
            **optimizer_kwargs: Additional arguments passed to the optimizer constructor

        Returns:
            Tuple of (converged, steps_taken, optimizer_used)
        """
        OptimizerClass = self.AVAILABLE_OPTIMIZERS[optimizer]
        opt = OptimizerClass(
            atoms, logfile=logfile, trajectory=trajectory, **optimizer_kwargs
        )

        try:
            converged = opt.run(fmax=fmax, steps=steps)
            steps_taken = opt.get_number_of_steps()
        except np.linalg.LinAlgError as e:
            print(
                f"Warning: {optimizer} failed with a linear algebra error "
                f"(e.g., Hessian not converging): {e}"
            )
            print(
                "This can sometimes happen with difficult geometries or "
                "specific optimizers."
            )
            raise RuntimeError(
                f"Optimization with {optimizer} failed due to a linear algebra "
                f"error: {e}"
            ) from e
        except Exception as e:
            print(f"Warning: {optimizer} failed with an unexpected error: {e}")
            raise RuntimeError(
                f"Optimization with {optimizer} failed unexpectedly: {e}"
            ) from e

        print(
            f"{optimizer} optimization completed: converged={converged}, "
            f"steps={steps_taken}"
        )

        return converged, steps_taken, optimizer

    def optimize_minimum(
        self,
        atoms: Optional[Atoms] = None,
        optimizer: str = "BFGS",
        fmax: float = 0.01,
        steps: int = 200,
        logfile: Optional[str] = None,
        trajectory: Optional[str] = None,
        initial_fmax_factor: float = 10.0,  # Factor to loosen fmax in initial stage
        initial_steps_fraction: float = 0.5,  # Fraction of total steps for initial
        constraints: Optional[List] = None,
        **optimizer_kwargs,
    ) -> Dict[str, Any]:
        """Optimize structure to find minimum energy geometry.

        Args:
            atoms: Structure to optimize. Uses self.atoms if None.
            optimizer: Optimizer name ('BFGS', 'LBFGS', 'FIRE').
            fmax: Force convergence criterion (eV/Å).
            steps: Maximum optimization steps.
            logfile: Optional log file for optimization output.
            trajectory: Optional trajectory file to save optimization steps.
            initial_fmax_factor: Factor by which to multiply `fmax` for an
                initial, looser optimization stage. Set to 1.0 or less to disable.
            initial_steps_fraction: Fraction of total `steps` to allocate for the
                initial, looser optimization stage.
            constraints: Optional list of ASE constraints.
            **optimizer_kwargs: Additional arguments passed to optimizer.

        Returns:
            Dictionary containing:
                - converged: Whether optimization converged
                - optimized_atoms: Final optimized structure
                - steps_taken: Number of optimization steps
                - energy_change: Energy change during optimization
                - final_max_force: Maximum force in final structure

        Raises:
            ValueError: If optimizer is not available.
            RuntimeError: If optimization fails.
        """

        if atoms is None:
            if self.atoms is None:
                raise ValueError("No structure loaded. Use load_structure() first.")
            atoms = self.atoms.copy()
        else:
            atoms = atoms.copy()

        atoms.calc = self.calculator

        # Apply constraints if provided
        if constraints:
            atoms.set_constraint(constraints)

        # Validate optimizer
        if optimizer not in self.AVAILABLE_OPTIMIZERS:
            available = list(self.AVAILABLE_OPTIMIZERS.keys())
            raise ValueError(f"Unknown optimizer: {optimizer}. Available: {available}")

        # Store initial state
        initial_energy = atoms.get_potential_energy()
        initial_forces = atoms.get_forces()
        initial_max_force = np.max(np.abs(initial_forces))

        total_steps_taken = 0
        converged = False
        optimizer_used = optimizer  # Default to the requested one

        # Adaptive fmax strategy: Looser initial stage, then tighter final stage
        if initial_fmax_factor > 1.0 and initial_steps_fraction > 0:
            # Stage 1: Looser fmax for initial steps
            stage1_fmax = fmax * initial_fmax_factor
            stage1_steps = int(steps * initial_steps_fraction)

            if stage1_steps > 0:
                print(
                    f"Stage 1: Optimizing with {optimizer} "
                    f"(fmax={stage1_fmax:.4f} eV/Å, steps={stage1_steps})..."
                )
                converged_stage1, steps_taken_stage1, _ = self._run_single_optimization(
                    atoms,
                    optimizer,
                    stage1_fmax,
                    stage1_steps,
                    logfile,
                    trajectory,
                    **optimizer_kwargs,
                )
                total_steps_taken += steps_taken_stage1

                if converged_stage1:
                    print(
                        "Stage 1 converged (forces below initial_fmax). "
                        "Proceeding to final convergence check."
                    )
                else:
                    print(
                        f"Stage 1 did not converge (max force: "
                        f"{np.max(np.abs(atoms.get_forces())):.4f} eV/Å)."
                    )

            # Stage 2: Tighter fmax for remaining steps
            remaining_steps = steps - total_steps_taken
            if remaining_steps > 0:
                print(
                    f"Stage 2: Optimizing with {optimizer} "
                    f"(fmax={fmax:.4f} eV/Å, steps={remaining_steps})..."
                )
                converged_stage2, steps_taken_stage2, _ = self._run_single_optimization(
                    atoms,
                    optimizer,
                    fmax,
                    remaining_steps,
                    logfile,
                    trajectory,
                    **optimizer_kwargs,
                )
                total_steps_taken += steps_taken_stage2
                converged = converged_stage2
            else:
                print(
                    "No remaining steps for Stage 2. Checking final convergence "
                    "based on Stage 1."
                )
                # If stage 1 used all steps, check if it met the final fmax
                converged = np.max(np.abs(atoms.get_forces())) < fmax
        else:
            # Single stage optimization
            print(
                f"Optimizing with {optimizer} (fmax={fmax:.4f} eV/Å, steps={steps})..."
            )
            converged, total_steps_taken, _ = self._run_single_optimization(
                atoms, optimizer, fmax, steps, logfile, trajectory, **optimizer_kwargs
            )

        # Store final state
        final_energy = atoms.get_potential_energy()
        final_forces = atoms.get_forces()
        final_max_force = np.max(np.abs(final_forces))

        results = {
            "converged": converged,
            "steps_taken": total_steps_taken,
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_change": final_energy - initial_energy,
            "initial_max_force": initial_max_force,
            "final_max_force": final_max_force,
            "optimized_atoms": atoms,
            "optimizer_used": optimizer_used,
        }

        self.results["minimum_optimization"] = results
        return results

    def find_transition_state(
        self,
        atoms: Optional[Atoms] = None,
        fmax: float = 0.01,
        steps: int = 200,
        logfile: Optional[str] = None,
        trajectory: Optional[str] = None,
        constraints: Optional[List] = None,
        **sella_kwargs,
    ) -> Dict[str, Any]:
        """
        Find transition state (saddle point) using SELLA.

        Parameters:
        -----------
        atoms : Atoms, optional
            Starting structure for TS search
        fmax : float
            Force convergence criterion
        steps : int
            Maximum optimization steps
        logfile : str, optional
            Log file path
        trajectory : str, optional
            Trajectory file path
        constraints : list, optional
            List of ASE constraints
        **sella_kwargs :
            Additional SELLA arguments

        Returns:
        --------
        dict
            TS optimization results
        """

        if not HAS_SELLA:
            raise ImportError(
                "SELLA is required for transition state searches. "
                "Install with: pip install sella"
            )

        if atoms is None:
            if self.atoms is None:
                raise ValueError("No structure loaded. Use load_structure() first.")
            atoms = self.atoms.copy()
        else:
            atoms = atoms.copy()

        atoms.calc = self.calculator

        # Apply constraints if provided
        if constraints:
            atoms.set_constraint(constraints)

        # Initialize SELLA optimizer for TS search
        Sella = deps.require("sella", "transition state searches")
        opt = Sella(atoms, logfile=logfile, trajectory=trajectory, **sella_kwargs)

        # Store initial state
        initial_energy = atoms.get_potential_energy()
        initial_forces = atoms.get_forces()
        initial_max_force = np.max(np.abs(initial_forces))

        # Run TS optimization
        converged = opt.run(fmax=fmax, steps=steps)

        # Store final state
        final_energy = atoms.get_potential_energy()
        final_forces = atoms.get_forces()
        final_max_force = np.max(np.abs(final_forces))

        results = {
            "converged": converged,
            "steps_taken": opt.get_number_of_steps(),
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_change": final_energy - initial_energy,
            "initial_max_force": initial_max_force,
            "final_max_force": final_max_force,
            "ts_atoms": atoms,
            "optimizer_used": "Sella",
        }

        self.results["transition_state_search"] = results
        return results

    def save_structure(
        self, atoms: Atoms, output_file: Union[str, Path], format: Optional[str] = None
    ):
        """
        Save optimized structure to file.

        Parameters:
        -----------
        atoms : Atoms
            Structure to save
        output_file : str or Path
            Output file path
        format : str, optional
            File format (inferred from extension if None)
        """

        output_file = Path(output_file)

        try:
            if format is not None:
                write(output_file, atoms, format=format)
            else:
                write(output_file, atoms)
        except Exception as e:
            raise RuntimeError(f"Failed to save structure to {output_file}: {e}")

    def get_optimization_summary(self) -> str:
        """
        Get summary of optimization results.

        Returns:
        --------
        str
            Formatted summary text
        """

        summary_lines = ["QME Optimization Summary", "=" * 25]

        for calc_type, results in self.results.items():
            summary_lines.extend(
                [
                    f"\n{calc_type.replace('_', ' ').title()}:",
                    f"  Converged: {results['converged']}",
                    f"  Steps: {results['steps_taken']}",
                    f"  Initial Energy: {results['initial_energy']:.6f} eV",
                    f"  Final Energy: {results['final_energy']:.6f} eV",
                    f"  Energy Change: {results['energy_change']:.6f} eV",
                    f"  Max Force (initial): {results['initial_max_force']:.4f} eV/Å",
                    f"  Max Force (final): {results['final_max_force']:.4f} eV/Å",
                    f"  Optimizer: {results['optimizer_used']}",
                ]
            )

        return "\n".join(summary_lines)

    def clear_results(self):
        """Clear all stored optimization results.
        Useful when reusing the QMEOptimizer instance for multiple tasks."""
        self.results = {}


def minimize_structure(
    atoms: Atoms,
    backend: str = "uma",
    optimizer: str = "BFGS",
    fmax: float = 0.01,
    steps: int = 200,
    logfile: Optional[str] = None,
    trajectory: Optional[str] = None,
    use_mock: bool = False,
    **optimizer_kwargs,
) -> Atoms:
    """Convenience function to minimize a molecular structure.

    This function creates a QMEOptimizer and runs a geometry optimization,
    returning the optimized structure.

    Args:
        atoms: Input molecular structure to optimize.
        backend: ML backend to use ('uma', 'so3lr', 'aimnet2', 'mock').
        optimizer: Optimizer to use ('BFGS', 'LBFGS', 'FIRE').
        fmax: Force convergence criterion (eV/Å).
        steps: Maximum optimization steps.
        logfile: Optional log file for optimization output.
        trajectory: Optional trajectory file to save optimization steps.
        use_mock: Use mock calculator for testing (backward compatibility).
        **optimizer_kwargs: Additional arguments passed to optimizer.

    Returns:
        Optimized ASE Atoms object.

    Raises:
        ValueError: If optimizer or backend is not available.
        ImportError: If required dependencies are not available.
        RuntimeError: If optimization fails.

    Example:
        >>> from ase.build import molecule
        >>> from qme.core import minimize_structure
        >>> water = molecule('H2O')
        >>> optimized = minimize_structure(water, backend='uma', optimizer='BFGS')
    """
    # Check if optimizer is available
    available_optimizers = ["BFGS", "LBFGS", "FIRE"]
    if HAS_SELLA:
        available_optimizers.append("Sella")

    if optimizer not in available_optimizers:
        raise ImportError(f"Optimizer {optimizer} not available")

    # Create optimizer instance
    qme_opt = QMEOptimizer(backend=backend, use_mock=use_mock)

    # Run optimization
    results = qme_opt.optimize_minimum(
        atoms=atoms,
        optimizer=optimizer,
        fmax=fmax,
        steps=steps,
        logfile=logfile,
        trajectory=trajectory,
        **optimizer_kwargs,
    )

    return results["optimized_atoms"]
