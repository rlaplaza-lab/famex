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

try:
    from sella import Sella

    HAS_SELLA = True
except ImportError:
    HAS_SELLA = False

from .aimnet2_potential import AIMNet2Potential, get_aimnet2_calculator
from .mock_calculator import get_mock_so3lr_calculator, get_mock_uma_calculator
from .so3lr_potential import (
    SO3LRPotential,
    get_so3lr_calculator,
)
from .uma_potential import UMAPotential, get_uma_calculator


class QMEOptimizer:
    """Main optimizer class that combines ASE and SELLA optimizers with neural network potentials.
    Supports UMA, SO3LR, and AIMNET2 backends.

    This class provides a unified interface for molecular geometry optimization
    using machine learning potentials, supporting both minimum energy
    optimization and transition state searches.
    and transition state searches.

    Attributes:
        calculator: The underlying energy/force calculator (UMA, SO3LR, AIMNET2, or mock)
        atoms: Currently loaded molecular structure
        results: Dictionary storing optimization results

    """

    AVAILABLE_OPTIMIZERS = {
        "BFGS": BFGS,
        "LBFGS": LBFGS,
        "FIRE": FIRE,
    }

    AVAILABLE_BACKENDS = {
        "uma": "UMA (Universal Model for Atoms)",
        "so3lr": "SO3LR (SO(3) Invariant Neural Network)",
        "aimnet2": "AIMNET2 (Accurate Neural Network Potential)",
    }

    if HAS_SELLA:
        AVAILABLE_OPTIMIZERS["Sella"] = Sella

    def __init__(
        self,
        calculator=None,
        backend: str = "so3lr",  # Changed default to SO3LR for testing
        model_name: str = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_mock: bool = False,
    ):
        """Initialize QME optimizer.

        Parameters:
        -----------
        calculator : Calculator, optional
            Pre-configured calculator. If None, creates one based on backend.
        backend : str
            Neural network backend to use ('uma', 'so3lr', 'aimnet2'). Default: 'so3lr'
        model_name : str, optional
            Model name to use. Defaults depend on backend.
        model_path : str, optional
            Path to model file (SO3LR only)
        device : str, optional
            Device for computations ('cpu', 'cuda')
        use_mock : bool
            Use mock calculator for testing (default: False)
        """

        if backend not in self.AVAILABLE_BACKENDS:
            available = list(self.AVAILABLE_BACKENDS.keys())
            raise ValueError(f"Unknown backend: {backend}. Available: {available}")

        self.backend = backend

        if calculator is None:
            if use_mock:
                if backend == "so3lr":
                    self.calculator = get_mock_so3lr_calculator()
                elif backend == "uma":
                    self.calculator = get_mock_uma_calculator()
                elif backend == "aimnet2":
                    from .mock_calculator import get_mock_aimnet2_calculator

                    self.calculator = get_mock_aimnet2_calculator()
            else:
                self.calculator = self._create_calculator(
                    backend, model_name, model_path, device
                )
        else:
            self.calculator = calculator

        self.atoms = None
        self.results = {}

    def _create_calculator(
        self,
        backend: str,
        model_name: Optional[str],
        model_path: Optional[str],
        device: Optional[str],
    ):
        """Create calculator based on backend."""
        try:
            if backend == "so3lr":
                # Set default model name for SO3LR if not provided
                if model_name is None:
                    model_name = "so3lr-small"
                return get_so3lr_calculator(
                    model_path=model_path, model_name=model_name, device=device
                )

            elif backend == "uma":
                # Set default model name for UMA if not provided
                if model_name is None:
                    model_name = "uma-4m"
                return get_uma_calculator(model_name=model_name, device=device)

            elif backend == "aimnet2":
                # Set default model name for AIMNET2 if not provided
                if model_name is None:
                    model_name = "aimnet2"
                return get_aimnet2_calculator(model_name=model_name, device=device)

        except ImportError as e:
            print(f"Warning: {e}")
            print(f"Falling back to mock {backend.upper()} calculator for testing.")
            if backend == "so3lr":
                return get_mock_so3lr_calculator()
            elif backend == "uma":
                return get_mock_uma_calculator()
            elif backend == "aimnet2":
                from .mock_calculator import get_mock_aimnet2_calculator

                return get_mock_aimnet2_calculator()

    def load_structure(self, structure_file: Union[str, Path]) -> Atoms:
        """Load molecular structure from file.

        Args:
            structure_file: Path to structure file (xyz, cif, pdb, etc.).

        Returns:
            Loaded ASE Atoms object with calculator attached.

        Raises:
            FileNotFoundError: If structure file doesn't exist.
            RuntimeError: If structure loading fails.
        """

        structure_file = Path(structure_file)
        if not structure_file.exists():
            raise FileNotFoundError(f"Structure file not found: {structure_file}")

        try:
            atoms = read(structure_file)
            atoms.calc = self.calculator
            self.atoms = atoms
            return atoms
        except Exception as e:
            raise RuntimeError(f"Failed to load structure from {structure_file}: {e}")

    def optimize_minimum(
        self,
        atoms: Optional[Atoms] = None,
        optimizer: str = "BFGS",
        fmax: float = 0.01,
        steps: int = 200,
        logfile: Optional[str] = None,
        trajectory: Optional[str] = None,
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

        # Get optimizer class
        if optimizer not in self.AVAILABLE_OPTIMIZERS:
            available = list(self.AVAILABLE_OPTIMIZERS.keys())
            raise ValueError(f"Unknown optimizer: {optimizer}. Available: {available}")

        OptimizerClass = self.AVAILABLE_OPTIMIZERS[optimizer]

        # Initialize optimizer
        opt = OptimizerClass(
            atoms, logfile=logfile, trajectory=trajectory, **optimizer_kwargs
        )

        # Store initial state
        initial_energy = atoms.get_potential_energy()
        initial_forces = atoms.get_forces()
        initial_max_force = np.max(np.abs(initial_forces))

        # Run optimization
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
            "optimized_atoms": atoms,
            "optimizer_used": optimizer,
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
            write(output_file, atoms, format=format)
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
