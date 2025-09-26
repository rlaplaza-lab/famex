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

    Supports UMA, SO3LR, AIMNET2, and MACE backends.

    This class provides a unified interface for molecular geometry optimization
    using machine learning potentials, supporting both minimum energy
    optimization and transition state searches.

    Attributes:
        calculator: The underlying energy/force calculator (UMA, SO3LR, AIMNET2,
            MACE, or mock)
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
        "mace": "MACE (Machine learning ACE potentials)",
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
        default_charge: int = 0,
        default_spin: int = 1,
    ):
        """Initialize QME optimizer.

        Parameters:
        -----------
        calculator : Calculator, optional
            Pre-configured calculator. If None, creates one based on backend.
        backend : str, optional
            Neural network backend to use ('uma', 'so3lr', 'aimnet2', 'mace', 'mock').
            Uses configuration default if None.
        model_name : str, optional
            Model name to use. Uses configuration defaults if None.
        model_path : str, optional
            Path to model file (SO3LR only)
        device : str, optional
            Device for computations ('cpu', 'cuda'). Auto-detected if None.
        default_charge : int, optional
            Default charge for molecular systems (default: 0)
        default_spin : int, optional
            Default spin multiplicity for molecular systems (default: 1)
        """

        # Use configuration defaults if not provided
        if backend is None:
            backend = config.default_backend

        if backend not in QMEOptimizer.AVAILABLE_BACKENDS:
            available = list(QMEOptimizer.AVAILABLE_BACKENDS.keys())
            raise ValueError(f"Unknown backend: {backend}. Available: {available}")

        self.backend = backend
        self.default_charge = default_charge
        self.default_spin = default_spin

        if calculator is None:
            if backend == "mock":
                # Use mock backend
                from .mock_calculator import MockCalculator

                # Use generic mock backend
                self.calculator = MockCalculator(backend="generic")
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
            backend=backend,
            model_name=model_name,
            model_path=model_path,
            device=device,
            default_charge=self.default_charge,
            default_spin=self.default_spin,
        )

    def parse_constraints(
        self,
        constraint_specs: Union[str, List, Dict],
        atoms: Atoms,
        verbose: bool = False,
    ) -> List:
        """
        Parse constraint specifications and return ASE-compatible constraints.

        Supports simplified constraint parsing with two core types:
        1. Fixed Atoms: Exactly fix atom positions
        2. Harmonic Constraints: Soft constraints based on initial geometry

        Parameters:
        - constraint_specs: Constraint specifications in various formats
        - atoms: Atoms object (used as reference geometry)
        - verbose: Print constraint information

        Supported string formats:
        - "fix 0,1,2,3": Fix atoms at indices 0,1,2,3
        - "harmonic_position 5,6 k=10.0": Harmonic position constraint for atoms 5,6
        - "harmonic_bond 0,1 k=5.0": Harmonic bond constraint between atoms 0,1
        - "harmonic_angle 0,1,2 k=2.0": Harmonic angle constraint for atoms 0,1,2

        Multiple constraints can be separated by semicolons:
        - "fix 0,1; harmonic_bond 2,3 k=5.0"

        Returns:
            List of ASE-compatible constraint objects
        """
        from .constraints import parse_constraint_string

        if constraint_specs is None:
            return []

        # Handle different input formats
        if isinstance(constraint_specs, str):
            constraint_manager = parse_constraint_string(constraint_specs, atoms)
        elif isinstance(constraint_specs, list):
            # Handle list of pre-made ASE constraints
            if all(
                hasattr(c, "__call__") or hasattr(c, "adjust_positions")
                for c in constraint_specs
            ):
                # Already ASE constraints
                return constraint_specs
            else:
                # List of constraint specifications to parse
                constraint_strings = []
                for spec in constraint_specs:
                    if isinstance(spec, str):
                        constraint_strings.append(spec)
                    else:
                        raise ValueError(
                            f"Unsupported constraint specification in list: {spec}"
                        )

                constraint_str = "; ".join(constraint_strings)
                constraint_manager = parse_constraint_string(constraint_str, atoms)
        else:
            raise ValueError(
                f"Unsupported constraint specification type: {type(constraint_specs)}"
            )

        # Apply constraints to atoms and get ASE constraint list
        constraint_manager.apply_constraints(atoms)

        if verbose:
            info = constraint_manager.get_constraint_info()
            print("Applied constraints:")
            if info["fixed_atoms"]:
                print(f"  Fixed atoms: {info['fixed_atoms']}")
            for hc in info["harmonic_constraints"]:
                print(
                    f"  Harmonic {hc['type']}: atoms {hc['atoms']}, "
                    f"k={hc['force_constant']}, ref={hc['reference_value']:.3f}"
                )

        return atoms.constraints

    def optimize_with_constraints(
        self,
        atoms: Optional[Atoms] = None,
        constraints: Optional[Union[str, List]] = None,
        optimizer: str = "BFGS",
        fmax: float = 0.01,
        steps: int = 200,
        verbose: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Optimize structure with enhanced constraint handling.

        This method provides simplified constraint specification using the
        enhanced constraint system while maintaining compatibility with
        existing ASE constraints.

        Parameters:
        - atoms: Structure to optimize (used as reference for harmonic constraints)
        - constraints: Constraint specifications (string, list, or ASE constraints)
        - optimizer: Optimization algorithm
        - fmax: Force convergence criterion
        - steps: Maximum optimization steps
        - verbose: Print detailed constraint information
        - **kwargs: Additional arguments passed to optimize_minimum

        Returns:
            Dictionary with optimization results and constraint information
        """

        if atoms is None:
            if self.atoms is None:
                raise ValueError("No structure loaded. Use load_structure() first.")
            atoms = self.atoms.copy()
        else:
            atoms = atoms.copy()

        # Parse and apply constraints
        parsed_constraints = []
        if constraints:
            parsed_constraints = self.parse_constraints(constraints, atoms, verbose)

        # Run optimization using existing optimize_minimum method
        results = self.optimize_minimum(
            atoms=atoms,
            optimizer=optimizer,
            fmax=fmax,
            steps=steps,
            constraints=parsed_constraints,
            **kwargs,
        )

        # Add constraint information to results
        from .constraints import get_constraint_summary

        results["constraint_info"] = get_constraint_summary(atoms)

        return results

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

    def calculate_frequencies(
        self,
        atoms: Optional[Atoms] = None,
        delta: float = 0.01,
        method: str = "auto",
        nfree: Optional[int] = None,
        temperature: float = 298.15,
        save_hessian: bool = True,
        indices: Optional[List[int]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Calculate vibrational frequencies and normal modes.

        Parameters:
        -----------
        atoms : Atoms, optional
            Structure to analyze. Uses self.atoms if None.
        delta : float
            Displacement for finite differences (Å)
        method : str
            Hessian calculation method: 'auto', 'direct', or 'finite_differences'
        nfree : int, optional
            Number of degrees of freedom to remove. Auto-determined if None.
        temperature : float
            Temperature for thermodynamic properties (K)
        save_hessian : bool
            Whether to save Hessian matrix in results
        indices : List[int], optional
            Indices of atoms to include. All atoms if None.
        **kwargs
            Additional arguments for FrequencyAnalysis

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing:
            - frequencies: List of frequencies in cm^-1
            - normal_modes: Normal mode vectors
            - hessian: Hessian matrix (if save_hessian=True)
            - is_ts: Boolean indicating if structure is transition state
            - zero_point_energy: Zero-point vibrational energy
            - thermodynamic_properties: Dict with entropy, heat capacity, etc.
        """
        from .frequency import FrequencyAnalysis

        if atoms is None:
            if self.atoms is None:
                raise ValueError("No structure loaded. Use load_structure() first.")
            atoms = self.atoms.copy()
        else:
            atoms = atoms.copy()

        atoms.calc = self.calculator

        # Initialize frequency analysis
        freq_analysis = FrequencyAnalysis(
            atoms=atoms,
            calculator=self.calculator,
            delta=delta,
            nfree=nfree,
            indices=indices,
        )

        # Calculate Hessian and frequencies
        print(f"Calculating frequencies using {method} method...")
        hessian = freq_analysis.calculate_hessian(method=method)
        frequencies, normal_modes = freq_analysis.diagonalize_hessian()

        # Get vibrational frequencies (excluding trans/rot modes)
        vib_frequencies = freq_analysis.get_frequencies()
        vib_normal_modes = freq_analysis.get_normal_modes()

        # Calculate thermodynamic properties
        thermo_props = freq_analysis.get_thermodynamic_properties(temperature)

        # Transition state verification
        ts_analysis = freq_analysis.is_transition_state()

        # Prepare results
        results = {
            "frequencies": vib_frequencies.tolist(),
            "all_frequencies": frequencies.tolist(),  # Including trans/rot modes
            "normal_modes": vib_normal_modes.tolist(),
            "zero_point_energy": freq_analysis.get_zero_point_energy(),
            "thermodynamic_properties": thermo_props,
            "ts_analysis": ts_analysis,
            "is_ts": ts_analysis["is_transition_state"],
            "method_used": method,
            "delta": delta,
            "temperature": temperature,
            "n_atoms": len(atoms),
            "indices": indices if indices is not None else list(range(len(atoms))),
        }

        if save_hessian:
            results["hessian"] = hessian.tolist()

        self.results["frequency_analysis"] = results
        print(
            f"Frequency analysis completed. Found {len(vib_frequencies)} vibrational modes."
        )

        return results

    def verify_transition_state(
        self, atoms: Optional[Atoms] = None, freq_threshold: float = 50.0, **freq_kwargs
    ) -> Dict[str, Any]:
        """
        Verify that a structure is a transition state by checking frequencies.

        Parameters:
        -----------
        atoms : Atoms, optional
            Structure to verify. Uses self.atoms if None.
        freq_threshold : float
            Minimum frequency magnitude in cm^-1 to consider significant
        **freq_kwargs
            Additional arguments passed to calculate_frequencies

        Returns:
        --------
        Dict[str, Any]
            Dictionary with verification results including number of imaginary
            frequencies and their values.
        """
        from .frequency import FrequencyAnalysis

        if atoms is None:
            if self.atoms is None:
                raise ValueError("No structure loaded. Use load_structure() first.")
            atoms = self.atoms.copy()
        else:
            atoms = atoms.copy()

        atoms.calc = self.calculator

        # Get frequency analysis parameters
        delta = freq_kwargs.get("delta", 0.01)
        method = freq_kwargs.get("method", "auto")
        nfree = freq_kwargs.get("nfree", None)
        indices = freq_kwargs.get("indices", None)

        print("Verifying transition state by frequency analysis...")

        # Initialize frequency analysis
        freq_analysis = FrequencyAnalysis(
            atoms=atoms,
            calculator=self.calculator,
            delta=delta,
            nfree=nfree,
            indices=indices,
        )

        # Calculate frequencies
        freq_analysis.calculate_hessian(method=method)
        freq_analysis.diagonalize_hessian()

        # Verify transition state
        ts_results = freq_analysis.is_transition_state(threshold=freq_threshold)

        # Add summary information
        ts_results.update(
            {
                "structure_verified": True,
                "method_used": method,
                "freq_threshold": freq_threshold,
                "verification_summary": self._format_ts_verification(ts_results),
            }
        )

        self.results["ts_verification"] = ts_results

        return ts_results

    def _format_ts_verification(self, ts_results: Dict[str, Any]) -> str:
        """Format TS verification results as readable summary."""
        n_imag = ts_results["n_imaginary_frequencies"]
        assessment = ts_results["assessment"]

        summary = f"Structure Assessment: {assessment}\n"
        summary += f"Number of imaginary frequencies: {n_imag}\n"

        if n_imag > 0:
            imag_freqs = ts_results["imaginary_frequencies"]
            freq_str = ", ".join([f"{f:.1f}" for f in imag_freqs])
            summary += f"Imaginary frequencies: {freq_str} cm⁻¹\n"

        if ts_results["is_transition_state"]:
            summary += "✓ Structure is a valid transition state (exactly one imaginary frequency)"
        elif n_imag == 0:
            summary += "✓ Structure is a minimum (no imaginary frequencies)"
        else:
            summary += (
                f"⚠ Structure is a higher-order saddle point "
                f"({n_imag} imaginary frequencies)"
            )

        return summary

    def calculate_reaction_thermodynamics(
        self,
        reactant_atoms: Atoms,
        product_atoms: Atoms,
        ts_atoms: Optional[Atoms] = None,
        temperature: float = 298.15,
        **freq_kwargs,
    ) -> Dict[str, Any]:
        """
        Calculate reaction thermodynamics including activation barriers,
        reaction energies, and rate constants.

        Parameters:
        -----------
        reactant_atoms : Atoms
            Optimized reactant structure
        product_atoms : Atoms
            Optimized product structure
        ts_atoms : Atoms, optional
            Optimized transition state structure
        temperature : float
            Temperature for analysis (K)
        **freq_kwargs
            Additional arguments for frequency calculations

        Returns:
        --------
        Dict[str, Any]
            Dictionary with reaction thermodynamics
        """
        from .frequency import FrequencyAnalysis

        print(f"Calculating reaction thermodynamics at {temperature} K...")

        # Ensure all structures have the calculator attached
        reactant_atoms = reactant_atoms.copy()
        product_atoms = product_atoms.copy()
        reactant_atoms.calc = self.calculator
        product_atoms.calc = self.calculator

        results = {
            "temperature": temperature,
            "has_transition_state": ts_atoms is not None,
        }

        # Calculate properties for reactant
        print("Analyzing reactant...")
        reactant_freq = FrequencyAnalysis(
            reactant_atoms, self.calculator, **freq_kwargs
        )
        reactant_freq.calculate_hessian()
        reactant_freq.diagonalize_hessian()

        reactant_energy = reactant_atoms.get_potential_energy()
        reactant_zpe = reactant_freq.get_zero_point_energy()
        reactant_thermo = reactant_freq.get_thermodynamic_properties(temperature)

        results["reactant"] = {
            "electronic_energy": reactant_energy,
            "zero_point_energy": reactant_zpe,
            "thermodynamic_properties": reactant_thermo,
            "total_energy": reactant_energy + reactant_zpe,
            "frequencies": reactant_freq.get_frequencies().tolist(),
        }

        # Calculate properties for product
        print("Analyzing product...")
        product_freq = FrequencyAnalysis(product_atoms, self.calculator, **freq_kwargs)
        product_freq.calculate_hessian()
        product_freq.diagonalize_hessian()

        product_energy = product_atoms.get_potential_energy()
        product_zpe = product_freq.get_zero_point_energy()
        product_thermo = product_freq.get_thermodynamic_properties(temperature)

        results["product"] = {
            "electronic_energy": product_energy,
            "zero_point_energy": product_zpe,
            "thermodynamic_properties": product_thermo,
            "total_energy": product_energy + product_zpe,
            "frequencies": product_freq.get_frequencies().tolist(),
        }

        # Calculate reaction energies
        results["reaction_energy"] = {
            "electronic": product_energy - reactant_energy,
            "zero_point_corrected": (product_energy + product_zpe)
            - (reactant_energy + reactant_zpe),
            "enthalpy": (
                product_thermo["internal_energy"] - reactant_thermo["internal_energy"]
            ),
            "free_energy": (
                (
                    product_thermo["internal_energy"]
                    - temperature * product_thermo["entropy"]
                )
                - (
                    reactant_thermo["internal_energy"]
                    - temperature * reactant_thermo["entropy"]
                )
            ),
        }

        # Calculate properties for transition state if provided
        if ts_atoms is not None:
            ts_atoms = ts_atoms.copy()
            ts_atoms.calc = self.calculator

            print("Analyzing transition state...")
            ts_freq = FrequencyAnalysis(ts_atoms, self.calculator, **freq_kwargs)
            ts_freq.calculate_hessian()
            ts_freq.diagonalize_hessian()

            ts_energy = ts_atoms.get_potential_energy()
            ts_zpe = ts_freq.get_zero_point_energy()
            ts_thermo = ts_freq.get_thermodynamic_properties(temperature)
            ts_verification = ts_freq.is_transition_state()

            results["transition_state"] = {
                "electronic_energy": ts_energy,
                "zero_point_energy": ts_zpe,
                "thermodynamic_properties": ts_thermo,
                "total_energy": ts_energy + ts_zpe,
                "frequencies": ts_freq.get_frequencies().tolist(),
                "is_valid_ts": ts_verification["is_transition_state"],
                "imaginary_frequency": (
                    ts_verification["imaginary_frequencies"][0]
                    if ts_verification["imaginary_frequencies"]
                    else None
                ),
            }

            # Calculate activation barriers
            results["activation_energy"] = {
                "electronic": ts_energy - reactant_energy,
                "zero_point_corrected": (ts_energy + ts_zpe)
                - (reactant_energy + reactant_zpe),
                "enthalpy": ts_thermo["internal_energy"]
                - reactant_thermo["internal_energy"],
                "free_energy": (
                    (ts_thermo["internal_energy"] - temperature * ts_thermo["entropy"])
                    - (
                        reactant_thermo["internal_energy"]
                        - temperature * reactant_thermo["entropy"]
                    )
                ),
            }

            # Estimate rate constant using transition state theory
            if ts_verification["is_transition_state"]:
                results["rate_constant"] = self._calculate_rate_constant(
                    results["activation_energy"]["free_energy"], temperature
                )

        self.results["reaction_thermodynamics"] = results
        print("Reaction thermodynamics analysis completed.")

        return results

    def _calculate_rate_constant(
        self, delta_g_act: float, temperature: float
    ) -> Dict[str, float]:
        """Calculate rate constant using transition state theory."""
        import math

        # Constants
        kB = 8.617333e-5  # eV/K
        h = 4.135667e-15  # eV·s

        # TST rate constant: k = (kB*T/h) * exp(-ΔG‡/kB*T)
        prefactor = kB * temperature / h  # in s^-1
        exponential = math.exp(-delta_g_act / (kB * temperature))
        rate_constant = prefactor * exponential

        return {
            "rate_constant_s-1": rate_constant,
            "prefactor_s-1": prefactor,
            "activation_free_energy_eV": delta_g_act,
            "temperature_K": temperature,
        }

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
    **optimizer_kwargs,
) -> Atoms:
    """Convenience function to minimize a molecular structure.

    This function creates a QMEOptimizer and runs a geometry optimization,
    returning the optimized structure.

    Args:
        atoms: Input molecular structure to optimize.
        backend: ML backend to use ('uma', 'so3lr', 'aimnet2', 'mace', 'mock').
        optimizer: Optimizer to use ('BFGS', 'LBFGS', 'FIRE').
        fmax: Force convergence criterion (eV/Å).
        steps: Maximum optimization steps.
        logfile: Optional log file for optimization output.
        trajectory: Optional trajectory file to save optimization steps.
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
    qme_opt = QMEOptimizer(backend=backend)

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
