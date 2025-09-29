"""
Full optimizer implementation for QME core.

This file contains the QMEOptimizer class and helper functions previously
defined in the top-level `qme/core.py`. It was moved here to form a proper
subpackage layout while keeping behavior unchanged.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS, FIRE, LBFGS

from ..core.registry import calculator_registry
from ..dependencies import deps
from ..utils.settings import config, get_default_model


class QMEOptimizer:
    """Main optimizer class that combines ASE and SELLA optimizers with neural
    network potentials.

    Supports UMA, SO3LR, AIMNET2, and MACE backends.

    This class provides a unified interface for molecular geometry optimization
    using machine learning potentials, supporting both minimum energy
    optimization and transition state searches.
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

    if deps.has("sella"):
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

        Parameters mirror the previous top-level implementation.
        """

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
                from ..mock_calculator import MockCalculator

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
            device = config.get_device()

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
        from ..constraints import parse_constraint_string

        if constraint_specs is None:
            return []

        if isinstance(constraint_specs, str):
            constraint_manager = parse_constraint_string(constraint_specs, atoms)
        elif isinstance(constraint_specs, list):
            if all(
                hasattr(c, "__call__") or hasattr(c, "adjust_positions")
                for c in constraint_specs
            ):
                return constraint_specs
            else:
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
        if atoms is None:
            if self.atoms is None:
                raise ValueError("No structure loaded. Use load_structure() first.")
            atoms = self.atoms.copy()
        else:
            atoms = atoms.copy()

        parsed_constraints = []
        if constraints:
            parsed_constraints = self.parse_constraints(constraints, atoms, verbose)

        results = self.optimize_minimum(
            atoms=atoms,
            optimizer=optimizer,
            fmax=fmax,
            steps=steps,
            constraints=parsed_constraints,
            **kwargs,
        )

        from ..constraints import get_constraint_summary

        results["constraint_info"] = get_constraint_summary(atoms)

        return results

    def load_structure(self, structure_file: Union[str, Path]) -> Atoms:
        structure_file = Path(structure_file)

        if not structure_file.exists():
            raise FileNotFoundError(f"Structure file not found: {structure_file}")

        if structure_file.stat().st_size == 0:
            raise ValueError(f"Structure file is empty: {structure_file}")

        valid_extensions = {".xyz", ".cif", ".pdb", ".mol", ".sdf", ".cml", ".traj"}
        if structure_file.suffix.lower() not in valid_extensions:
            if config.enable_warnings:
                print(
                    f"Warning: Unrecognized file extension '{structure_file.suffix}'. "
                    f"Supported formats: {', '.join(valid_extensions)}"
                )

        try:
            atoms_result = read(structure_file)

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

            if len(atoms) == 0:
                raise ValueError(f"No atoms found in structure file: {structure_file}")

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
        initial_fmax_factor: float = 10.0,
        initial_steps_fraction: float = 0.5,
        constraints: Optional[List] = None,
        **optimizer_kwargs,
    ) -> Dict[str, Any]:
        if atoms is None:
            if self.atoms is None:
                raise ValueError("No structure loaded. Use load_structure() first.")
            atoms = self.atoms.copy()
        else:
            atoms = atoms.copy()

        atoms.calc = self.calculator

        if constraints:
            atoms.set_constraint(constraints)

        if optimizer not in self.AVAILABLE_OPTIMIZERS:
            available = list(self.AVAILABLE_OPTIMIZERS.keys())
            raise ValueError(f"Unknown optimizer: {optimizer}. Available: {available}")

        initial_energy = atoms.get_potential_energy()
        initial_forces = atoms.get_forces()
        initial_max_force = np.max(np.abs(initial_forces))

        total_steps_taken = 0
        converged = False
        optimizer_used = optimizer

        if initial_fmax_factor > 1.0 and initial_steps_fraction > 0:
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
                converged = np.max(np.abs(atoms.get_forces())) < fmax
        else:
            print(
                f"Optimizing with {optimizer} (fmax={fmax:.4f} eV/Å, steps={steps})..."
            )
            converged, total_steps_taken, _ = self._run_single_optimization(
                atoms, optimizer, fmax, steps, logfile, trajectory, **optimizer_kwargs
            )

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
        if not deps.has("sella"):
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

        if constraints:
            atoms.set_constraint(constraints)

        Sella = deps.require("sella", "transition state searches")
        opt = Sella(
            atoms,
            logfile=logfile,
            trajectory=trajectory,
            internal=True,
            order=1,
            **sella_kwargs,
        )

        initial_energy = atoms.get_potential_energy()
        initial_forces = atoms.get_forces()
        initial_max_force = np.max(np.abs(initial_forces))

        converged = opt.run(fmax=fmax, steps=steps)

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
        from ..frequency import FrequencyAnalysis

        if atoms is None:
            if self.atoms is None:
                raise ValueError("No structure loaded. Use load_structure() first.")
            atoms = self.atoms.copy()
        else:
            atoms = atoms.copy()

        atoms.calc = self.calculator

        freq_analysis = FrequencyAnalysis(
            atoms=atoms,
            calculator=self.calculator,
            delta=delta,
            nfree=nfree,
            indices=indices,
        )

        print(f"Calculating frequencies using {method} method...")
        hessian = freq_analysis.calculate_hessian(method=method)
        frequencies, normal_modes = freq_analysis.diagonalize_hessian()

        vib_frequencies = freq_analysis.get_frequencies()
        vib_normal_modes = freq_analysis.get_normal_modes()

        thermo_props = freq_analysis.get_thermodynamic_properties(temperature)

        ts_analysis = freq_analysis.is_transition_state()

        results = {
            "frequencies": vib_frequencies.tolist(),
            "all_frequencies": frequencies.tolist(),
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
        from ..frequency import FrequencyAnalysis

        if atoms is None:
            if self.atoms is None:
                raise ValueError("No structure loaded. Use load_structure() first.")
            atoms = self.atoms.copy()
        else:
            atoms = atoms.copy()

        atoms.calc = self.calculator

        delta = freq_kwargs.get("delta", 0.01)
        method = freq_kwargs.get("method", "auto")
        nfree = freq_kwargs.get("nfree", None)
        indices = freq_kwargs.get("indices", None)

        print("Verifying transition state by frequency analysis...")

        freq_analysis = FrequencyAnalysis(
            atoms=atoms,
            calculator=self.calculator,
            delta=delta,
            nfree=nfree,
            indices=indices,
        )

        freq_analysis.calculate_hessian(method=method)
        freq_analysis.diagonalize_hessian()

        ts_results = freq_analysis.is_transition_state(threshold=freq_threshold)

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
        from ..frequency import FrequencyAnalysis

        print(f"Calculating reaction thermodynamics at {temperature} K...")

        reactant_atoms = reactant_atoms.copy()
        product_atoms = product_atoms.copy()
        reactant_atoms.calc = self.calculator
        product_atoms.calc = self.calculator

        results = {
            "temperature": temperature,
            "has_transition_state": ts_atoms is not None,
        }

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

        # Compute reaction energetics; use numpy.errstate to avoid noisy runtime warnings
        import numpy as _np

        def _safe_get(dct, key):
            return float(dct.get(key, _np.nan))

        with _np.errstate(invalid="ignore", divide="ignore"):
            electronic = float(product_energy - reactant_energy)
            zero_point_corrected = float((product_energy + product_zpe) - (reactant_energy + reactant_zpe))
            enthalpy = float(_safe_get(product_thermo, "internal_energy") - _safe_get(reactant_thermo, "internal_energy"))
            free_energy = float((
                _safe_get(product_thermo, "internal_energy") - temperature * _safe_get(product_thermo, "entropy")
            ) - (
                _safe_get(reactant_thermo, "internal_energy") - temperature * _safe_get(reactant_thermo, "entropy")
            ))

        results["reaction_energy"] = {
            "electronic": electronic,
            "zero_point_corrected": zero_point_corrected,
            "enthalpy": enthalpy,
            "free_energy": free_energy,
        }

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

            # Compute activation energetics with safe numeric guards
            with _np.errstate(invalid="ignore", divide="ignore"):
                electronic_act = float(ts_energy - reactant_energy)
                zpe_act = float((ts_energy + ts_zpe) - (reactant_energy + reactant_zpe))
                enthalpy_act = float(_safe_get(ts_thermo, "internal_energy") - _safe_get(reactant_thermo, "internal_energy"))
                free_energy_act = float((
                    _safe_get(ts_thermo, "internal_energy") - temperature * _safe_get(ts_thermo, "entropy")
                ) - (
                    _safe_get(reactant_thermo, "internal_energy") - temperature * _safe_get(reactant_thermo, "entropy")
                ))

            results["activation_energy"] = {
                "electronic": electronic_act,
                "zero_point_corrected": zpe_act,
                "enthalpy": enthalpy_act,
                "free_energy": free_energy_act,
            }

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
        import math

        kB = 8.617333e-5
        h = 4.135667e-15

        prefactor = kB * temperature / h
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
        output_file = Path(output_file)

        try:
            if format is not None:
                write(output_file, atoms, format=format)
            else:
                write(output_file, atoms)
        except Exception as e:
            raise RuntimeError(f"Failed to save structure to {output_file}: {e}")

    def get_optimization_summary(self) -> str:
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
    available_optimizers = ["BFGS", "LBFGS", "FIRE"]
    if deps.has("sella"):
        available_optimizers.append("Sella")

    if optimizer not in available_optimizers:
        raise ImportError(f"Optimizer {optimizer} not available")

    qme_opt = QMEOptimizer(backend=backend)

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
