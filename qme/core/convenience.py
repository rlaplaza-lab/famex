"""Convenience functions for QME."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from ase import Atoms
from ase.io import write

from qme.analysis.frequency import FrequencyAnalysis
from qme.core.calculator_setup import create_calculator
from qme.core.constraint_parser import parse_constraints as core_parse_constraints

# QMEOptimizer is now an alias for QMEAdapter for backward compatibility
from qme.core.explorer import Explorer
from qme.core.geometry import read_geometry
from qme.core.local_strategies import local_ts_runner


class QMEAdapter:
    """Compatibility adapter that exposes a small legacy QMEOptimizer-like API
    while delegating behavior to the new Explorer class.

    This keeps the CLI and other callers backward compatible while the
    internal Explorer implementation remains the canonical runner.
    """

    # Class attributes for backward compatibility
    AVAILABLE_OPTIMIZERS = ["BFGS", "LBFGS", "FIRE", "sella"]
    AVAILABLE_BACKENDS = ["uma", "so3lr", "aimnet2", "mace", "mock"]

    def __init__(
        self,
        backend: str = "mock",
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        default_charge: int = 0,
        default_spin: int = 1,
        **kwargs,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.default_charge = default_charge
        self.default_spin = default_spin

        self.explorer: Optional[Explorer] = None
        self.atoms = None
        self.calculator = None

    def _ensure_explorer_for_atoms(self, atoms):
        if self.explorer is None or self.explorer.atoms_list != [atoms]:
            self.explorer = Explorer(
                atoms,
                backend=self.backend,
                model_name=self.model_name,
                model_path=self.model_path,
                device=self.device,
                default_charge=self.default_charge,
                default_spin=self.default_spin,
            )
        try:
            calc = self.explorer._create_and_attach_calculator(atoms)
        except Exception:
            calc = create_calculator(
                backend=self.backend,
                model_name=self.model_name,
                model_path=self.model_path,
                device=self.device,
                default_charge=self.default_charge,
                default_spin=self.default_spin,
            )
            atoms.calc = calc
        self.atoms = atoms
        self.calculator = calc
        return calc

    def load_structure(self, filename_or_geom):
        if isinstance(filename_or_geom, str):
            geom = read_geometry(filename_or_geom)
        else:
            geom = filename_or_geom

        if isinstance(geom, list):
            geom = geom[0]

        self._ensure_explorer_for_atoms(geom)
        return geom

    def optimize_minimum(
        self, atoms=None, optimizer=None, fmax=0.05, steps=1000, **kwargs
    ) -> Dict[str, Any]:
        from qme.core.validation import (
            validate_atoms_structure,
            validate_optimization_parameters,
        )

        # Handle backward compatibility: if atoms is passed, load it
        if atoms is not None:
            self.load_structure(atoms)

        if self.atoms is None:
            raise RuntimeError(
                "No structure loaded. Call load_structure() first or pass atoms parameter."
            )

        # Validate input parameters
        local_opt_name = (optimizer or "sella").lower()
        validate_optimization_parameters(fmax, steps, local_opt_name)
        validate_atoms_structure(self.atoms, "optimization")

        # Allow callers to pass optimizer_kwargs inside kwargs
        self.explorer.optimizer_kwargs = kwargs.get("optimizer_kwargs", {})
        res = self.explorer.run(
            mode="minima", fmax=fmax, steps=steps, local_optimizer_name=local_opt_name
        )
        optimized = res[0] if isinstance(res, list) and len(res) == 1 else res
        energy = None
        try:
            energy = float(optimized.get_potential_energy())
        except Exception:
            energy = None
        try:
            forces = optimized.get_forces()
            max_force = float(abs(forces).max())
        except Exception:
            max_force = None

        return {
            "converged": True,
            "final_energy": energy,
            "steps_taken": None,
            "optimized_atoms": optimized,
            "max_force": max_force,
        }

    def find_transition_state(
        self, atoms=None, fmax=0.05, steps=1000, **kwargs
    ) -> Dict[str, Any]:
        from qme.core.validation import (
            validate_atoms_structure,
            validate_optimization_parameters,
        )

        # Handle backward compatibility: if atoms is passed, load it
        if atoms is not None:
            self.load_structure(atoms)

        if self.atoms is None:
            raise RuntimeError(
                "No structure loaded. Call load_structure() first or pass atoms parameter."
            )

        # Validate input parameters
        local_opt_name = kwargs.get(
            "local_optimizer_name",
            self.explorer.local_optimizer_name if self.explorer else "sella",
        )
        validate_optimization_parameters(fmax, steps, local_opt_name)
        validate_atoms_structure(self.atoms, "transition state search")

        if self.explorer is None:
            self._ensure_explorer_for_atoms(self.atoms)
        if "ts" not in getattr(self.explorer, "_strategies", {}):
            self.explorer.ts_method = kwargs.get("ts_method", "dimer")
            self.explorer.ts_kwargs = kwargs.get("ts_kwargs", {}) or {}
            self.explorer.register_strategy(
                "ts", local_ts_runner, strategy_type="local"
            )
        res = self.explorer.run(
            mode="ts",
            fmax=fmax,
            steps=steps,
            local_optimizer_name=kwargs.get(
                "local_optimizer_name", self.explorer.local_optimizer_name
            ),
        )
        optimized = res[0] if isinstance(res, list) and len(res) == 1 else res
        energy = None
        try:
            energy = float(optimized.get_potential_energy())
        except Exception:
            energy = None
        try:
            forces = optimized.get_forces()
            max_force = float(abs(forces).max())
        except Exception:
            max_force = None

        return {
            "converged": True,
            "final_energy": energy,
            "steps_taken": None,
            "optimized_atoms": optimized,
            "ts_atoms": optimized,  # Alias for backward compatibility
            "max_force": max_force,
        }

    def ts_opt(self, atoms=None, fmax=0.05, steps=1000, **kwargs) -> Dict[str, Any]:
        """Alias for find_transition_state for backward compatibility."""
        if atoms is not None:
            self.load_structure(atoms)
        return self.find_transition_state(fmax=fmax, steps=steps, **kwargs)

    def save_structure(
        self, atoms: Atoms, output_file: Union[str, Path], format: Optional[str] = None
    ):
        """Save structure to file."""
        return save_structure(atoms, output_file, format)

    def calculate_frequencies(
        self,
        atoms=None,
        delta=0.01,
        method="auto",
        temperature=298.15,
        save_hessian=True,
        indices=None,
        **kwargs,
    ) -> Dict[str, Any]:
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise RuntimeError("No structure available for frequency calculation")
        if getattr(atoms, "calc", None) is None:
            self._ensure_explorer_for_atoms(atoms)

        freq_analysis = FrequencyAnalysis(
            atoms=atoms, calculator=atoms.calc, delta=delta, indices=indices
        )
        hessian = freq_analysis.calculate_hessian(method=method)
        frequencies, normal_modes = freq_analysis.diagonalize_hessian()
        vib_frequencies = freq_analysis.get_frequencies()
        vib_normal_modes = freq_analysis.get_normal_modes()
        thermo = freq_analysis.get_thermodynamic_properties(temperature)
        ts_analysis = freq_analysis.is_transition_state()

        results = {
            "frequencies": vib_frequencies.tolist(),
            "all_frequencies": frequencies.tolist(),
            "normal_modes": vib_normal_modes.tolist(),
            "zero_point_energy": freq_analysis.get_zero_point_energy(),
            "thermodynamic_properties": thermo,
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

        return results


def setup_optimization(
    input_file,
    backend,
    model,
    model_path,
    device,
    constraint_atoms,
    verbose,
    charge=0,
    spin=1,
    geometry=None,
) -> Tuple[QMEAdapter, Optional[list]]:
    """Backward-compatible helper used by the CLI.

    Returns a QMEAdapter and parsed constraints (or None).
    """
    adapter = QMEAdapter(
        backend=backend,
        model_name=model,
        model_path=model_path,
        device=device,
        default_charge=charge,
        default_spin=spin,
    )

    atoms = None
    if geometry is not None:
        adapter._ensure_explorer_for_atoms(geometry)
        atoms = geometry
    elif input_file:
        atoms = adapter.load_structure(input_file)

    constraints = None
    if atoms is not None:
        constraints = core_parse_constraints(constraint_atoms, atoms, verbose=verbose)

    return adapter, constraints


def save_structure(
    atoms: Atoms, output_file: Union[str, Path], format: Optional[str] = None
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


# Backward compatibility alias
QMEOptimizer = QMEAdapter
