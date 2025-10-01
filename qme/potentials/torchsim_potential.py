"""
TorchSim Machine Learning Potential integration for ASE.

This module implements TorchSim calculator integration for supported ML potentials,
providing significant speedup over traditional ASE-based implementations through
automatic batching and efficient GPU memory management.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from ase import Atoms
from ase.calculators.calculator import all_changes

from qme.dependencies import deps
from qme.potentials.base_potential import BasePotential


class TorchSimPotential(BasePotential):
    """
    TorchSim potential calculator for supported ML models.

    This calculator provides access to TorchSim's accelerated ML potential
    implementations, offering significant speedup over ASE-based approaches
    through automatic batching and efficient GPU memory management.

    Supported models:
    - MACE (via TorchSim's MACE integration)
    - Fairchem models (UMA, etc.)
    - Other TorchSim-supported models
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        backend: str = "mace",  # Default to MACE
        default_charge: int = 0,
        default_spin: int = 1,
        **kwargs,
    ):
        """
        Initialize TorchSim potential calculator.

        Parameters:
        -----------
        model_name : str, optional
            Model name to use. Defaults to backend-specific default
        device : str, optional
            Device to run computations on ('cpu', 'cuda'). Auto-detected if None.
        backend : str
            Backend to use ('mace', 'fairchem', etc.)
        default_charge : int
            Default charge to use if not specified in atoms.info (default: 0)
        default_spin : int
            Default spin multiplicity to use if not specified in atoms.info (default: 1)
        **kwargs : dict
            Additional arguments passed to BasePotential
        """
        if not deps.has("torch_sim"):
            raise ImportError(
                "TorchSim is required for TorchSimPotential. "
                "Install with: pip install torch-sim-atomistic"
            )

        if not deps.has("torch"):
            raise ImportError(
                "PyTorch is required for TorchSimPotential. "
                "Install with: pip install torch"
            )

        # Set device if not provided
        if device is None:
            torch = deps.get("torch")
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize base class
        super().__init__(model_name=model_name, device=device, **kwargs)

        # TorchSim-specific attributes
        self.backend = backend
        self.default_charge = default_charge
        self.default_spin = default_spin
        self._torch_sim = None
        self._model = None
        self._state = None

    def _load_calculator(self):
        """Load the TorchSim model and setup."""
        from qme.potentials.logging_utils import quiet_backend_loading

        if self._torch_sim is not None:
            return  # Already loaded

        try:
            self._torch_sim = deps.get("torch_sim")
            torch = deps.get("torch")

            with quiet_backend_loading(
                "torchsim", self.model_name or self.backend, None, self.device
            ):
                if self.backend.lower() == "mace":
                    self._load_mace_model()
                elif self.backend.lower() in ["fairchem", "uma"]:
                    self._load_fairchem_model()
                else:
                    raise ValueError(f"Unsupported TorchSim backend: {self.backend}")

        except Exception as e:
            deps.warn_fallback(
                "torchsim",
                f"TorchSim not available ({e}). Install with: pip install torch-sim-atomistic",
            )
            # Fall back to mock calculator
            from qme.potentials import MockCalculator

            self._calc = MockCalculator(backend="torchsim")

    def _load_mace_model(self):
        """Load MACE model through TorchSim."""
        if self.model_name is None:
            self.model_name = "mace-omol-0"

        # Load MACE model through TorchSim
        from mace.calculators.foundations_models import mace_mp, mace_off, mace_omol
        from torch_sim.models.mace import MaceModel

        if self.model_name == "mace-omol-0":
            mace_model = mace_omol(return_raw_model=True)
        elif self.model_name.startswith("mace-mp"):
            model_size = self.model_name.replace("mace-mp-", "") or "medium"
            mace_model = mace_mp(model=model_size, return_raw_model=True)
        elif self.model_name.startswith("mace-off"):
            model_size = self.model_name.replace("mace-off-", "") or "medium"
            mace_model = mace_off(model=model_size, return_raw_model=True)
        else:
            # Default to MACE-OMOL
            mace_model = mace_omol(return_raw_model=True)

        # Create TorchSim MACE model
        torch_device = deps.get("torch").device(self.device)
        mace_model = mace_model.to(torch_device)
        self._mace_model = MaceModel(model=mace_model, device=torch_device)

        # Monkey-patch both the TorchSim wrapper and the underlying MACE model
        original_macemodel_forward = self._mace_model.forward
        original_mace_forward = self._mace_model.model.forward

        def patched_macemodel_forward(state, *args, **kwargs):
            import torch

            # Enable gradients on positions before processing
            if hasattr(state, "positions") and not state.positions.requires_grad:
                state.positions.requires_grad_(True)
            return original_macemodel_forward(state, *args, **kwargs)

        def patched_mace_forward(data, *args, **kwargs):
            import torch

            # Get atoms object from the calculator instance
            atoms = getattr(self, "_current_atoms", None)

            # Add total_spin and total_charge if not present
            # Use the same dtype as positions to ensure consistency
            dtype = data["positions"].dtype if "positions" in data else torch.float64

            if "total_spin" not in data:
                # Get spin from atoms.info or use default
                if atoms is not None and "spin" in atoms.info:
                    spin_value = float(atoms.info["spin"])
                else:
                    spin_value = float(self.default_spin)
                data["total_spin"] = torch.tensor(
                    [spin_value], device=torch_device, dtype=dtype
                )

            if "total_charge" not in data:
                # Get charge from atoms.info or use default
                if atoms is not None and "charge" in atoms.info:
                    charge_value = float(atoms.info["charge"])
                else:
                    charge_value = float(self.default_charge)
                data["total_charge"] = torch.tensor(
                    [charge_value], device=torch_device, dtype=dtype
                )

            # Enable gradients for positions if not already enabled
            if "positions" in data and not data["positions"].requires_grad:
                data["positions"].requires_grad_(True)
            return original_mace_forward(data, *args, **kwargs)

        self._mace_model.forward = patched_macemodel_forward
        self._mace_model.model.forward = patched_mace_forward
        self._model = self._mace_model

    def _load_fairchem_model(self):
        """Load Fairchem model through TorchSim."""
        if self.model_name is None:
            self.model_name = "equiformer_v2_31M_s2ef_all_md"

        try:
            # Try to load Fairchem model through TorchSim
            from fairchem.core.models.model_registry import model_name_to_local_file
            from torch_sim.models.fairchem import FairchemModel

            # Get model path
            model_path = model_name_to_local_file(self.model_name, local_cache_dir=".")
            torch_device = deps.get("torch").device(self.device)
            self._model = FairchemModel(model_path=model_path, device=torch_device)
        except ImportError as e:
            # If TorchSim Fairchem is not available, fall back to regular Fairchem
            deps.warn_fallback(
                "torchsim_fairchem",
                f"TorchSim Fairchem not available ({e}). Using regular Fairchem calculator.",
            )
            
            # Use regular Fairchem calculator as fallback
            from fairchem.core.pretrained_mlip import get_predict_unit
            from fairchem.core import FAIRChemCalculator
            
            # Load the model using regular Fairchem
            device_param = "cuda" if self.device == "cuda" else "cpu"
            predictor = get_predict_unit(self.model_name, device=device_param)
            self._model = FAIRChemCalculator(predictor, task_name="omol")

    def _atoms_to_state(self, atoms: Atoms):
        """Convert ASE Atoms to TorchSim state."""
        torch = deps.get("torch")
        device = torch.device(self.device)
        # Use float64 for MACE models (they expect float64)
        dtype = torch.float64 if self.backend.lower() == "mace" else torch.float32

        # For non-periodic systems with zero cell, set a large cell to avoid issues
        atoms_copy = atoms.copy()
        if not any(atoms.get_pbc()) and torch.allclose(
            torch.tensor(atoms.get_cell().array, dtype=dtype),
            torch.zeros(3, 3, dtype=dtype),
        ):
            # Center the molecule with vacuum
            atoms_copy.center(vacuum=10.0)

        if self._state is None:
            self._state = self._torch_sim.io.atoms_to_state(
                atoms_copy, device=device, dtype=dtype
            )
        else:
            # Update existing state
            self._state = self._torch_sim.io.atoms_to_state(
                atoms_copy, device=device, dtype=dtype
            )

        # Enable gradients for positions (needed for force calculations)
        self._state.positions.requires_grad_(True)

        # Return the SimState directly - the MACE model will convert it if needed
        return self._state

    def _state_to_atoms(self, state):
        """Convert TorchSim state back to ASE Atoms."""
        return self._torch_sim.io.state_to_atoms(state)

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        """Calculate properties using TorchSim."""
        # Common setup
        super().calculate(atoms, properties, system_changes)

        # Set default charge and spin if not already set to avoid warnings
        if "charge" not in self.atoms.info:
            self.atoms.info["charge"] = self.default_charge
        if "spin" not in self.atoms.info:
            self.atoms.info["spin"] = self.default_spin

        # Ensure calculator is loaded
        if self._torch_sim is None:
            self._load_calculator()

        if self._model is None:
            raise RuntimeError("Failed to load TorchSim model")

        # Check if we're using a TorchSim model or regular calculator
        if hasattr(self._model, 'forward'):
            # TorchSim model - use the state-based approach
            state = self._atoms_to_state(self.atoms)
            self._current_atoms = self.atoms
            
            # Calculate properties using TorchSim model
            torch = deps.get("torch")
            # Enable gradients for force calculations
            results = self._model(state)
        else:
            # Regular calculator (e.g., Fairchem fallback) - use direct calculation
            self._model.calculate(self.atoms, properties, system_changes)
            
            # Extract results from the calculator
            if "energy" in properties:
                try:
                    self.results["energy"] = self._model.results["energy"]
                except Exception:
                    self.results["energy"] = self.results.get("energy")

            if "forces" in properties:
                try:
                    self.results["forces"] = self._model.results["forces"]
                except Exception:
                    self.results["forces"] = self.results.get("forces")
            return

        if "energy" in properties and "energy" in results:
            energy = results["energy"]
            # Handle both single system and batched results
            if energy.dim() == 0:
                self.results["energy"] = float(energy)
            else:
                self.results["energy"] = float(
                    energy[0]
                )  # Take first system if batched

        if "forces" in properties and "forces" in results:
            forces = results["forces"]
            # Convert to numpy and ensure correct shape
            forces_np = forces.detach().cpu().numpy()
            self.results["forces"] = forces_np

    def get_potential_energy(self, atoms=None, force_consistent: bool = False):
        """Get potential energy."""
        if atoms is not None:
            self.atoms = atoms
        return super().get_potential_energy(atoms, force_consistent)

    def get_forces(self, atoms=None):
        """Get forces on atoms."""
        if atoms is not None:
            self.atoms = atoms
        return super().get_forces(atoms)

    def _get_backend_name(self) -> str:
        """Get the backend name for this calculator."""
        return f"torchsim_{self.backend}"


def get_torchsim_calculator(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    backend: str = "mace",
    **kwargs,
) -> TorchSimPotential:
    """
    Factory function to create TorchSim calculator.

    Parameters:
    -----------
    model_name : str, optional
        Model name to use
    device : str, optional
        Device for computations ('cpu', 'cuda')
    backend : str
        Backend to use ('mace', 'fairchem', etc.)
    **kwargs : dict
        Additional arguments passed to TorchSimPotential

    Returns:
    --------
    TorchSimPotential
        Configured TorchSim calculator instance

    Examples:
    ---------
    >>> calc = get_torchsim_calculator()  # Uses MACE-OMOL-0
    >>> calc = get_torchsim_calculator(backend="mace", model_name="mace-mp-medium")
    >>> calc = get_torchsim_calculator(backend="fairchem", model_name="equiformer_v2_31M_s2ef_all_md")
    """
    return TorchSimPotential(
        model_name=model_name, device=device, backend=backend, **kwargs
    )


def get_torchsim_mace_calculator(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs,
) -> TorchSimPotential:
    """Convenience function for TorchSim MACE calculator."""
    return get_torchsim_calculator(
        model_name=model_name, device=device, backend="mace", **kwargs
    )


def get_torchsim_fairchem_calculator(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs,
) -> TorchSimPotential:
    """Convenience function for TorchSim Fairchem calculator."""
    return get_torchsim_calculator(
        model_name=model_name, device=device, backend="fairchem", **kwargs
    )
