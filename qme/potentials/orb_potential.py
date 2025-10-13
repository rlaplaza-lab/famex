"""
Orb Machine Learning Potential integration for ASE.

This module implements integration with Orbital Materials' Orb models,
providing universal forcefields for molecular and materials calculations.
"""

from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import all_changes

from qme.dependencies import deps
from qme.potentials.base_potential import BasePotential


class OrbPotential(BasePotential):
    """ASE Calculator interface for Orb neural network potential.

    Orb provides universal neural network potentials for molecular and materials
    property prediction and geometry optimization. This implementation uses the
    OrbMol variant which requires charge and spin multiplicity specification.

    Parameters
    ----------
    model_name : str, default "orb-v3-conservative-omol"
        Name of Orb model to use. Available models:
        - "orb-v3-conservative-omol": Conservative molecular model (default)
        - "orb-v3-conservative-inf-omat": Inference materials model
        - "orb-v2": Orb v2 model
    device : str, optional
        Device for computations ('cpu', 'cuda'). Auto-detected if None.
    charge : int, default 0
        Total charge of the system
    spin : int, default 1
        Spin multiplicity (2S + 1)
    **kwargs
        Additional arguments passed to BasePotential
    """

    def __init__(
        self,
        model_name: str = "orb-v3-conservative-omol",
        device: str | None = None,
        charge: int = 0,
        spin: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize Orb potential calculator.

        Parameters
        ----------
        model_name : str, default "orb-v3-conservative-omol"
            Name of Orb model to use
        device : str, optional
            Device for computations ('cpu', 'cuda'). Auto-detected if None.
        charge : int, default 0
            Molecular charge
        spin : int, default 1
            Spin multiplicity (2S + 1)
        **kwargs
            Additional arguments passed to parent Calculator
        """

        # Check dependencies
        if not deps.has("orb_models"):
            raise ImportError(
                "orb-models is required for Orb potentials. " "Install with: pip install orb-models"
            )

        if not deps.has("torch"):
            raise ImportError(
                "PyTorch is required for Orb potentials. " "Install with: pip install torch"
            )

        # Set device if not provided
        if device is None:
            from qme.utils.device import get_optimal_device

            device = get_optimal_device()

        # Initialize base class
        super().__init__(
            model_name=model_name,
            device=device,
            implemented_properties=["energy", "forces"],
            **kwargs,
        )

        # Orb-specific attributes
        self.charge = charge
        self.spin = spin

        # Ensure results dict exists for ASE-style API
        self.results = {}

    def _load_calculator(self) -> None:
        """Load the Orb model and create calculator."""
        from qme.logging_utils import quiet_backend_loading

        try:
            # Import Orb modules
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator

            # Ensure model_name is not None
            if self.model_name is None:
                self.model_name = "orb-v3-conservative-omol"

            # Ensure device is not None
            if self.device is None:
                self.device = "cpu"

            # Map model names to pretrained functions
            model_registry = {
                "orb-v3-conservative-omol": pretrained.orb_v3_conservative_omol,
                "orb-v3-conservative-inf-omat": pretrained.orb_v3_conservative_inf_omat,
                "orb-v2": pretrained.orb_v2,
                # Add aliases
                "orb-v3-omol": pretrained.orb_v3_conservative_omol,
                "orb-v3-omat": pretrained.orb_v3_conservative_inf_omat,
                "omol": pretrained.orb_v3_conservative_omol,
                "omat": pretrained.orb_v3_conservative_inf_omat,
            }

            # Get the pretrained model loader function
            if self.model_name in model_registry:
                model_loader = model_registry[self.model_name]
            else:
                # Default to omol if unknown
                print(
                    f"Warning: Unknown Orb model '{self.model_name}', "
                    f"using default 'orb-v3-conservative-omol'"
                )
                model_loader = pretrained.orb_v3_conservative_omol
                self.model_name = "orb-v3-conservative-omol"

            # Load the pretrained forcefield
            with quiet_backend_loading(
                "orb", self.model_name, "pretrained", self.device, show_model_info=False
            ):
                orbff = model_loader(device=self.device)
                self._calc = ORBCalculator(orbff, device=self.device)

                # Disable PyTorch compilation to avoid tensor size assertion errors
                import torch

                if hasattr(torch._dynamo, "config"):
                    torch._dynamo.config.disable = True

        except Exception as e:
            raise RuntimeError(
                f"Failed to load Orb model '{self.model_name}'. "
                f"Error: {e}. Please check the model name or installation."
            )

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: Any = all_changes,
    ) -> None:
        """Calculate properties using Orb potential."""
        # Use self.atoms if atoms is None (standard ASE behavior)
        if atoms is None:
            atoms = self.atoms

        if atoms is None:
            raise ValueError("No atoms provided for calculation")

        # Set charge and spin in atoms.info for OrbMol models
        # This is required by the Orb calculator and must be done early
        atoms.info["charge"] = self.charge
        atoms.info["spin"] = self.spin

        # Ensure backend loaded
        if self._calc is None:
            self._load_calculator()

        # Attach calculator to atoms
        atoms.calc = self._calc

        # Calculate properties
        if "energy" in properties:
            energy = atoms.get_potential_energy()
            self.results["energy"] = float(energy)

        if "forces" in properties:
            forces = atoms.get_forces()
            self.results["forces"] = np.array(forces)

    def set_charge(self, charge: int) -> None:
        """Set molecular charge."""
        self.charge = charge

    def set_spin(self, spin: int) -> None:
        """Set spin multiplicity."""
        self.spin = spin

    def _get_backend_name(self) -> str:
        """Get the backend name for Orb."""
        return "orb"

    def get_potential_energy(
        self, atoms: Atoms | None = None, force_consistent: bool = False
    ) -> float:
        """Get potential energy (ASE-compatible)."""
        if atoms is not None:
            self.atoms = atoms

        # Ensure charge and spin are set in atoms.info
        if self.atoms is not None:
            self.atoms.info["charge"] = self.charge
            self.atoms.info["spin"] = self.spin

        # Ensure calculator is loaded
        return super().get_potential_energy(atoms, force_consistent)

    def get_forces(self, atoms: Atoms | None = None) -> np.ndarray | None:
        """Get forces (ASE-compatible)."""
        if atoms is not None:
            self.atoms = atoms

        # Ensure charge and spin are set in atoms.info
        if self.atoms is not None:
            self.atoms.info["charge"] = self.charge
            self.atoms.info["spin"] = self.spin

        return super().get_forces(atoms)


def get_orb_calculator(
    model_name: str = "orb-v3-conservative-omol",
    device: str | None = None,
    charge: int = 0,
    spin: int = 1,
    **kwargs: Any,
) -> OrbPotential:
    """
    Convenience function to get Orb calculator.

    Parameters:
    -----------
    model_name : str
        Name of Orb model to use
    device : str, optional
        Device for computations ('cpu', 'cuda')
    charge : int
        Molecular charge
    spin : int
        Spin multiplicity (2S + 1)
    **kwargs :
        Additional arguments passed to OrbPotential

    Returns:
    --------
    OrbPotential
        Configured Orb calculator
    """
    return OrbPotential(model_name=model_name, device=device, charge=charge, spin=spin, **kwargs)
