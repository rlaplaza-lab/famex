"""
SO3LR Neural Network Potential integration for ASE.

SO3LR is an open source neural network potential with SO(3) invariant architecture.
This module provides ASE Calculator interface for SO3LR models.
"""

from typing import Optional

import numpy as np
from ase.calculators.calculator import all_changes

from qme.dependencies import deps
from qme.potentials.base_potential import BasePotential


class SO3LRPotential(BasePotential):
    """
    ASE Calculator interface for SO3LR neural network potential.

    This is a wrapper around the native SO3LR ASE calculator to provide
    compatibility with the QME interface.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "so3lr-small",
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize SO3LR potential calculator.

        Parameters:
        -----------
        model_path : str, optional
            Path to trained SO3LR model file (currently not used by SO3LR)
        model_name : str
            Name of pre-trained SO3LR model (default: "so3lr-small")
        device : str, optional
            Device to run computations on ('cpu', 'cuda'). Auto-detected if None.

        """

        if not deps.has("so3lr"):
            raise ImportError(
                "SO3LR is required for SO3LR potentials. "
                "Install SO3LR with: git clone "
                "https://github.com/general-molecular-simulations/so3lr.git && "
                "cd so3lr && pip install ."
            )

        # Store additional SO3LR-specific parameters
        self.model_path = model_path

        # SO3LR-specific attributes
        # Standard backend attribute used by BasePotential helpers
        self._calc = None

        # Initialize base class (this will call _load_calculator)
        super().__init__(model_name=model_name, device=device, **kwargs)

    def _load_calculator(self):
        """Load the SO3LR ASE calculator."""
        # Skip if already loaded
        if hasattr(self, "_calc") and self._calc is not None:
            return

        from .logging_utils import quiet_backend_loading

        with quiet_backend_loading(
            "so3lr", self.model_name, self.model_path, self.device
        ):
            # Get SO3LR module
            so3lr = deps.get("so3lr")
            if so3lr is None:
                raise RuntimeError("SO3LR module not available")

        # Create SO3LR calculator with appropriate parameters
        # Use high cutoff for gas-phase systems as recommended
        lr_cutoff = 1000.0

        self._calc = so3lr.So3lrCalculator(
            calculate_stress=False, lr_cutoff=lr_cutoff, dtype=np.float32
        )

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=all_changes,
    ):
        """Calculate properties using SO3LR potential."""
        if atoms is None:
            return

        super().calculate(atoms, properties, system_changes)

        # Ensure atoms has charge information as required by SO3LR
        if "charge" not in atoms.info:
            atoms.info["charge"] = 0.0

        # Ensure calculator is loaded
        if self._calc is None:
            self._load_calculator()

        # Use the underlying calculator directly
        self._calc.calculate(atoms, properties, system_changes)

        # Extract results from the underlying calculator
        if "energy" in properties:
            try:
                self.results["energy"] = self._calc.results["energy"]
            except Exception:
                self.results["energy"] = self.results.get("energy")

        if "forces" in properties:
            try:
                self.results["forces"] = self._calc.results["forces"]
            except Exception:
                self.results["forces"] = self.results.get("forces")

    def get_potential_energy(self, atoms=None, force_consistent: bool = False):
        """Get potential energy (ASE-compatible)."""
        if atoms is not None:
            self.atoms = atoms

        return super().get_potential_energy(atoms, force_consistent)

    def get_forces(self, atoms=None):
        """Get forces (ASE-compatible)."""
        if atoms is not None:
            self.atoms = atoms

        return super().get_forces(atoms)

    def _get_backend_name(self) -> str:
        """Get the backend name for SO3LR."""
        return "so3lr"


def get_so3lr_calculator(
    model_path: Optional[str] = None,
    model_name: str = "so3lr-small",
    device: Optional[str] = None,
    **kwargs,
) -> SO3LRPotential:
    """
    Convenience function to get SO3LR calculator.

    Parameters:
    -----------
    model_path : str, optional
        Path to trained SO3LR model file (currently not used by SO3LR)
    model_name : str
        Name of pre-trained SO3LR model (currently not used by SO3LR)
    device : str, optional
        Device preference ('cpu', 'cuda')
    **kwargs :
        Additional arguments passed to SO3LRPotential

    Returns:
    --------
    SO3LRPotential
        Configured SO3LR calculator
    """
    return SO3LRPotential(
        model_path=model_path, model_name=model_name, device=device, **kwargs
    )
