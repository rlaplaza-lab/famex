"""
SO3LR Neural Network Potential integration for ASE.

SO3LR is an open source neural network potential with SO(3) invariant architecture.
This module provides ASE Calculator interface for SO3LR models.
"""

from typing import Optional

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .dependencies import HAS_SO3LR, deps


class SO3LRPotential(Calculator):
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

        Calculator.__init__(self, **kwargs)

        # Store parameters
        self.model_path = model_path
        self.model_name = model_name
        self.device = device

        if not HAS_SO3LR:
            raise ImportError(
                "SO3LR is required for SO3LR potentials. "
                "Install SO3LR with: git clone "
                "https://github.com/general-molecular-simulations/so3lr.git && "
                "cd so3lr && pip install ."
            )

        # Initialize the actual SO3LR calculator
        self.calculator = None
        self._load_calculator()

    def _load_calculator(self):
        """Load the SO3LR ASE calculator."""
        # Get SO3LR module
        so3lr = deps.get("so3lr")
        if so3lr is None:
            raise RuntimeError("SO3LR module not available")

        # Create SO3LR calculator with appropriate parameters
        # Use high cutoff for gas-phase systems as recommended
        lr_cutoff = 1000.0

        self.calculator = so3lr.So3lrCalculator(
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

        Calculator.calculate(self, atoms, properties, system_changes)

        # Ensure atoms has charge information as required by SO3LR
        if "charge" not in atoms.info:
            atoms.info["charge"] = 0.0

        # Set the calculator on the atoms and get properties
        atoms.calc = self.calculator

        if "energy" in properties:
            energy = atoms.get_potential_energy()
            self.results["energy"] = energy

        if "forces" in properties:
            forces = atoms.get_forces()
            self.results["forces"] = forces

    def get_calculator(self):
        """Get the calculator instance.

        For SO3LR, this returns self since SO3LRPotential is itself the calculator.

        Returns:
        --------
        SO3LRPotential
            The calculator instance that can be used with ASE
        """
        return self


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
