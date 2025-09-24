"""
MLPCalculator convenience class for machine learning potentials in QME.
"""

from typing import Optional, Union

from .core import QMEOptimizer


class MLPCalculator:
    """
    Convenience class for creating machine learning potential calculators.

    This class provides a simple interface to create calculators for different
    ML backends (UMA, SO3LR, AIMNET2) used in the examples.
    """

    def __init__(
        self,
        model_type: str = "so3lr",
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_mock: bool = False,
    ):
        """
        Initialize MLP calculator.

        Parameters
        ----------
        model_type : str, default "so3lr"
            Type of ML model ("so3lr", "uma", "aimnet2", "mock")
        model_name : str, optional
            Specific model name to use
        model_path : str, optional
            Path to model file (SO3LR only)
        device : str, optional
            Device for computation ("cpu", "cuda")
        use_mock : bool, default False
            Use mock calculator for testing
        """

        # Handle special case for mock model type
        if model_type == "mock":
            use_mock = True
            model_type = "so3lr"  # Default backend for mock

        # Create QME optimizer to get the calculator
        self._qme = QMEOptimizer(
            backend=model_type,
            model_name=model_name,
            model_path=model_path,
            device=device,
            use_mock=use_mock,
        )

        # Extract the calculator
        self.calculator = self._qme.calculator
        self.backend = model_type
        self.model_name = model_name

    def __call__(self, atoms):
        """Make the MLPCalculator callable to set calculator on atoms."""
        atoms.calc = self.calculator
        return atoms

    def calculate(self, geometry):
        """
        Calculate energy and forces for a geometry.

        Parameters
        ----------
        geometry : Geometry
            Geometry object to calculate for
        """

        # Set calculator on the geometry's atoms
        geometry.atoms.calc = self.calculator

        # Force calculation
        try:
            energy = geometry.atoms.get_potential_energy()
            forces = geometry.atoms.get_forces()

            # Store in geometry
            geometry.energy = energy
            geometry.forces = forces

        except Exception as e:
            print(f"Warning: Energy calculation failed: {e}")

    @property
    def name(self) -> str:
        """Get calculator name."""
        return f"{self.backend.upper()}Calculator"

    def __str__(self) -> str:
        """String representation."""
        return f"MLPCalculator({self.backend}, {self.model_name})"

    def __repr__(self) -> str:
        return self.__str__()
