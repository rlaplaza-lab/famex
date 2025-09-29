"""
Base class for ML potential calculators in QME.

This module provides a common base class to reduce code duplication across
different ML potential implementations (UMA, SO3LR, AIMNet2).
"""

from abc import ABC, abstractmethod
from typing import Optional

from ase.calculators.calculator import Calculator


class BasePotential(Calculator, ABC):
    """
    Abstract base class for ML potential calculators.

    This class provides common functionality and interface for all ML potential
    calculators, reducing code duplication and ensuring consistency.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self, model_name: Optional[str] = None, device: Optional[str] = None, **kwargs
    ):
        """
        Initialize base potential calculator.

        Parameters:
        -----------
        model_name : str, optional
            Name of the model to load
        device : str, optional
            Device to run computations on ('cpu', 'cuda')
        **kwargs : dict
            Additional arguments passed to Calculator
        """
        Calculator.__init__(self, **kwargs)

        self.model_name = model_name
        self.device = device

        # Initialize the underlying calculator
        self._load_calculator()

    @abstractmethod
    def _load_calculator(self):
        """Load the specific ML calculator implementation.

        This method must be implemented by each subclass to handle
        the specific loading logic for that ML potential.
        """
        pass

    @abstractmethod
    def _get_backend_name(self) -> str:
        """Get the backend name for this calculator.

        Returns:
        --------
        str
            Backend name (e.g., 'uma', 'so3lr', 'aimnet2')
        """
        pass

    def get_calculator(self):
        """Get the calculator instance.

        For most implementations, this returns self since the potential
        class is itself the calculator.

        Returns:
        --------
        Calculator
            The calculator instance that can be used with ASE
        """
        return self

    def __str__(self) -> str:
        """String representation."""
        backend = self._get_backend_name()
        return f"{backend.upper()}Potential(model={self.model_name})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()
