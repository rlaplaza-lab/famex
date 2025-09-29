"""
Base class for ML potential calculators in QME (moved into qme.potentials).

This is a non-destructive copy of the original `qme/base_potential.py` for
incremental refactoring. The original file remains at root until we finish
updating imports across the codebase.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ase.calculators.calculator import Calculator


class BasePotential(Calculator, ABC):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self, model_name: Optional[str] = None, device: Optional[str] = None, **kwargs
    ):
        Calculator.__init__(self, **kwargs)

        self.model_name = model_name
        self.device = device

        # Initialize the underlying calculator
        self._load_calculator()

    @abstractmethod
    def _load_calculator(self):
        pass

    @abstractmethod
    def _get_backend_name(self) -> str:
        pass

    def get_calculator(self):
        return self

    def __str__(self) -> str:
        backend = self._get_backend_name()
        return f"{backend.upper()}Potential(model={self.model_name})"

    def __repr__(self) -> str:
        return self.__str__()
