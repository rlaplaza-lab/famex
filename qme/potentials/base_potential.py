"""Base potential classes for QME."""

from typing import Any


class BasePotential:
    """Abstract base class for ML potential calculators.

    Concrete backends should implement a compatible interface or provide
    an ASE Calculator wrapper.
    """

    def __init__(self, **kwargs: Any):
        self.backend = kwargs.get("backend", "generic")

    def calculate(self, atoms):
        raise NotImplementedError()


__all__ = ["BasePotential"]
