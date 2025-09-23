"""Calculator interfaces for MLP/NNP energy and force evaluations."""

from abc import ABC, abstractmethod

import numpy as np

from .geometry import Geometry


class Calculator(ABC):
    """Abstract base class for energy/force calculators."""

    @abstractmethod
    def calculate(self, geometry: Geometry) -> None:
        """Calculate energy and forces for a geometry (in-place update)."""
        pass

    @abstractmethod
    def get_energy(self, geometry: Geometry) -> float:
        """Get energy for a geometry."""
        pass

    @abstractmethod
    def get_forces(self, geometry: Geometry) -> np.ndarray:
        """Get forces for a geometry."""
        pass


class MLPCalculator(Calculator):
    """Mock MLP/NNP Calculator for demonstration purposes.

    In a real implementation, this would interface with actual ML models
    like ANI, SchNet, or other neural network potentials.
    """

    def __init__(self, model_type: str = "mock", **kwargs):
        """Initialize MLP calculator.

        Args:
            model_type: Type of ML model to use
            **kwargs: Additional model-specific parameters
        """
        self.model_type = model_type
        self.kwargs = kwargs
        self.call_count = 0

    def calculate(self, geometry: Geometry) -> None:
        """Calculate energy and forces for a geometry."""
        self.call_count += 1

        # Mock calculation - in reality this would call the ML model
        energy = self._mock_energy_calculation(geometry)
        forces = self._mock_forces_calculation(geometry)

        geometry.energy = energy
        geometry.forces = forces

    def get_energy(self, geometry: Geometry) -> float:
        """Get energy for a geometry."""
        if geometry.energy is None:
            self.calculate(geometry)
        return geometry.energy

    def get_forces(self, geometry: Geometry) -> np.ndarray:
        """Get forces for a geometry."""
        if geometry.forces is None:
            self.calculate(geometry)
        return geometry.forces

    def _mock_energy_calculation(self, geometry: Geometry) -> float:
        """Mock energy calculation using simple harmonic potential.

        In reality, this would call a trained ML model.
        """
        coords = geometry.coords3d

        # Simple harmonic potential around equilibrium positions
        # This gives reasonable-looking energy surfaces for testing
        bond_energy = 0.0

        # Mock bonding interactions
        for i in range(len(geometry.atoms)):
            for j in range(i + 1, len(geometry.atoms)):
                distance = np.linalg.norm(coords[i] - coords[j])

                # Different equilibrium distances for different atom pairs
                if geometry.atoms[i] == "H" or geometry.atoms[j] == "H":
                    r_eq = 1.0  # H-X bonds
                    k = 100.0  # Force constant
                elif geometry.atoms[i] == "C" and geometry.atoms[j] == "C":
                    r_eq = 1.54  # C-C bond
                    k = 150.0
                else:
                    r_eq = 1.4  # Other bonds
                    k = 120.0

                # Add repulsion at short distances and attraction at medium distances
                if distance < 3.0:  # Only consider nearby atoms
                    bond_energy += 0.5 * k * (distance - r_eq) ** 2

                # Add long-range repulsion to prevent overlap
                if distance < 0.8:
                    bond_energy += 1000.0 / distance**6

        return bond_energy * 0.001  # Convert to reasonable energy scale

    def _mock_forces_calculation(self, geometry: Geometry) -> np.ndarray:
        """Mock forces calculation using numerical differentiation.

        In reality, this would be computed by the ML model directly.
        """
        forces = np.zeros_like(geometry.coords)
        delta = 0.001  # Small displacement for numerical differentiation

        for i in range(len(geometry.coords)):
            # Forward difference
            coords_plus = geometry.coords.copy()
            coords_plus[i] += delta

            geom_plus = Geometry(
                geometry.atoms, coords_plus, charge=geometry.charge, mult=geometry.mult
            )
            energy_plus = self._mock_energy_calculation(geom_plus)

            # Backward difference
            coords_minus = geometry.coords.copy()
            coords_minus[i] -= delta

            geom_minus = Geometry(
                geometry.atoms, coords_minus, charge=geometry.charge, mult=geometry.mult
            )
            energy_minus = self._mock_energy_calculation(geom_minus)

            # Central difference approximation for force
            forces[i] = -(energy_plus - energy_minus) / (2.0 * delta)

        return forces

    def __repr__(self) -> str:
        """String representation."""
        return f"MLPCalculator(type='{self.model_type}', calls={self.call_count})"


class HarmonicCalculator(Calculator):
    """Simple harmonic calculator for testing purposes."""

    def __init__(self, equilibrium_geometry: Geometry, force_constant: float = 100.0):
        """Initialize harmonic calculator.

        Args:
            equilibrium_geometry: Reference geometry for harmonic expansion
            force_constant: Harmonic force constant
        """
        self.eq_geom = equilibrium_geometry
        self.k = force_constant
        self.call_count = 0

    def calculate(self, geometry: Geometry) -> None:
        """Calculate harmonic energy and forces."""
        self.call_count += 1

        displacement = geometry.coords - self.eq_geom.coords
        energy = 0.5 * self.k * np.sum(displacement**2)
        forces = -self.k * displacement

        geometry.energy = energy
        geometry.forces = forces

    def get_energy(self, geometry: Geometry) -> float:
        """Get harmonic energy."""
        if geometry.energy is None:
            self.calculate(geometry)
        return geometry.energy

    def get_forces(self, geometry: Geometry) -> np.ndarray:
        """Get harmonic forces."""
        if geometry.forces is None:
            self.calculate(geometry)
        return geometry.forces
