"""Test calculator interfaces and implementations."""

import numpy as np
import pytest

from qme import Geometry, HarmonicCalculator, MLPCalculator


class TestCalculators:
    """Test suite for calculator implementations."""

    @pytest.fixture
    def h2_molecule(self):
        """Hydrogen molecule geometry."""
        atoms = ["H", "H"]
        coords = np.array([0.0, 0.0, 0.0, 0.74, 0.0, 0.0])
        return Geometry(atoms=atoms, coords=coords, charge=0, mult=1)

    @pytest.fixture
    def water_molecule(self):
        """Water molecule geometry."""
        atoms = ["O", "H", "H"]
        coords = np.array(
            [
                0.0,
                0.0,
                0.0,  # O
                0.96,
                0.0,
                0.0,  # H
                -0.24,
                0.93,
                0.0,  # H
            ]
        )
        return Geometry(atoms=atoms, coords=coords, charge=0, mult=1)

    def test_mlp_calculator_creation(self):
        """Test MLP calculator creation and basic properties."""
        calc = MLPCalculator(model_type="mock")

        assert calc.model_type == "mock"
        assert calc.call_count == 0

        # Test with additional kwargs
        calc_with_args = MLPCalculator(model_type="ani", cutoff=5.0, device="cpu")
        assert calc_with_args.model_type == "ani"
        assert calc_with_args.kwargs["cutoff"] == 5.0
        assert calc_with_args.kwargs["device"] == "cpu"

    def test_mlp_calculator_energy(self, h2_molecule):
        """Test energy calculation with MLP calculator."""
        calc = MLPCalculator(model_type="mock_h2")

        # Initially no energy
        assert h2_molecule.energy is None

        # Calculate energy
        energy = calc.get_energy(h2_molecule)

        assert energy is not None
        assert h2_molecule.energy is not None
        assert h2_molecule.energy == energy
        assert calc.call_count == 1

    def test_mlp_calculator_forces(self, h2_molecule):
        """Test force calculation with MLP calculator."""
        calc = MLPCalculator(model_type="mock_forces")

        # Initially no forces
        assert h2_molecule.forces is None

        # Calculate forces
        forces = calc.get_forces(h2_molecule)

        assert forces is not None
        assert h2_molecule.forces is not None
        assert np.allclose(forces, h2_molecule.forces)
        assert len(forces) == 6  # 3 coordinates * 2 atoms
        assert calc.call_count == 1

    def test_mlp_calculator_full_calculation(self, water_molecule):
        """Test full calculation (energy and forces together)."""
        calc = MLPCalculator(model_type="mock_water")

        # Calculate both energy and forces
        calc.calculate(water_molecule)

        assert water_molecule.energy is not None
        assert water_molecule.forces is not None
        assert len(water_molecule.forces) == 9  # 3 coordinates * 3 atoms
        assert calc.call_count == 1

        # Calling get_energy again shouldn't trigger new calculation
        energy = calc.get_energy(water_molecule)
        assert calc.call_count == 1  # Still 1

    def test_mlp_calculator_mock_energy_scaling(self, h2_molecule):
        """Test that mock energy calculation gives reasonable values."""
        calc = MLPCalculator(model_type="mock_scaling_test")

        # Calculate for equilibrium geometry
        eq_energy = calc.get_energy(h2_molecule)

        # Create stretched geometry
        stretched = h2_molecule.copy()
        stretched.coords[3] = 1.5  # Stretch H-H bond from 0.74 to 1.5 Å

        stretched_energy = calc.get_energy(stretched)

        # Stretched geometry should have higher energy
        assert stretched_energy > eq_energy

        # Create compressed geometry
        compressed = h2_molecule.copy()
        compressed.coords[3] = 0.5  # Compress H-H bond to 0.5 Å

        compressed_energy = calc.get_energy(compressed)

        # Compressed geometry should also have higher energy
        assert compressed_energy > eq_energy

    def test_harmonic_calculator_creation(self, water_molecule):
        """Test harmonic calculator creation."""
        calc = HarmonicCalculator(
            equilibrium_geometry=water_molecule, force_constant=100.0
        )

        assert calc.eq_geom == water_molecule
        assert calc.k == 100.0
        assert calc.call_count == 0

    def test_harmonic_calculator_at_equilibrium(self, water_molecule):
        """Test harmonic calculator at equilibrium geometry."""
        calc = HarmonicCalculator(
            equilibrium_geometry=water_molecule, force_constant=50.0
        )

        # Calculate at equilibrium
        calc.calculate(water_molecule)

        assert water_molecule.energy == 0.0  # Should be zero at equilibrium
        assert np.allclose(water_molecule.forces, 0.0)  # Forces should be zero
        assert calc.call_count == 1

    def test_harmonic_calculator_displaced(self, water_molecule):
        """Test harmonic calculator with displaced geometry."""
        calc = HarmonicCalculator(
            equilibrium_geometry=water_molecule, force_constant=100.0
        )

        # Create displaced geometry
        displaced = water_molecule.copy()
        displacement = np.array([0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1])
        displaced.coords += displacement

        # Calculate energy and forces
        calc.calculate(displaced)

        # Energy should be positive (displaced from equilibrium)
        assert displaced.energy > 0.0

        # Forces should oppose displacement
        assert np.dot(displacement, displaced.forces) < 0

        # Check harmonic energy formula: E = 0.5 * k * displacement^2
        expected_energy = 0.5 * calc.k * np.sum(displacement**2)
        assert np.isclose(displaced.energy, expected_energy)

        # Check force formula: F = -k * displacement
        expected_forces = -calc.k * displacement
        assert np.allclose(displaced.forces, expected_forces)

    def test_harmonic_calculator_different_force_constants(self, h2_molecule):
        """Test harmonic calculator with different force constants."""
        # Weak harmonic potential
        calc_weak = HarmonicCalculator(h2_molecule, force_constant=10.0)

        # Strong harmonic potential
        calc_strong = HarmonicCalculator(h2_molecule, force_constant=1000.0)

        # Same displacement
        displaced = h2_molecule.copy()
        displaced.coords += 0.1

        # Calculate with both
        displaced_weak = displaced.copy()
        displaced_strong = displaced.copy()

        calc_weak.calculate(displaced_weak)
        calc_strong.calculate(displaced_strong)

        # Strong potential should give higher energy and forces
        assert displaced_strong.energy > displaced_weak.energy
        assert np.linalg.norm(displaced_strong.forces) > np.linalg.norm(
            displaced_weak.forces
        )

        # Energy ratio should match force constant ratio
        energy_ratio = displaced_strong.energy / displaced_weak.energy
        force_ratio = calc_strong.k / calc_weak.k
        assert np.isclose(energy_ratio, force_ratio)

    def test_calculator_multiple_calls(self, water_molecule):
        """Test multiple calculator calls update call count."""
        calc = MLPCalculator(model_type="mock_multiple")

        # Multiple calculations on different geometries
        geoms = [water_molecule.copy() for _ in range(3)]

        for i, geom in enumerate(geoms):
            # Slightly displace each geometry
            geom.coords += i * 0.1
            calc.calculate(geom)
            assert calc.call_count == i + 1

    def test_calculator_repr(self, water_molecule):
        """Test calculator string representations."""
        mlp_calc = MLPCalculator(model_type="test")
        harmonic_calc = HarmonicCalculator(water_molecule, force_constant=50.0)

        mlp_repr = repr(mlp_calc)
        harmonic_repr = repr(harmonic_calc)

        assert "MLPCalculator" in mlp_repr
        assert "type='test'" in mlp_repr
        assert "calls=0" in mlp_repr

        # After calculation, call count should update in repr
        mlp_calc.calculate(water_molecule)
        mlp_repr_after = repr(mlp_calc)
        assert "calls=1" in mlp_repr_after

    def test_force_numerical_stability(self, h2_molecule):
        """Test numerical stability of force calculations."""
        calc = MLPCalculator(model_type="mock_stability")

        # Calculate forces for the same geometry multiple times
        forces_list = []
        for _ in range(5):
            geom_copy = h2_molecule.copy()
            calc.calculate(geom_copy)
            forces_list.append(geom_copy.forces.copy())

        # All force calculations should be identical (deterministic)
        for forces in forces_list[1:]:
            assert np.allclose(forces, forces_list[0], rtol=1e-10)

    def test_energy_conservation_harmonic(self, water_molecule):
        """Test energy conservation in harmonic potential."""
        calc = HarmonicCalculator(water_molecule, force_constant=100.0)

        # Create several geometries at different distances from equilibrium
        displacements = [0.1, 0.2, 0.3, 0.5, 1.0]
        energies = []

        for disp in displacements:
            displaced = water_molecule.copy()
            displaced.coords += disp

            calc.calculate(displaced)
            energies.append(displaced.energy)

        # Energy should increase quadratically with displacement
        for i in range(1, len(displacements)):
            ratio_disp = (displacements[i] / displacements[0]) ** 2
            ratio_energy = energies[i] / energies[0]
            assert np.isclose(ratio_energy, ratio_disp, rtol=0.01)
