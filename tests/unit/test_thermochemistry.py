"""Unit tests for thermochemistry modules."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from qme.analysis.quasiharmonic import (
    QuasiHarmonicHandler,
    calculate_damping_function,
    calculate_rrho_entropy,
)
from qme.analysis.solvation import SolvationHandler, get_free_space
from qme.analysis.statistical_thermo import (
    StatisticalThermodynamics,
    calculate_electronic_entropy,
    calculate_rotational_energy,
    calculate_rotational_entropy,
    calculate_translational_energy,
    calculate_translational_entropy,
)
from qme.analysis.symmetry import SymmetryHandler, get_point_group_symmetry_number


class TestQuasiHarmonic:
    """Test quasi-harmonic corrections."""

    def test_rrho_entropy(self):
        """Test RRHO entropy calculation."""
        frequencies = np.array([100, 500, 1000])  # cm^-1
        temperature = 298.15  # K
        entropy = calculate_rrho_entropy(frequencies, temperature)

        # Should return per-mode entropies
        assert len(entropy) == 3
        assert np.all(entropy > 0)  # All entropies should be positive
        # Higher frequencies should have lower entropy
        assert entropy[2] < entropy[1] < entropy[0]

    def test_damping_function(self):
        """Test damping function for Grimme method."""
        frequencies = np.array([50, 100, 500, 1000])  # cm^-1
        freq_cutoff = 100.0  # cm^-1
        damp = calculate_damping_function(frequencies, freq_cutoff)

        # Damping should decrease with decreasing frequency
        assert np.all(damp >= 0) and np.all(damp <= 1)
        assert damp[3] > damp[2] > damp[1] > damp[0]  # Higher freq -> more damping (closer to RRHO)

    def test_quasi_harmonic_handler(self):
        """Test QuasiHarmonicHandler."""
        handler = QuasiHarmonicHandler(method="grimme", freq_cutoff=100.0)

        frequencies = np.array([50, 100, 500, 1000])  # cm^-1
        temperature = 298.15

        entropy, per_mode = handler.vibrational_entropy(frequencies, temperature)
        assert entropy > 0
        assert len(per_mode) == 4

        energy, per_mode_energy = handler.vibrational_energy(frequencies, temperature)
        assert energy > 0
        assert len(per_mode_energy) == 4


class TestStatisticalThermodynamics:
    """Test statistical thermodynamics contributions."""

    def test_translational_energy(self):
        """Test translational energy calculation."""
        temperature = 298.15  # K
        energy = calculate_translational_energy(temperature)

        # Translational energy should be 3/2 RT
        expected = 1.5 * 8.3144621 * temperature
        assert abs(energy - expected) < 0.002  # Slight tolerance for constant precision differences

    def test_translational_entropy(self):
        """Test translational entropy calculation."""
        molecular_mass = 18.0  # H2O
        temperature = 298.15
        entropy = calculate_translational_entropy(molecular_mass, temperature)

        assert entropy > 0

    def test_rotational_energy(self):
        """Test rotational energy calculation."""
        temperature = 298.15

        # Linear molecule
        energy_linear = calculate_rotational_energy(temperature, linear=True)
        assert (
            abs(energy_linear - 8.3144621 * temperature) < 0.002
        )  # Tolerance for constant precision

        # Non-linear molecule
        energy_nonlinear = calculate_rotational_energy(temperature, linear=False)
        assert (
            abs(energy_nonlinear - 1.5 * 8.3144621 * temperature) < 0.002
        )  # Tolerance for constant precision

        # Atom
        energy_atom = calculate_rotational_energy(temperature, is_atom=True)
        assert abs(energy_atom) < 1e-10

    def test_rotational_entropy(self):
        """Test rotational entropy calculation."""
        temperature = 298.15
        symmetry_number = 1

        # Non-linear molecule
        rot_temps = np.array([10.0, 10.0, 10.0])
        entropy = calculate_rotational_entropy(
            rot_temps, symmetry_number, temperature, linear=False
        )
        assert entropy > 0

        # Linear molecule
        rot_temps_linear = np.array([10.0])
        entropy_linear = calculate_rotational_entropy(
            rot_temps_linear, symmetry_number, temperature, linear=True
        )
        assert entropy_linear > 0

        # Atom
        entropy_atom = calculate_rotational_entropy(
            np.array([0.0, 0.0, 0.0]), symmetry_number, temperature, is_atom=True
        )
        assert abs(entropy_atom) < 1e-10

    def test_electronic_entropy(self):
        """Test electronic entropy calculation."""
        # Singlet (multiplicity = 1)
        entropy = calculate_electronic_entropy(1)
        assert abs(entropy) < 1e-10

        # Triplet (multiplicity = 3)
        entropy_triplet = calculate_electronic_entropy(3)
        expected = 8.3144621 * np.log(3)
        assert abs(entropy_triplet - expected) < 1e-5  # Tolerance for constant precision

    def test_statistical_thermodynamics(self):
        """Test StatisticalThermodynamics class."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0.82, 0, -0.26]])

        stat_thermo = StatisticalThermodynamics(atoms)

        energy_trans = stat_thermo.translational_energy(298.15)
        assert energy_trans > 0

        entropy_trans = stat_thermo.translational_entropy(298.15)
        assert entropy_trans > 0


class TestSolvation:
    """Test solvation corrections."""

    def test_get_free_space(self):
        """Test free space calculation for different solvents."""
        # Gas phase
        free_space_gas = get_free_space("none")
        assert abs(free_space_gas - 1000.0) < 0.1

        # Water
        free_space_water = get_free_space("H2O")
        assert free_space_water < 1000.0  # Less accessible space than gas phase
        assert free_space_water > 0

        # Unknown solvent should raise error
        with pytest.raises(ValueError):
            get_free_space("unknown_solvent")

    def test_solvation_handler(self):
        """Test SolvationHandler."""
        handler = SolvationHandler(solvent="H2O", concentration=1.0)
        assert not handler.is_gas_phase()
        assert handler.concentration == 1.0

        handler_gas = SolvationHandler(solvent="none")
        assert handler_gas.is_gas_phase()


class TestSymmetry:
    """Test symmetry handling."""

    def test_get_point_group_symmetry_number(self):
        """Test symmetry number lookup."""
        assert get_point_group_symmetry_number("C1") == 1
        assert get_point_group_symmetry_number("C2v") == 2
        assert get_point_group_symmetry_number("Td") == 12
        assert get_point_group_symmetry_number("Oh") == 24

        # Unknown point group should raise error
        with pytest.raises(ValueError):
            get_point_group_symmetry_number("unknown")

    def test_symmetry_handler_default(self):
        """Test SymmetryHandler with default C1 assumption."""
        with pytest.warns(UserWarning, match="C1 symmetry"):
            handler = SymmetryHandler()
        assert handler.symmetry_number == 1
        assert handler.point_group == "C1"

    def test_symmetry_handler_explicit(self):
        """Test SymmetryHandler with explicit symmetry number."""
        handler = SymmetryHandler(symmetry_number=12)
        assert handler.symmetry_number == 12

    def test_symmetry_handler_point_group(self):
        """Test SymmetryHandler with point group."""
        handler = SymmetryHandler(point_group="C2v")
        assert handler.symmetry_number == 2
        assert handler.point_group == "C2v"

    def test_rotational_symmetry_number(self):
        """Test rotational symmetry number for linear vs non-linear."""
        handler_linear = SymmetryHandler(symmetry_number=2)
        assert handler_linear.get_rotational_symmetry_number(linear=True) == 1

        handler_nonlinear = SymmetryHandler(symmetry_number=2)
        assert handler_nonlinear.get_rotational_symmetry_number(linear=False) == 2


@pytest.fixture
def simple_molecule():
    """Create a simple water molecule for testing."""
    return Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0.82, 0, -0.26]])


class TestCompleteThermodynamics:
    """Test complete thermodynamics calculations."""

    def test_basic_thermodynamics(self, simple_molecule):
        """Test basic vibrational thermodynamics."""
        from qme.analysis.thermodynamics import ThermodynamicProperties

        # Simple frequencies for water (approximate)
        frequencies = np.array([1000, 1650, 3650])

        thermo = ThermodynamicProperties(frequencies, simple_molecule, temperature=298.15)

        zpe = thermo.calculate_zero_point_energy()
        assert zpe > 0

        entropy = thermo.entropy_vibrational()
        assert entropy > 0

    def test_complete_thermodynamics_gas_phase(self, simple_molecule):
        """Test complete thermodynamics in gas phase."""
        from qme.analysis.thermodynamics import ThermodynamicProperties

        frequencies = np.array([1000, 1650, 3650])

        thermo = ThermodynamicProperties(
            frequencies,
            simple_molecule,
            temperature=298.15,
            method="rrho",
            multiplicity=1,
            solvent="none",
        )

        results = thermo.calculate_complete_thermodynamics(energy=0.0)

        # Check all keys present
        expected_keys = [
            "energy",
            "zpe",
            "enthalpy_trans",
            "enthalpy_rot",
            "enthalpy_vib",
            "enthalpy_total",
            "entropy_trans",
            "entropy_rot",
            "entropy_vib",
            "entropy_elec",
            "entropy_total",
            "gibbs_free_energy",
            "temperature",
            "method",
            "contributions",
        ]
        for key in expected_keys:
            assert key in results

        # Check some reasonable values
        assert results["zpe"] > 0
        assert results["entropy_total"] > 0
        assert results["temperature"] == 298.15

    def test_quasi_harmonic_methods(self, simple_molecule):
        """Test different quasi-harmonic methods."""
        from qme.analysis.thermodynamics import ThermodynamicProperties

        frequencies = np.array([50, 100, 500, 1000, 1650, 3650])

        # RRHO
        thermo_rrho = ThermodynamicProperties(
            frequencies, simple_molecule, temperature=298.15, method="rrho"
        )
        results_rrho = thermo_rrho.calculate_complete_thermodynamics(energy=0.0)

        # Grimme
        thermo_grimme = ThermodynamicProperties(
            frequencies, simple_molecule, temperature=298.15, method="grimme"
        )
        results_grimme = thermo_grimme.calculate_complete_thermodynamics(energy=0.0)

        # Truhlar
        thermo_truhlar = ThermodynamicProperties(
            frequencies, simple_molecule, temperature=298.15, method="truhlar"
        )
        results_truhlar = thermo_truhlar.calculate_complete_thermodynamics(energy=0.0)

        # All should give positive entropies
        assert results_rrho["entropy_vib"] > 0
        assert results_grimme["entropy_vib"] > 0
        assert results_truhlar["entropy_vib"] > 0

        # Methods should give similar but not identical results
        # Grimme and Truhlar should generally give higher entropies than RRHO for low frequencies
        assert results_grimme["entropy_vib"] >= results_rrho["entropy_vib"] - 0.01
        assert results_truhlar["entropy_vib"] >= results_rrho["entropy_vib"] - 0.01
