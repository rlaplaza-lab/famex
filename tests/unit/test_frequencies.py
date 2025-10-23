"""Test frequency analysis functionality.

This module tests the frequency analysis capabilities of QME,
including Hessian calculation and vibrational mode analysis.
"""

import numpy as np
import pytest
from ase.build import molecule

import qme
from qme.analysis.frequency import FrequencyAnalysis, HessianCalculator


class TestFrequencyBasics:
    """Test basic frequency analysis functionality."""

    def setup_method(self) -> None:
        """Set up test molecules with mock calculator."""
        self.h2 = molecule("H2")
        self.h2.calc = qme.MockCalculator(backend="mock")

        self.h2o = molecule("H2O")
        self.h2o.calc = qme.MockCalculator(backend="mock")

    @pytest.mark.parametrize(
        ("molecule_name", "expected_dof"),
        [
            ("H2", 5),  # Linear molecule: 3N-5
            ("H2O", 6),  # Non-linear molecule: 3N-6
        ],
    )
    def test_molecule_degrees_of_freedom(self, molecule_name, expected_dof) -> None:
        """Test degrees of freedom calculation for different molecules."""
        atoms = self.h2 if molecule_name == "H2" else self.h2o

        fa = FrequencyAnalysis(atoms, atoms.calc, delta=0.01)
        assert fa.nfree == expected_dof

    @pytest.mark.parametrize("molecule_name", ["H2O"])
    def test_hessian_and_modes(self, molecule_name) -> None:
        """Test Hessian calculation and mode diagonalization."""
        atoms = self.h2o
        fa = FrequencyAnalysis(atoms, atoms.calc, delta=0.01)
        hessian = fa.calculate_hessian(method="finite_differences")
        freqs, modes = fa.diagonalize_hessian()

        # Check dimensions
        expected_size = len(atoms) * 3
        assert hessian.shape == (expected_size, expected_size)
        assert len(freqs) == expected_size
        assert modes.shape == (expected_size, expected_size)

        # Check that frequencies are reasonable
        assert not np.any(np.isnan(freqs)), "Frequencies should not contain NaN"
        assert not np.any(np.isinf(freqs)), "Frequencies should not contain infinity"

    @pytest.mark.parametrize("unit", ["cm-1", "meV", "THz"])
    def test_frequency_units(self, unit) -> None:
        """Test frequency unit conversion."""
        fa = FrequencyAnalysis(self.h2o, self.h2o.calc)
        fa.calculate_hessian()
        fa.diagonalize_hessian()

        frequencies = fa.get_frequencies(unit)
        assert isinstance(frequencies, (list, np.ndarray))
        assert len(frequencies) > 0
        assert not np.any(np.isnan(frequencies)), f"Frequencies in {unit} should not contain NaN"

    def test_zero_point_energy(self) -> None:
        """Test zero-point energy calculation."""
        fa = FrequencyAnalysis(self.h2o, self.h2o.calc)
        fa.calculate_hessian()
        fa.diagonalize_hessian()

        zpe = fa.get_zero_point_energy()
        assert isinstance(zpe, float)
        assert zpe >= 0, "Zero-point energy should be non-negative"


class TestHessianCalculator:
    """Test Hessian calculator functionality."""

    def setup_method(self) -> None:
        """Set up test molecule with mock calculator."""
        self.h2o = molecule("H2O")
        self.h2o.calc = qme.MockCalculator(backend="mock")

    @pytest.mark.parametrize(
        ("indices", "expected_shape"),
        [
            (None, (9, 9)),  # All atoms
            ([0], (3, 3)),  # Single atom
            ([0, 1], (6, 6)),  # Two atoms
        ],
    )
    def test_hessian_dimensions(self, indices, expected_shape) -> None:
        """Test Hessian matrix dimensions for different atom subsets."""
        hc = HessianCalculator(self.h2o, self.h2o.calc, indices=indices)
        h = hc.calculate_numerical_hessian()
        assert h.shape == expected_shape

        # Check that Hessian is symmetric
        assert np.allclose(h, h.T), "Hessian should be symmetric"
