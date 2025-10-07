"""
Test frequency analysis functionality.

This module tests the frequency analysis capabilities of QME,
including Hessian calculation and vibrational mode analysis.
"""

import numpy as np
import pytest
from ase.build import molecule

import qme
from qme.analysis.frequency import FrequencyAnalysis, HessianCalculator
from tests.test_utils import StandardTestAssertions


class TestFrequencyBasics:
    """Test basic frequency analysis functionality."""

    def setup_method(self):
        """Set up test molecules with mock calculator."""
        self.h2 = molecule("H2")
        self.h2.calc = qme.MockCalculator(backend="mock")

        self.h2o = molecule("H2O")
        self.h2o.calc = qme.MockCalculator(backend="mock")

    def test_linear_molecule_degrees(self):
        """Test degrees of freedom calculation for linear molecule."""
        fa = FrequencyAnalysis(self.h2, self.h2.calc, delta=0.01)
        assert fa.nfree == 5

    def test_water_hessian_and_modes(self):
        """Test Hessian calculation and mode diagonalization for water."""
        fa = FrequencyAnalysis(self.h2o, self.h2o.calc, delta=0.01)
        hessian = fa.calculate_hessian(method="finite_differences")
        freqs, modes = fa.diagonalize_hessian()

        # Check dimensions
        assert hessian.shape == (9, 9)
        assert len(freqs) == 9
        assert modes.shape == (9, 9)

        # Check that frequencies are reasonable
        assert not np.any(np.isnan(freqs)), "Frequencies should not contain NaN"
        assert not np.any(np.isinf(freqs)), "Frequencies should not contain infinity"

    def test_units_and_zpe(self):
        """Test frequency unit conversion and zero-point energy calculation."""
        fa = FrequencyAnalysis(self.h2o, self.h2o.calc)
        fa.calculate_hessian()
        fa.diagonalize_hessian()

        # Test different units
        cm1 = fa.get_frequencies("cm-1")
        meV = fa.get_frequencies("meV")
        THz = fa.get_frequencies("THz")

        # All should have same length
        assert len(cm1) == len(meV) == len(THz)

        # Test zero-point energy
        zpe = fa.get_zero_point_energy()
        assert isinstance(zpe, float)
        assert zpe >= 0, "Zero-point energy should be non-negative"


class TestHessianCalculator:
    """Test Hessian calculator functionality."""

    def setup_method(self):
        """Set up test molecule with mock calculator."""
        self.h2o = molecule("H2O")
        self.h2o.calc = qme.MockCalculator(backend="mock")

    def test_hessian_dimensions(self):
        """Test Hessian matrix dimensions."""
        hc = HessianCalculator(self.h2o, self.h2o.calc)
        h = hc.calculate_numerical_hessian()
        assert h.shape == (9, 9)

        # Check that Hessian is symmetric
        assert np.allclose(h, h.T), "Hessian should be symmetric"

    def test_subset_indices(self):
        """Test Hessian calculation for subset of atoms."""
        hc = HessianCalculator(self.h2o, self.h2o.calc, indices=[0])
        h = hc.calculate_numerical_hessian()
        assert h.shape == (3, 3)

        # Check that subset Hessian is symmetric
        assert np.allclose(h, h.T), "Subset Hessian should be symmetric"
