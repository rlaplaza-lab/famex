"""Test hessian consistency between analytical and finite differences.

This module tests that analytical hessian calculations match finite difference
approximations for backends that support analytical hessians (MACE and UMA).
The tests are optional and will be skipped if the required backends are not available.
"""

import numpy as np
import pytest
from ase import Atoms

import qme
from qme.analysis.frequency import FrequencyAnalysis, HessianCalculator
from qme.backends.availability import is_backend_available
from tests.test_utils import TestMoleculeFactory


class TestHessianConsistency:
    """Test hessian consistency between analytical and finite difference methods."""

    @pytest.fixture
    def water_molecule(self) -> Atoms:
        """Water molecule for hessian testing."""
        return TestMoleculeFactory.get_water_distorted()

    @pytest.fixture
    def methane_molecule(self) -> Atoms:
        """Methane molecule for testing."""
        return Atoms(
            symbols="CHHHH",
            positions=[
                [0.0, 0.0, 0.0],
                [1.09, 0.0, 0.0],
                [-0.36, 1.03, 0.0],
                [-0.36, -0.51, 0.89],
                [-0.36, -0.51, -0.89],
            ],
        )

    @pytest.mark.skipif(not is_backend_available("mace"), reason="MACE backend not available")
    def test_mace_hessian_consistency(self, water_molecule: Atoms) -> None:
        """Test MACE analytical hessian matches finite differences."""
        atoms = water_molecule.copy()
        atoms.calc = qme.get_mace_calculator(model_name="mace-omol-0")
        atoms.calc.ensure_loaded()

        assert "hessian" in atoms.calc.implemented_properties

        # Analytical hessian
        hessian_analytical = atoms.calc.get_hessian(atoms)

        # Finite difference hessian
        hc = HessianCalculator(atoms, atoms.calc, delta=0.01, verbose=0)
        hessian_fd = hc.calculate_numerical_hessian()

        # Compare
        assert hessian_analytical.shape == hessian_fd.shape
        np.testing.assert_allclose(hessian_analytical, hessian_fd, rtol=0.1, atol=3.0)

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_uma_hessian_consistency(self, water_molecule: Atoms) -> None:
        """Test UMA analytical hessian matches finite differences."""
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        assert "hessian" in atoms.calc.implemented_properties

        # Analytical hessian
        hessian_analytical = atoms.calc.get_hessian(atoms)

        # Finite difference hessian
        hc = HessianCalculator(atoms, atoms.calc, delta=0.01, verbose=0)
        hessian_fd = hc.calculate_numerical_hessian()

        # Compare
        assert hessian_analytical.shape == hessian_fd.shape
        np.testing.assert_allclose(hessian_analytical, hessian_fd, rtol=0.1, atol=3.0)

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_uma_hessian_methane(self, methane_molecule: Atoms) -> None:
        """Test UMA hessian on methane molecule."""
        atoms = methane_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        # Analytical hessian
        hessian_analytical = atoms.calc.get_hessian(atoms)

        # Finite difference hessian
        hc = HessianCalculator(atoms, atoms.calc, delta=0.01, verbose=0)
        hessian_fd = hc.calculate_numerical_hessian()

        # Compare
        assert hessian_analytical.shape == hessian_fd.shape
        np.testing.assert_allclose(hessian_analytical, hessian_fd, rtol=0.1, atol=3.0)

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_uma_frequency_comparison(self, water_molecule: Atoms) -> None:
        """Test that analytical and finite difference Hessians yield similar frequencies."""
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        # Analytical frequencies
        freq_analytical = FrequencyAnalysis(atoms, atoms.calc, delta=0.01, verbose=0)
        freq_analytical.calculate_hessian(method="direct")
        freq_analytical.diagonalize_hessian()
        freqs_analytical = freq_analytical.get_frequencies()

        # Finite difference frequencies
        freq_fd = FrequencyAnalysis(atoms, atoms.calc, delta=0.001, verbose=0)
        freq_fd.calculate_hessian(method="finite_differences")
        freq_fd.diagonalize_hessian()
        freqs_fd = freq_fd.get_frequencies()

        # Compare vibrational frequencies only (skip translational/rotational modes)
        # For water (3 atoms, non-linear), skip first 6 modes
        freqs_analytical_vib = freqs_analytical[6:]
        freqs_fd_vib = freqs_fd[6:]

        # Check that vibrational frequencies match within reasonable tolerance
        np.testing.assert_allclose(
            freqs_analytical_vib,
            freqs_fd_vib,
            rtol=0.05,  # 5% relative tolerance
            atol=50.0,  # 50 cm^-1 absolute tolerance
        )

        # Verify that frequencies are positive (stable molecule)
        assert np.all(freqs_analytical_vib > 0)
        assert np.all(freqs_fd_vib > 0)
