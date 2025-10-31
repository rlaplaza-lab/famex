"""Test hessian consistency between analytical and finite differences.

This module tests that analytical hessian calculations match finite difference
approximations for backends that support analytical hessians (MACE and UMA).
The tests are optional and will be skipped if the required backends are not available.

Also includes tests for different finite difference schemes including 5-point
and Richardson extrapolation.
"""

import numpy as np
import pytest
from ase import Atoms

import qme
from qme.analysis.frequency import FrequencyAnalysis, HessianCalculator
from qme.backends.availability import is_backend_available
from tests.test_utils import TestMoleculeFactory


class HarmonicCalculator:
    """Mock calculator for harmonic potential for testing."""

    def __init__(self, k: float = 1.0) -> None:
        """Initialize with force constant k."""
        self.k = k

    def get_forces(self, atoms: Atoms) -> np.ndarray:
        """Compute harmonic forces: F = -k * r."""
        forces = -self.k * atoms.positions
        return forces

    def get_hessian(self, atoms: Atoms) -> np.ndarray:
        """Compute analytical harmonic Hessian: H = k * I."""
        n_atoms = len(atoms)
        n_coords = 3 * n_atoms
        hessian = self.k * np.eye(n_coords)
        return hessian


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

    @pytest.fixture
    def harmonic_atoms(self) -> Atoms:
        """Simple harmonic system for testing."""
        return Atoms(
            symbols="HH",
            positions=[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]],
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

    def test_5point_harmonic_accuracy(self, harmonic_atoms: Atoms) -> None:
        """Test 5-point scheme against analytical harmonic Hessian."""
        calc = HarmonicCalculator(k=1.0)

        # 5-point finite difference
        hc_5point = HessianCalculator(harmonic_atoms, calc, delta=0.01, method="5point", verbose=0)
        hessian_5point = hc_5point.calculate_numerical_hessian()

        # Analytical
        hessian_analytical = calc.get_hessian(harmonic_atoms)

        # Should be very accurate with small delta
        np.testing.assert_allclose(hessian_5point, hessian_analytical, rtol=1e-6, atol=1e-6)

    def test_5point_with_richardson(self, harmonic_atoms: Atoms) -> None:
        """Test 5-point scheme with Richardson extrapolation."""
        calc = HarmonicCalculator(k=1.0)

        # 5-point + Richardson
        hc_5point_rich = HessianCalculator(
            harmonic_atoms,
            calc,
            delta=0.01,
            method="5point",
            richardson=True,
            delta2=0.005,
            verbose=0,
        )
        hessian_5point_rich = hc_5point_rich.calculate_numerical_hessian()

        # 5-point without Richardson
        hc_5point = HessianCalculator(harmonic_atoms, calc, delta=0.01, method="5point", verbose=0)
        hessian_5point = hc_5point.calculate_numerical_hessian()

        # Analytical
        hessian_analytical = calc.get_hessian(harmonic_atoms)

        # Both should be very accurate (harmonic is exact)
        error_with_rich = np.max(np.abs(hessian_5point_rich - hessian_analytical))
        error_without_rich = np.max(np.abs(hessian_5point - hessian_analytical))

        assert error_with_rich < 1e-10, "5-point with Richardson should be accurate"
        assert error_without_rich < 1e-10, "5-point without Richardson should be accurate"

    def test_richardson_order_detection(self, harmonic_atoms: Atoms) -> None:
        """Test that Richardson order is correctly detected."""
        calc = HarmonicCalculator()

        # 3-point + Richardson should have order 2
        hc_3 = HessianCalculator(harmonic_atoms, calc, method="central", richardson=True, verbose=0)
        assert hc_3._richardson_order == 2

        # 5-point + Richardson should have order 4
        hc_5 = HessianCalculator(harmonic_atoms, calc, method="5point", richardson=True, verbose=0)
        assert hc_5._richardson_order == 4

    def test_backward_compatibility(self, water_molecule: Atoms) -> None:
        """Test that existing 3-point and forward methods still work."""
        calc = HarmonicCalculator(k=1.0)

        # All methods should work without error
        hc_forward = HessianCalculator(water_molecule, calc, method="forward", verbose=0)
        hessian_forward = hc_forward.calculate_numerical_hessian()

        hc_central = HessianCalculator(water_molecule, calc, method="central", verbose=0)
        hessian_central = hc_central.calculate_numerical_hessian()

        hc_5point = HessianCalculator(water_molecule, calc, method="5point", verbose=0)
        hessian_5point = hc_5point.calculate_numerical_hessian()

        # Should all produce same shape
        assert hessian_forward.shape == hessian_central.shape == hessian_5point.shape

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises error."""
        atoms = Atoms(symbols="H", positions=[[0, 0, 0]])
        calc = HarmonicCalculator()

        with pytest.raises(ValueError, match="Unknown finite difference method"):
            HessianCalculator(atoms, calc, method="7point", verbose=0)

    def test_richardson_with_forward_raises_error(self) -> None:
        """Test that Richardson with forward method raises error."""
        atoms = Atoms(symbols="H", positions=[[0, 0, 0]])
        calc = HarmonicCalculator()

        with pytest.raises(ValueError, match="Richardson extrapolation currently supported only"):
            HessianCalculator(atoms, calc, method="forward", richardson=True, verbose=0)

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_5point_uma_consistency(self, water_molecule: Atoms) -> None:
        """Test 5-point scheme against UMA analytical Hessian."""
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        assert "hessian" in atoms.calc.implemented_properties

        # Analytical Hessian
        hessian_analytical = atoms.calc.get_hessian(atoms)

        # 5-point finite difference
        hc_5point = HessianCalculator(atoms, atoms.calc, delta=0.01, method="5point", verbose=0)
        hessian_5point = hc_5point.calculate_numerical_hessian()

        # Should match within reasonable tolerance
        np.testing.assert_allclose(hessian_5point, hessian_analytical, rtol=0.05, atol=5.0)
