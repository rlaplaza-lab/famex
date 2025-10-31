"""Test 5-point finite difference Hessian calculation.

This module tests the 5-point finite difference scheme for Hessian calculations,
including accuracy comparisons and Richardson extrapolation.
"""

import numpy as np
import pytest
from ase import Atoms

import qme
from qme.analysis.frequency import HessianCalculator
from qme.backends.availability import is_backend_available
from tests.test_utils import TestMoleculeFactory


def harmonic_potential_hessian(atoms: Atoms) -> np.ndarray:
    """Analytical Hessian for a harmonic potential.

    For a harmonic potential E = 0.5 * k * sum(r_i^2),
    the Hessian is diagonal with entries k for each coordinate.
    """
    k = 1.0  # Force constant
    n_atoms = len(atoms)
    n_coords = 3 * n_atoms
    hessian = k * np.eye(n_coords)
    return hessian


def harmonic_potential_forces(atoms: Atoms, k: float = 1.0) -> np.ndarray:
    """Compute forces for a harmonic potential.

    E = 0.5 * k * sum(r_i^2)
    F = -∇E = -k * r
    """
    forces = -k * atoms.positions
    return forces


class HarmonicCalculator:
    """Mock calculator for harmonic potential for testing."""

    def __init__(self, k: float = 1.0) -> None:
        """Initialize with force constant k."""
        self.k = k

    def get_forces(self, atoms: Atoms) -> np.ndarray:
        """Compute harmonic forces."""
        return harmonic_potential_forces(atoms, self.k)

    def get_hessian(self, atoms: Atoms) -> np.ndarray:
        """Compute analytical harmonic Hessian."""
        return harmonic_potential_hessian(atoms)


class Test5PointHessian:
    """Test 5-point finite difference scheme."""

    @pytest.fixture
    def water_molecule(self) -> Atoms:
        """Water molecule for testing."""
        return TestMoleculeFactory.get_water_distorted()

    @pytest.fixture
    def harmonic_atoms(self) -> Atoms:
        """Simple harmonic system for testing."""
        return Atoms(
            symbols="HH",
            positions=[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]],
        )

    def test_5point_against_analytical_harmonic(self, harmonic_atoms: Atoms) -> None:
        """Test 5-point scheme against analytical harmonic Hessian."""
        calc = HarmonicCalculator(k=1.0)

        # 5-point finite difference
        hc_5point = HessianCalculator(harmonic_atoms, calc, delta=0.01, method="5point", verbose=0)
        hessian_5point = hc_5point.calculate_numerical_hessian()

        # Analytical
        hessian_analytical = calc.get_hessian(harmonic_atoms)

        # Should be very accurate with small delta
        np.testing.assert_allclose(hessian_5point, hessian_analytical, rtol=1e-6, atol=1e-6)

    def test_5point_vs_3point_accuracy(self, harmonic_atoms: Atoms) -> None:
        """Compare 5-point accuracy against 3-point on harmonic potential."""
        calc = HarmonicCalculator(k=1.0)

        # Use a relatively large delta to see difference in accuracy
        delta = 0.05

        # 5-point finite difference
        hc_5point = HessianCalculator(harmonic_atoms, calc, delta=delta, method="5point", verbose=0)
        hessian_5point = hc_5point.calculate_numerical_hessian()

        # 3-point finite difference
        hc_3point = HessianCalculator(
            harmonic_atoms, calc, delta=delta, method="central", verbose=0
        )
        hessian_3point = hc_3point.calculate_numerical_hessian()

        # Analytical
        hessian_analytical = calc.get_hessian(harmonic_atoms)

        # 5-point should be more accurate
        error_5point = np.max(np.abs(hessian_5point - hessian_analytical))
        error_3point = np.max(np.abs(hessian_3point - hessian_analytical))

        assert error_5point < error_3point, "5-point should be more accurate than 3-point"

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

        # Richardson should be more accurate
        error_with_rich = np.max(np.abs(hessian_5point_rich - hessian_analytical))
        error_without_rich = np.max(np.abs(hessian_5point - hessian_analytical))

        assert error_with_rich < error_without_rich, (
            "5-point with Richardson should be more accurate than without"
        )

    def test_convergence_analysis(self, harmonic_atoms: Atoms) -> None:
        """Test convergence rate with decreasing step size."""
        calc = HarmonicCalculator(k=1.0)
        hessian_analytical = calc.get_hessian(harmonic_atoms)

        deltas = [0.1, 0.05, 0.01, 0.005, 0.001]
        errors_3point = []
        errors_5point = []

        for delta in deltas:
            # 3-point
            hc_3point = HessianCalculator(
                harmonic_atoms, calc, delta=delta, method="central", verbose=0
            )
            hessian_3point = hc_3point.calculate_numerical_hessian()
            error_3point = np.max(np.abs(hessian_3point - hessian_analytical))
            errors_3point.append(error_3point)

            # 5-point
            hc_5point = HessianCalculator(
                harmonic_atoms, calc, delta=delta, method="5point", verbose=0
            )
            hessian_5point = hc_5point.calculate_numerical_hessian()
            error_5point = np.max(np.abs(hessian_5point - hessian_analytical))
            errors_5point.append(error_5point)

        # Check that 5-point converges faster (O(h⁴) vs O(h²))
        # The ratio of errors should decrease as delta decreases
        for i in range(1, len(deltas)):
            ratio_3point = errors_3point[i - 1] / errors_3point[i]
            ratio_5point = errors_5point[i - 1] / errors_5point[i]

            # 5-point should converge faster
            assert ratio_5point > ratio_3point, (
                f"5-point should converge faster at delta={deltas[i]}, "
                f"but ratio_5point={ratio_5point} <= ratio_3point={ratio_3point}"
            )

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_5point_against_analytical_uma(self, water_molecule: Atoms) -> None:
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

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_5point_vs_3point_uma(self, water_molecule: Atoms) -> None:
        """Compare 5-point vs 3-point with UMA on water."""
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        # Analytical Hessian as reference
        hessian_analytical = atoms.calc.get_hessian(atoms)

        # Use relatively large delta to see difference
        delta = 0.02

        # 3-point
        hc_3point = HessianCalculator(atoms, atoms.calc, delta=delta, method="central", verbose=0)
        hessian_3point = hc_3point.calculate_numerical_hessian()

        # 5-point
        hc_5point = HessianCalculator(atoms, atoms.calc, delta=delta, method="5point", verbose=0)
        hessian_5point = hc_5point.calculate_numerical_hessian()

        # 5-point should be closer to analytical
        error_3point = np.max(np.abs(hessian_3point - hessian_analytical))
        error_5point = np.max(np.abs(hessian_5point - hessian_analytical))

        assert error_5point < error_3point, "5-point should be more accurate than 3-point"

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_5point_richardson_uma(self, water_molecule: Atoms) -> None:
        """Test 5-point + Richardson with UMA."""
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        # Analytical Hessian as reference
        hessian_analytical = atoms.calc.get_hessian(atoms)

        # 5-point without Richardson
        hc_5point = HessianCalculator(atoms, atoms.calc, delta=0.01, method="5point", verbose=0)
        hessian_5point = hc_5point.calculate_numerical_hessian()

        # 5-point with Richardson
        hc_5point_rich = HessianCalculator(
            atoms, atoms.calc, delta=0.01, method="5point", richardson=True, verbose=0
        )
        hessian_5point_rich = hc_5point_rich.calculate_numerical_hessian()

        # Richardson should be more accurate
        error_without = np.max(np.abs(hessian_5point - hessian_analytical))
        error_with = np.max(np.abs(hessian_5point_rich - hessian_analytical))

        assert error_with < error_without, (
            "5-point with Richardson should be more accurate than without"
        )

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

    def test_richardson_order_detection(self) -> None:
        """Test that Richardson order is correctly detected."""
        atoms = Atoms(symbols="H", positions=[[0, 0, 0]])
        calc = HarmonicCalculator()

        # 3-point + Richardson should have order 2
        hc_3 = HessianCalculator(atoms, calc, method="central", richardson=True, verbose=0)
        assert hc_3._richardson_order == 2

        # 5-point + Richardson should have order 4
        hc_5 = HessianCalculator(atoms, calc, method="5point", richardson=True, verbose=0)
        assert hc_5._richardson_order == 4

    def test_invalid_method(self) -> None:
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
