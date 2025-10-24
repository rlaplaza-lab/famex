"""Test hessian consistency between analytical and finite differences.

This module tests that analytical hessian calculations match finite difference
approximations for backends that support analytical hessians (MACE and UMA).
The tests are optional and will be skipped if the required backends are not available.

The test uses a water molecule and compares:
1. Analytical hessian from the backend's get_hessian() method
2. Finite difference hessian computed by numerical differentiation of forces

Sign convention differences are handled by checking both original and sign-flipped versions.
"""

import numpy as np
import pytest
from ase import Atoms

import qme
from qme.analysis.frequency import HessianCalculator
from qme.backends.availability import is_backend_available
from tests.test_utils import TestMoleculeFactory


class TestHessianConsistency:
    """Test hessian consistency between analytical and finite difference methods."""

    @pytest.fixture
    def water_molecule(self) -> Atoms:
        """Water molecule for hessian testing."""
        return TestMoleculeFactory.get_water_distorted()

    def _compute_finite_difference_hessian(self, atoms: Atoms, delta: float = 0.01) -> np.ndarray:
        """Compute hessian using finite differences via QME's HessianCalculator.

        Parameters
        ----------
        atoms : Atoms
            The molecular system
        delta : float
            Step size for finite differences

        Returns:
        -------
        np.ndarray
            Hessian matrix (3N x 3N)
        """
        # Use QME's HessianCalculator for consistent finite difference calculation
        hessian_calc = HessianCalculator(
            atoms=atoms,
            calculator=atoms.calc,
            delta=delta,
            method="central",
            verbose=0,  # Quiet mode for testing
        )

        return hessian_calc.calculate_numerical_hessian()

    def _compare_hessians(
        self,
        analytical: np.ndarray,
        finite_diff: np.ndarray,
        rtol: float = 1e-1,
        atol: float = 1e-1,
    ) -> None:
        """Compare analytical and finite difference hessians.

        Parameters
        ----------
        analytical : np.ndarray
            Analytical hessian
        finite_diff : np.ndarray
            Finite difference hessian
        rtol : float
            Relative tolerance
        atol : float
            Absolute tolerance
        """
        # Check shapes match
        assert analytical.shape == finite_diff.shape, (
            f"Hessian shapes don't match: {analytical.shape} vs {finite_diff.shape}"
        )

        # Check symmetry (both should be symmetric)
        # Analytical Hessian should be perfectly symmetric
        assert np.allclose(analytical, analytical.T, rtol=rtol, atol=atol), (
            "Analytical hessian is not symmetric"
        )
        # Finite difference Hessian may have numerical noise from ML potentials
        # Use relaxed tolerance for symmetry check
        finite_diff_sym_tol = max(atol, 1e-2)  # At least 1e-2 for ML potential noise
        assert np.allclose(finite_diff, finite_diff.T, rtol=rtol, atol=finite_diff_sym_tol), (
            f"Finite difference hessian is not symmetric (tolerance: {finite_diff_sym_tol})"
        )

        # Compare the hessians
        matches_original = np.allclose(analytical, finite_diff, rtol=rtol, atol=atol)

        assert matches_original, (
            f"Hessians don't match within tolerance (rtol={rtol}, atol={atol}) "
            f"in either original or sign-flipped form"
        )

    @pytest.mark.skipif(not is_backend_available("mace"), reason="MACE backend not available")
    def test_mace_hessian_consistency(self, water_molecule: Atoms) -> None:
        """Test MACE analytical hessian matches finite differences."""
        # Set up MACE calculator
        atoms = water_molecule.copy()
        atoms.calc = qme.get_mace_calculator(model_name="mace-omol-0")

        # Ensure calculator is loaded
        atoms.calc.ensure_loaded()

        # Check that MACE supports hessian
        assert "hessian" in atoms.calc.implemented_properties, (
            "MACE calculator should support hessian calculations"
        )

        # Compute analytical hessian
        hessian_analytical = atoms.calc.get_hessian(atoms)

        # Compute finite difference hessian
        hessian_finite_diff = self._compute_finite_difference_hessian(atoms)

        # Compare hessians
        self._compare_hessians(hessian_analytical, hessian_finite_diff)

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_uma_hessian_consistency(self, water_molecule: Atoms) -> None:
        """Test UMA analytical hessian matches finite differences."""
        # Set up UMA calculator
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")

        # Ensure calculator is loaded
        atoms.calc.ensure_loaded()

        # Check that UMA supports hessian
        assert "hessian" in atoms.calc.implemented_properties, (
            "UMA calculator should support hessian calculations"
        )

        # Compute analytical hessian
        hessian_analytical = atoms.calc.get_hessian(atoms)

        # Compute finite difference hessian
        hessian_finite_diff = self._compute_finite_difference_hessian(atoms)

        # Compare hessians
        self._compare_hessians(hessian_analytical, hessian_finite_diff)
