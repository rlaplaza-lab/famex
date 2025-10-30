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

    def test_atoms_not_modified_in_place(self) -> None:
        """Test that input atoms object is not modified in-place."""
        atoms = self.h2o.copy()
        atoms.calc = qme.MockCalculator(backend="mock")  # Ensure calculator is attached
        original_positions = atoms.positions.copy()
        original_calc_id = id(atoms.calc)

        # Run frequency analysis
        fa = FrequencyAnalysis(atoms, atoms.calc)
        fa.calculate_hessian()
        fa.diagonalize_hessian()

        # Verify positions unchanged
        assert np.allclose(atoms.positions, original_positions), "Atoms positions were modified!"

        # Verify calculator unchanged (same object)
        assert id(atoms.calc) == original_calc_id, "Atoms calculator was replaced!"

        # Test with different calculator
        different_calc = qme.MockCalculator(backend="mock")
        FrequencyAnalysis(atoms, different_calc)

        # Calculator should be updated to different one
        assert id(atoms.calc) == id(different_calc), "Different calculator not applied!"

        # But positions should still be unchanged
        assert np.allclose(atoms.positions, original_positions), (
            "Positions modified with different calc!"
        )


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

    def test_richardson_improves_accuracy_over_central(self) -> None:
        """Test that Richardson extrapolation improves accuracy over central differences."""
        from ase import Atoms

        # Simple H2 molecule aligned on x-axis
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        atoms.calc = qme.MockCalculator(backend="mock")

        # Reference with very small step
        hess_ref = HessianCalculator(
            atoms,
            atoms.calc,
            delta=1e-4,
            method="central",
            richardson=False,
            indices=None,
            verbose=0,
        ).calculate_numerical_hessian()

        # Baseline central with practical step
        hess_central = HessianCalculator(
            atoms,
            atoms.calc,
            delta=0.02,
            method="central",
            richardson=False,
            indices=None,
            verbose=0,
        ).calculate_numerical_hessian()

        # Richardson with two deltas (delta2 defaults to delta/2)
        hess_rich = HessianCalculator(
            atoms,
            atoms.calc,
            delta=0.02,
            method="central",
            richardson=True,
            delta2=None,
            indices=None,
            verbose=0,
        ).calculate_numerical_hessian()

        # Compare errors to reference
        err_central = np.linalg.norm(hess_central - hess_ref)
        err_rich = np.linalg.norm(hess_rich - hess_ref)

        # Richardson should be strictly better
        assert err_rich < err_central * 0.8  # expect noticeable improvement

        # Also check elementwise max error improves
        max_err_central = np.max(np.abs(hess_central - hess_ref))
        max_err_rich = np.max(np.abs(hess_rich - hess_ref))
        assert max_err_rich < max_err_central
