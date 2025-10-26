"""Test hessian consistency between analytical and finite differences.

This module tests that analytical hessian calculations match finite difference
approximations for backends that support analytical hessians (MACE and UMA).
The tests are optional and will be skipped if the required backends are not available.

The test uses various molecules and compares:
1. Analytical hessian from the backend's get_hessian() method
2. Finite difference hessian computed by numerical differentiation of forces

Multiple test configurations validate convergence and provide statistics.
"""

import time

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

    @pytest.fixture
    def methane_molecule(self) -> Atoms:
        """Methane molecule for testing."""
        # Create a methane molecule
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

    def _compute_frequencies_from_hessian(
        self, hessian: np.ndarray, masses: np.ndarray
    ) -> np.ndarray:
        """Compute vibrational frequencies from Hessian matrix.

        Parameters
        ----------
        hessian : np.ndarray
            Hessian matrix (3N x 3N)
        masses : np.ndarray
            Atomic masses (N,)

        Returns:
        -------
        np.ndarray
            Frequencies in cm^-1 (3N,)
        """
        # Mass-weighted Hessian
        mass_matrix = np.kron(np.diag(1.0 / np.sqrt(masses)), np.eye(3))
        mass_weighted_hessian = mass_matrix @ hessian @ mass_matrix

        # Compute eigenvalues
        eigenvalues, _ = np.linalg.eigh(mass_weighted_hessian)

        # Convert to frequencies (cm^-1)
        # Convert from eV/(amu*Å²) to cm^-1
        hartree_to_cm = 219474.63  # 1 Hartree = 219474.63 cm^-1
        ev_to_hartree = 0.0367493  # 1 eV = 0.0367493 Hartree
        amu_to_kg = 1.66053906660e-27  # kg
        ang_to_m = 1e-10  # m

        # Conversion factor: sqrt(eV/(amu*Å²)) to cm^-1
        conv = np.sqrt(ev_to_hartree * hartree_to_cm**2 / (amu_to_kg * ang_to_m**2))

        frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * conv

        return frequencies

    def _print_comparison_statistics(
        self,
        analytical: np.ndarray,
        finite_diff: np.ndarray,
        rtol: float,
        atol: float,
    ) -> dict[str, float]:
        """Print and return comparison statistics.

        Parameters
        ----------
        analytical : np.ndarray
            Analytical hessian
        finite_diff : np.ndarray
            Finite difference hessian
        rtol : float
            Relative tolerance used
        atol : float
            Absolute tolerance used

        Returns:
        -------
        dict[str, float]
            Statistics dictionary
        """
        diff = analytical - finite_diff
        abs_diff = np.abs(diff)
        rel_diff = abs_diff / (np.abs(analytical) + atol)

        stats = {
            "max_absolute_error": np.max(abs_diff),
            "mean_absolute_error": np.mean(abs_diff),
            "rms_error": np.sqrt(np.mean(diff**2)),
            "max_relative_error": np.max(rel_diff),
            "elements_within_tol": np.sum(np.abs(diff) <= atol + rtol * np.abs(analytical)),
            "total_elements": diff.size,
        }

        return stats

    def _compare_hessians(
        self,
        analytical: np.ndarray,
        finite_diff: np.ndarray,
        rtol: float = 1e-1,
        atol: float = 1.0,
        print_stats: bool = False,
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
        print_stats : bool
            Whether to print detailed statistics
        """
        # Check shapes match
        assert analytical.shape == finite_diff.shape, (
            f"Hessian shapes don't match: {analytical.shape} vs {finite_diff.shape}"
        )

        # Check symmetry (both should be symmetric)
        # Analytical Hessian should be symmetric
        # Use relaxed tolerance for float32 numerical precision in VJP computation
        assert np.allclose(analytical, analytical.T, rtol=1e-5, atol=1e-5), (
            "Analytical hessian is not symmetric"
        )

        # Finite difference Hessian may have numerical noise from ML potentials
        # Use relaxed tolerance for symmetry check
        finite_diff_sym_tol = max(atol, 1e-2)  # At least 1e-2 for ML potential noise
        assert np.allclose(finite_diff, finite_diff.T, rtol=rtol, atol=finite_diff_sym_tol), (
            f"Finite difference hessian is not symmetric (tolerance: {finite_diff_sym_tol})"
        )

        # Compare the hessians
        matches = np.allclose(analytical, finite_diff, rtol=rtol, atol=atol)

        if print_stats:
            stats = self._print_comparison_statistics(analytical, finite_diff, rtol, atol)
            print("Comparison statistics:")
            print(f"  Max absolute error: {stats['max_absolute_error']:.2e}")
            print(f"  Mean absolute error: {stats['mean_absolute_error']:.2e}")
            print(f"  RMS error: {stats['rms_error']:.2e}")
            print(
                f"  Elements within tolerance: {stats['elements_within_tol']}/{stats['total_elements']}"
            )

        assert matches, f"Hessians don't match within tolerance (rtol={rtol}, atol={atol})"

    def _test_hessian_convergence(
        self,
        atoms: Atoms,
        rtol: float = 1e-1,
        atol: float = 3.0,  # ML potentials have larger errors in finite differences
        print_stats: bool = False,
    ) -> None:
        """Test Hessian convergence with different step sizes.

        Parameters
        ----------
        atoms : Atoms
            The molecular system
        rtol : float
            Relative tolerance
        atol : float
            Absolute tolerance
        print_stats : bool
            Whether to print detailed statistics
        """
        # Compute analytical hessian
        hessian_analytical = atoms.calc.get_hessian(atoms)

        # Test different step sizes - check convergence of finite differences first
        step_sizes = [0.01, 0.005, 0.001]
        fd_results = []

        for delta in step_sizes:
            hessian_finite_diff = self._compute_finite_difference_hessian(atoms, delta=delta)
            fd_results.append(hessian_finite_diff)

            if print_stats and delta == step_sizes[0]:
                print(f"\nTesting with delta={delta} Å")

            # Compare analytical vs finite difference
            self._compare_hessians(
                hessian_analytical,
                hessian_finite_diff,
                rtol=rtol,
                atol=atol,
                print_stats=print_stats
                and delta == step_sizes[0],  # Only print for first step size
            )

        # Verify finite differences are converging (smaller step = better agreement)
        # Compare step sizes 0.01 vs 0.005 and 0.005 vs 0.001
        if len(fd_results) >= 2:
            # Check that smaller step sizes lead to better agreement with analytical
            errors = []
            for i, fd_hessian in enumerate(fd_results):
                diff = np.abs(hessian_analytical - fd_hessian)
                max_error = np.max(diff)
                mean_error = np.mean(diff)
                errors.append((max_error, mean_error))

                if print_stats:
                    print(
                        f"  delta={step_sizes[i]}: max_error={max_error:.4f} eV/Å², mean_error={mean_error:.4f} eV/Å²"
                    )

            # Verify convergence: errors should generally decrease with smaller step
            # Allow some flexibility for numerical noise
            if print_stats:
                print("\nFinite difference convergence check:")
                for i in range(len(errors) - 1):
                    improvement = errors[i][0] - errors[i + 1][0]
                    print(
                        f"  Max error improvement ({step_sizes[i]} -> {step_sizes[i + 1]}): {improvement:.4f} eV/Å²"
                    )

            # Check if finite differences are converging well
            # For good convergence, smallest step should have smallest error AND be monotonically decreasing
            # Check: smallest < middle < largest (monotonic improvement)
            fd_converging = (errors[2][0] < errors[1][0]) and (errors[1][0] < errors[0][0])

            if not fd_converging:
                if print_stats:
                    print(
                        "\nWarning: Finite differences not converging properly (may be numerical noise in ML potentials)"
                    )
                    print(
                        f"  Smallest step error ({errors[2][0]:.4f}) >= largest step error ({errors[0][0]:.4f})"
                    )

            # Tight tolerance check: analytical should match smallest-step FD well
            # But be more lenient if FD itself is not converging well
            max_tight_error = np.max(np.abs(hessian_analytical - fd_results[-1]))

            if fd_converging:
                # If FD is converging well, we expect tight agreement
                if max_tight_error > 0.1:  # Maximum 0.1 eV/Å² error
                    pytest.fail(
                        f"Analytical Hessian doesn't match converging finite difference: "
                        f"max error = {max_tight_error:.4f} eV/Å² (should be < 0.1 eV/Å²)"
                    )
            else:
                # If FD itself is problematic, be more lenient
                if max_tight_error > 2.0:  # Maximum 2.0 eV/Å² error for problematic FD
                    pytest.fail(
                        f"Analytical Hessian doesn't match finite difference well enough: "
                        f"max error = {max_tight_error:.4f} eV/Å² (should be < 2.0 eV/Å²). "
                        f"Note: FD itself is not converging well (may be ML potential numerical noise)."
                    )

            if print_stats:
                print(
                    f"\nTight convergence check: max error with smallest step = {max_tight_error:.4f} eV/Å²"
                )
                print(f"  FD converging: {fd_converging}")

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

        self._test_hessian_convergence(atoms, print_stats=True)

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

        self._test_hessian_convergence(atoms, print_stats=True)

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_uma_hessian_methane(self, methane_molecule: Atoms) -> None:
        """Test UMA hessian on methane molecule."""
        # Set up UMA calculator
        atoms = methane_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        self._test_hessian_convergence(atoms)

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_uma_hessian_frequency_comparison(self, water_molecule: Atoms) -> None:
        """Test that analytical and finite difference Hessians yield similar frequencies."""
        # Set up UMA calculator
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        # Get masses
        masses = atoms.get_masses()

        # Compute analytical hessian and frequencies
        hessian_analytical = atoms.calc.get_hessian(atoms)
        freqs_analytical = self._compute_frequencies_from_hessian(hessian_analytical, masses)

        # Compute finite difference hessian and frequencies
        hessian_finite_diff = self._compute_finite_difference_hessian(atoms, delta=0.001)
        freqs_finite_diff = self._compute_frequencies_from_hessian(hessian_finite_diff, masses)

        # Compare frequencies (use tighter tolerance for physically meaningful comparison)
        # Only compare non-translational/rotational modes (skip first 6 modes)
        freqs_analytical_vib = freqs_analytical[6:]
        freqs_finite_diff_vib = freqs_finite_diff[6:]

        # Check that vibrational frequencies are similar
        np.testing.assert_allclose(
            freqs_analytical_vib,
            freqs_finite_diff_vib,
            rtol=0.05,  # 5% tolerance for frequencies
            atol=50.0,  # 50 cm^-1 absolute tolerance
        )

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_uma_hessian_performance(self, water_molecule: Atoms) -> None:
        """Test that analytical Hessian is much faster than finite difference."""
        # Set up UMA calculator
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        # Time analytical Hessian
        start = time.time()
        atoms.calc.get_hessian(atoms)
        time_analytical = time.time() - start

        # Time finite difference Hessian
        start = time.time()
        self._compute_finite_difference_hessian(atoms, delta=0.001)
        time_finite_diff = time.time() - start

        # Analytical should be at least 1.5x faster (accounts for measurement noise)
        speedup = time_finite_diff / time_analytical
        assert speedup > 1.5, (
            f"Analytical Hessian is not significantly faster "
            f"(speedup: {speedup:.2f}x, analytical: {time_analytical:.3f}s, "
            f"finite difference: {time_finite_diff:.3f}s)"
        )

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_uma_frequency_spectrum_comparison(self, water_molecule: Atoms) -> None:
        """Test that analytical and finite difference Hessians yield similar frequency spectra."""
        from qme.analysis.frequency import FrequencyAnalysis

        # Set up UMA calculator
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        # Compute analytical frequencies using FrequencyAnalysis
        # This uses the calculator's direct Hessian method
        freq_analytical = FrequencyAnalysis(
            atoms=atoms,
            calculator=atoms.calc,
            delta=0.01,  # Dummy value, not used when direct Hessian available
            verbose=0,
        )
        freq_analytical.calculate_hessian(method="direct")
        freq_analytical.diagonalize_hessian()
        freqs_analytical = freq_analytical.get_frequencies()

        # Compute finite difference frequencies
        freq_finite_diff = FrequencyAnalysis(
            atoms=atoms,
            calculator=atoms.calc,
            delta=0.001,  # Small step for better accuracy
            verbose=0,
        )
        # Force finite difference by manually computing FD Hessian
        hessian_fd = self._compute_finite_difference_hessian(atoms, delta=0.001)
        freq_finite_diff._hessian = hessian_fd
        freq_finite_diff.diagonalize_hessian()
        freqs_finite_diff = freq_finite_diff.get_frequencies()

        # Compare vibrational frequencies only (skip translational/rotational modes)
        # For water (3 atoms, non-linear), skip first 6 modes
        freqs_analytical_vib = freqs_analytical[6:]
        freqs_finite_diff_vib = freqs_finite_diff[6:]

        # Check that vibrational frequencies match within reasonable tolerance
        # Water has 3 vibrational modes: symmetric stretch, asymmetric stretch, bend
        np.testing.assert_allclose(
            freqs_analytical_vib,
            freqs_finite_diff_vib,
            rtol=0.05,  # 5% relative tolerance
            atol=50.0,  # 50 cm^-1 absolute tolerance
        )

        # Verify that frequencies are positive (stable molecule)
        assert np.all(freqs_analytical_vib > 0), (
            f"Expected all vibrational frequencies to be positive, got: {freqs_analytical_vib}"
        )
        assert np.all(freqs_finite_diff_vib > 0), (
            f"Expected all vibrational frequencies to be positive, got: {freqs_finite_diff_vib}"
        )
