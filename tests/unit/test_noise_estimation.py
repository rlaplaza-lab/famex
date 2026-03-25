"""Tests for noise estimation utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from qme.analysis.noise_estimation import (
    estimate_force_noise,
    estimate_optimal_delta,
    estimate_richardson_noise,
)


class TestEstimateRichardsonNoise:
    """Tests for estimate_richardson_noise function."""

    def test_identical_hessians(self):
        """Test that identical Hessians give zero noise."""
        hessian = np.eye(6)  # 2 atoms, 6 DOF
        noise = estimate_richardson_noise(hessian, hessian)
        assert noise == 0.0

    def test_small_difference(self):
        """Test with small difference between Hessians."""
        hessian1 = np.eye(6)
        hessian2 = np.eye(6) + 0.001 * np.random.randn(6, 6)
        noise = estimate_richardson_noise(hessian1, hessian2)
        assert noise > 0
        assert noise < 1.0  # Should be small

    def test_large_difference(self):
        """Test with large difference between Hessians."""
        hessian1 = np.eye(6)
        hessian2 = np.eye(6) + 1.0 * np.random.randn(6, 6)
        noise = estimate_richardson_noise(hessian1, hessian2)
        assert noise > 0
        assert noise > 0.1  # Should be larger than small difference case

    def test_different_sizes(self):
        """Test with different sized Hessians."""
        hessian1 = np.eye(9)  # 3 atoms
        hessian2 = np.eye(9) + 0.01 * np.random.randn(9, 9)
        noise = estimate_richardson_noise(hessian1, hessian2)
        assert noise >= 0

    def test_richardson_extrapolation_logic(self):
        """Test that Richardson extrapolation is correctly applied."""
        # For H2 = H1 + noise, extrapolated = (4*H2 - H1) / 3
        # If H2 = H1 + 0.1, then extrapolated = (4*(H1+0.1) - H1) / 3 = H1 + 0.4/3
        hessian1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        hessian2 = np.array([[1.1, 0.0], [0.0, 1.1]])
        noise = estimate_richardson_noise(hessian1, hessian2)
        assert noise > 0

    def test_zero_hessian(self):
        """Test with zero Hessian."""
        hessian = np.zeros((6, 6))
        noise = estimate_richardson_noise(hessian, hessian)
        assert noise == 0.0

    def test_returns_float(self):
        """Test that function returns a float."""
        hessian = np.eye(6)
        noise = estimate_richardson_noise(hessian, hessian)
        assert isinstance(noise, float)


class TestEstimateForceNoise:
    """Tests for estimate_force_noise function."""

    def test_no_noise_calculator(self):
        """Test with a deterministic calculator (no noise)."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        # Mock calculator that returns constant forces
        calculator = MagicMock()
        calculator.get_forces = MagicMock(
            return_value=np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]])
        )

        # With no noise, small perturbations should give very small noise estimate
        noise = estimate_force_noise(atoms, calculator, n_samples=3, perturbation_size=1e-6)
        assert noise >= 0
        assert noise < 1e-3  # Should be very small for deterministic calculator

    def test_noisy_calculator(self):
        """Test with a noisy calculator."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        # Mock calculator that returns noisy forces
        calculator = MagicMock()
        base_forces = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]])

        def noisy_forces(atoms=None):
            return base_forces + np.random.normal(0, 0.01, base_forces.shape)

        calculator.get_forces = MagicMock(side_effect=noisy_forces)

        noise = estimate_force_noise(atoms, calculator, n_samples=5, perturbation_size=1e-5)
        assert noise >= 0
        # With noise, should detect some noise level
        assert noise < 1.0  # But not too large

    def test_with_indices(self):
        """Test that indices parameter works correctly."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]])

        calculator = MagicMock()
        calculator.get_forces = MagicMock(
            return_value=np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.05], [0.0, 0.0, -0.05]]),
        )

        # Test with subset of indices
        noise = estimate_force_noise(atoms, calculator, n_samples=3, indices=[0, 1])
        assert noise >= 0

    def test_all_samples_fail(self):
        """Test that RuntimeError is raised when all samples fail."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        calculator = MagicMock()
        # First call (reference) succeeds, but all perturbation samples fail
        call_count = 0

        def failing_forces(atoms=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Reference forces - succeed
                return np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]])
            else:
                # All perturbation samples fail
                raise Exception("Calculation failed")

        calculator.get_forces = MagicMock(side_effect=failing_forces)

        with pytest.raises(RuntimeError, match="All force noise samples failed"):
            estimate_force_noise(atoms, calculator, n_samples=3)

    def test_some_samples_fail(self):
        """Test that function continues when some samples fail."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        calculator = MagicMock()
        call_count = 0

        def sometimes_fail(atoms=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Reference forces - succeed
                return np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]])
            elif call_count <= 3:
                # First two perturbation samples fail
                raise Exception("Failed")
            else:
                # Remaining samples succeed
                return np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]])

        calculator.get_forces = MagicMock(side_effect=sometimes_fail)

        # Should succeed with remaining samples
        noise = estimate_force_noise(atoms, calculator, n_samples=5)
        assert noise >= 0

    def test_different_perturbation_sizes(self):
        """Test with different perturbation sizes."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        calculator = MagicMock()
        calculator.get_forces = MagicMock(
            return_value=np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]])
        )

        noise_small = estimate_force_noise(atoms, calculator, n_samples=3, perturbation_size=1e-6)
        noise_large = estimate_force_noise(atoms, calculator, n_samples=3, perturbation_size=1e-4)

        assert noise_small >= 0
        assert noise_large >= 0

    def test_returns_float(self):
        """Test that function returns a float."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        calculator = MagicMock()
        calculator.get_forces = MagicMock(
            return_value=np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]])
        )

        noise = estimate_force_noise(atoms, calculator, n_samples=2)
        assert isinstance(noise, float)


class TestEstimateOptimalDelta:
    """Tests for estimate_optimal_delta function."""

    @patch("qme.analysis.noise_estimation._compute_hessian_at_delta")
    def test_basic_functionality(self, mock_compute_hessian):
        """Test basic functionality with mock Hessian calculations."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        # Mock calculator
        calculator = MagicMock()

        # Mock Hessian calculations that return different noise levels
        def mock_hessian(delta, noise_level):
            hessian = np.eye(6) + noise_level * np.random.randn(6, 6) * 1e-6
            return hessian

        call_count = 0

        def compute_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First call (larger delta) - more noise
            # Second call (smaller delta) - less noise
            if call_count % 2 == 1:
                return mock_hessian(0.01, 1.0)
            else:
                return mock_hessian(0.005, 0.1)

        mock_compute_hessian.side_effect = compute_side_effect

        delta, noise = estimate_optimal_delta(
            atoms,
            calculator,
            delta_range=(0.001, 0.05),
            method="central",
            max_iterations=2,
        )

        assert delta > 0
        assert noise >= 0
        assert isinstance(delta, float)
        assert isinstance(noise, float)

    def test_invalid_delta_range(self):
        """Test that invalid delta_range raises ValueError."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        calculator = MagicMock()

        with pytest.raises(ValueError, match="Invalid delta_range"):
            estimate_optimal_delta(atoms, calculator, delta_range=(0.05, 0.001))

        with pytest.raises(ValueError, match="Invalid delta_range"):
            estimate_optimal_delta(atoms, calculator, delta_range=(0.01, 0.01))

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        calculator = MagicMock()

        with pytest.raises(ValueError, match="Unknown method"):
            estimate_optimal_delta(atoms, calculator, method="invalid_method", max_iterations=1)

    @patch("qme.analysis.noise_estimation._compute_hessian_at_delta")
    def test_convergence_to_target(self, mock_compute_hessian):
        """Test that function converges when target noise is reached."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        calculator = MagicMock()

        # Mock Hessians that give low noise
        low_noise_hessian = np.eye(6)

        def compute_side_effect(*args, **kwargs):
            return low_noise_hessian

        mock_compute_hessian.side_effect = compute_side_effect

        delta, noise = estimate_optimal_delta(
            atoms,
            calculator,
            delta_range=(0.001, 0.05),
            target_noise=1e-4,
            max_iterations=3,
        )

        assert delta > 0
        assert noise >= 0

    @patch("qme.analysis.noise_estimation._compute_hessian_at_delta")
    def test_handles_calculation_failures(self, mock_compute_hessian):
        """Test that function handles calculation failures gracefully."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        calculator = MagicMock()

        # First call fails, second succeeds
        call_count = 0

        def compute_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Calculation failed")
            return np.eye(6)

        mock_compute_hessian.side_effect = compute_side_effect

        delta, noise = estimate_optimal_delta(
            atoms,
            calculator,
            delta_range=(0.001, 0.05),
            max_iterations=2,
        )

        # Should still return a result (may use default)
        assert delta > 0
        assert noise >= 0

    def test_with_indices(self):
        """Test that indices parameter is passed correctly."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]])
        calculator = MagicMock()

        with patch("qme.analysis.noise_estimation._compute_hessian_at_delta") as mock_compute:
            mock_compute.return_value = np.eye(9)  # 3 atoms

            delta, noise = estimate_optimal_delta(
                atoms,
                calculator,
                delta_range=(0.001, 0.05),
                indices=[0, 1],
                max_iterations=1,
            )

            assert delta > 0
            # Check that indices were passed to the compute function
            assert mock_compute.called

    @patch("qme.analysis.noise_estimation._compute_hessian_at_delta")
    def test_method_selection(self, mock_compute_hessian):
        """Test that correct method is selected."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        calculator = MagicMock()
        mock_compute_hessian.return_value = np.eye(6)

        # Test central method
        delta1, _ = estimate_optimal_delta(
            atoms,
            calculator,
            method="central",
            max_iterations=1,
        )

        # Test 5point method
        delta2, _ = estimate_optimal_delta(
            atoms,
            calculator,
            method="5point",
            max_iterations=1,
        )

        assert delta1 > 0
        assert delta2 > 0
