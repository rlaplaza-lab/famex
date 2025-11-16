"""Tests for finite difference schemes."""

from __future__ import annotations

import numpy as np
import pytest

from qme.analysis.finite_differences import (
    CentralDifferenceScheme,
    FivePointCentralDifferenceScheme,
    ForwardDifferenceScheme,
    SevenPointCentralDifferenceScheme,
)
from tests.test_constants import DEFAULT_DELTA


class TestCentralDifferenceScheme:
    """Tests for CentralDifferenceScheme."""

    def test_compute_derivative_success(self):
        """Test successful computation with both forces."""
        scheme = CentralDifferenceScheme()
        forces_plus = np.array([1.0, 2.0, 3.0])
        forces_minus = np.array([0.5, 1.5, 2.5])
        delta = DEFAULT_DELTA

        result = scheme.compute_derivative(forces_plus, forces_minus, None, delta)

        assert isinstance(result, np.ndarray)
        assert result.shape == forces_plus.shape
        # Expected: -(1.0 - 0.5) / (2 * DEFAULT_DELTA) = -25.0
        expected = np.array([-25.0, -25.0, -25.0])
        np.testing.assert_allclose(result, expected)

    def test_compute_derivative_missing_forces_minus(self):
        """Test that missing forces_minus raises ValueError."""
        scheme = CentralDifferenceScheme()
        forces_plus = np.array([1.0, 2.0, 3.0])
        delta = DEFAULT_DELTA

        with pytest.raises(ValueError, match="Central difference requires forces_minus"):
            scheme.compute_derivative(forces_plus, None, None, delta)


class TestForwardDifferenceScheme:
    """Tests for ForwardDifferenceScheme."""

    def test_compute_derivative_success(self):
        """Test successful computation with forces_ref."""
        scheme = ForwardDifferenceScheme()
        forces_plus = np.array([1.0, 2.0, 3.0])
        forces_ref = np.array([0.5, 1.5, 2.5])
        delta = DEFAULT_DELTA

        result = scheme.compute_derivative(forces_plus, None, forces_ref, delta)

        assert isinstance(result, np.ndarray)
        assert result.shape == forces_plus.shape
        # Expected: -(1.0 - 0.5) / DEFAULT_DELTA = -50.0
        expected = np.array([-50.0, -50.0, -50.0])
        np.testing.assert_allclose(result, expected)

    def test_compute_derivative_missing_forces_ref(self):
        """Test that missing forces_ref raises ValueError."""
        scheme = ForwardDifferenceScheme()
        forces_plus = np.array([1.0, 2.0, 3.0])
        delta = DEFAULT_DELTA

        with pytest.raises(ValueError, match="Forward difference requires forces_ref"):
            scheme.compute_derivative(forces_plus, None, None, delta)


class TestFivePointCentralDifferenceScheme:
    """Tests for FivePointCentralDifferenceScheme."""

    def test_compute_derivative_success(self):
        """Test successful computation with all required forces."""
        scheme = FivePointCentralDifferenceScheme()
        forces_plus = np.array([1.0, 2.0, 3.0])
        forces_minus = np.array([0.5, 1.5, 2.5])
        forces_plus2 = np.array([1.5, 2.5, 3.5])
        forces_minus2 = np.array([0.0, 1.0, 2.0])
        delta = DEFAULT_DELTA

        result = scheme.compute_derivative(
            forces_plus,
            forces_minus,
            None,
            delta,
            forces_plus2=forces_plus2,
            forces_minus2=forces_minus2,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == forces_plus.shape

    def test_compute_derivative_missing_forces_minus(self):
        """Test that missing forces_minus raises ValueError."""
        scheme = FivePointCentralDifferenceScheme()
        forces_plus = np.array([1.0, 2.0, 3.0])
        delta = DEFAULT_DELTA

        with pytest.raises(ValueError, match="5-point central difference requires forces_minus"):
            scheme.compute_derivative(forces_plus, None, None, delta)

    def test_compute_derivative_missing_forces_plus2(self):
        """Test that missing forces_plus2 raises ValueError."""
        scheme = FivePointCentralDifferenceScheme()
        forces_plus = np.array([1.0, 2.0, 3.0])
        forces_minus = np.array([0.5, 1.5, 2.5])
        forces_minus2 = np.array([0.0, 1.0, 2.0])
        delta = DEFAULT_DELTA

        with pytest.raises(
            ValueError, match="5-point central difference requires forces_plus2 and forces_minus2"
        ):
            scheme.compute_derivative(
                forces_plus, forces_minus, None, delta, forces_minus2=forces_minus2
            )

    def test_compute_derivative_missing_forces_minus2(self):
        """Test that missing forces_minus2 raises ValueError."""
        scheme = FivePointCentralDifferenceScheme()
        forces_plus = np.array([1.0, 2.0, 3.0])
        forces_minus = np.array([0.5, 1.5, 2.5])
        forces_plus2 = np.array([1.5, 2.5, 3.5])
        delta = DEFAULT_DELTA

        with pytest.raises(
            ValueError, match="5-point central difference requires forces_plus2 and forces_minus2"
        ):
            scheme.compute_derivative(
                forces_plus, forces_minus, None, delta, forces_plus2=forces_plus2
            )

    def test_compute_derivative_missing_both_2delta_forces(self):
        """Test that missing both forces_plus2 and forces_minus2 raises ValueError."""
        scheme = FivePointCentralDifferenceScheme()
        forces_plus = np.array([1.0, 2.0, 3.0])
        forces_minus = np.array([0.5, 1.5, 2.5])
        delta = DEFAULT_DELTA

        with pytest.raises(
            ValueError, match="5-point central difference requires forces_plus2 and forces_minus2"
        ):
            scheme.compute_derivative(forces_plus, forces_minus, None, delta)


class TestSevenPointCentralDifferenceScheme:
    """Tests for SevenPointCentralDifferenceScheme."""

    def test_compute_derivative_success(self):
        """Test successful computation with all required forces."""
        scheme = SevenPointCentralDifferenceScheme()
        forces_plus = np.array([1.0, 2.0, 3.0])
        forces_minus = np.array([0.5, 1.5, 2.5])
        forces_plus2 = np.array([1.5, 2.5, 3.5])
        forces_minus2 = np.array([0.0, 1.0, 2.0])
        forces_plus3 = np.array([2.0, 3.0, 4.0])
        forces_minus3 = np.array([-0.5, 0.5, 1.5])
        delta = DEFAULT_DELTA

        result = scheme.compute_derivative(
            forces_plus,
            forces_minus,
            None,
            delta,
            forces_plus2=forces_plus2,
            forces_minus2=forces_minus2,
            forces_plus3=forces_plus3,
            forces_minus3=forces_minus3,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == forces_plus.shape

    def test_compute_derivative_missing_forces_minus(self):
        """Test that missing forces_minus raises ValueError."""
        scheme = SevenPointCentralDifferenceScheme()
        forces_plus = np.array([1.0, 2.0, 3.0])
        delta = DEFAULT_DELTA

        with pytest.raises(ValueError, match="7-point central difference requires forces_minus"):
            scheme.compute_derivative(forces_plus, None, None, delta)

    def test_compute_derivative_missing_forces_plus2(self):
        """Test that missing forces_plus2 raises ValueError."""
        scheme = SevenPointCentralDifferenceScheme()
        forces_plus = np.array([1.0, 2.0, 3.0])
        forces_minus = np.array([0.5, 1.5, 2.5])
        forces_minus2 = np.array([0.0, 1.0, 2.0])
        forces_plus3 = np.array([2.0, 3.0, 4.0])
        forces_minus3 = np.array([-0.5, 0.5, 1.5])
        delta = DEFAULT_DELTA

        with pytest.raises(ValueError, match="7-point central difference requires"):
            scheme.compute_derivative(
                forces_plus,
                forces_minus,
                None,
                delta,
                forces_minus2=forces_minus2,
                forces_plus3=forces_plus3,
                forces_minus3=forces_minus3,
            )

    def test_compute_derivative_missing_forces_minus2(self):
        """Test that missing forces_minus2 raises ValueError."""
        scheme = SevenPointCentralDifferenceScheme()
        forces_plus = np.array([1.0, 2.0, 3.0])
        forces_minus = np.array([0.5, 1.5, 2.5])
        forces_plus2 = np.array([1.5, 2.5, 3.5])
        forces_plus3 = np.array([2.0, 3.0, 4.0])
        forces_minus3 = np.array([-0.5, 0.5, 1.5])
        delta = DEFAULT_DELTA

        with pytest.raises(ValueError, match="7-point central difference requires"):
            scheme.compute_derivative(
                forces_plus,
                forces_minus,
                None,
                delta,
                forces_plus2=forces_plus2,
                forces_plus3=forces_plus3,
                forces_minus3=forces_minus3,
            )

    def test_compute_derivative_missing_forces_plus3(self):
        """Test that missing forces_plus3 raises ValueError."""
        scheme = SevenPointCentralDifferenceScheme()
        forces_plus = np.array([1.0, 2.0, 3.0])
        forces_minus = np.array([0.5, 1.5, 2.5])
        forces_plus2 = np.array([1.5, 2.5, 3.5])
        forces_minus2 = np.array([0.0, 1.0, 2.0])
        forces_minus3 = np.array([-0.5, 0.5, 1.5])
        delta = DEFAULT_DELTA

        with pytest.raises(ValueError, match="7-point central difference requires"):
            scheme.compute_derivative(
                forces_plus,
                forces_minus,
                None,
                delta,
                forces_plus2=forces_plus2,
                forces_minus2=forces_minus2,
                forces_minus3=forces_minus3,
            )

    def test_compute_derivative_missing_forces_minus3(self):
        """Test that missing forces_minus3 raises ValueError."""
        scheme = SevenPointCentralDifferenceScheme()
        forces_plus = np.array([1.0, 2.0, 3.0])
        forces_minus = np.array([0.5, 1.5, 2.5])
        forces_plus2 = np.array([1.5, 2.5, 3.5])
        forces_minus2 = np.array([0.0, 1.0, 2.0])
        forces_plus3 = np.array([2.0, 3.0, 4.0])
        delta = DEFAULT_DELTA

        with pytest.raises(ValueError, match="7-point central difference requires"):
            scheme.compute_derivative(
                forces_plus,
                forces_minus,
                None,
                delta,
                forces_plus2=forces_plus2,
                forces_minus2=forces_minus2,
                forces_plus3=forces_plus3,
            )

    def test_compute_derivative_missing_multiple_kwargs(self):
        """Test that missing multiple kwargs raises ValueError with all missing listed."""
        scheme = SevenPointCentralDifferenceScheme()
        forces_plus = np.array([1.0, 2.0, 3.0])
        forces_minus = np.array([0.5, 1.5, 2.5])
        delta = DEFAULT_DELTA

        with pytest.raises(ValueError, match="7-point central difference requires"):
            scheme.compute_derivative(forces_plus, forces_minus, None, delta)
