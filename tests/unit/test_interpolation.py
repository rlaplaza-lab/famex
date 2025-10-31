"""Unit tests for interpolation strategies."""

from __future__ import annotations

import numpy as np
import pytest

from qme.interpolation.strategies import (
    CubicSplineInterpolation,
    GeodesicInterpolation,
    IDPPInterpolation,
    LinearInterpolation,
    QuadraticInterpolation,
    get_interpolation_strategy,
    list_interpolation_methods,
)


class TestLinearInterpolation:
    """Test linear interpolation strategy."""

    def test_linear_interpolation_basic(self) -> None:
        """Test basic linear interpolation."""
        interp = LinearInterpolation()
        start = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        end = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        path = interp.interpolate(start, end, npoints=3)

        assert len(path) == 3
        assert np.allclose(path[0], start)
        assert np.allclose(path[-1], end)
        # Middle point should be average
        assert np.allclose(path[1], 0.5 * (start + end))

    def test_linear_interpolation_continuity(self) -> None:
        """Test that linear interpolation produces continuous paths."""
        interp = LinearInterpolation()
        start = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        end = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        path = interp.interpolate(start, end, npoints=10)

        # Check endpoints
        assert np.allclose(path[0], start)
        assert np.allclose(path[-1], end)

        # Check that path is monotonic for each atom
        for atom_idx in range(len(start)):
            x_coords = [p[atom_idx, 0] for p in path]
            assert all(x_coords[i] <= x_coords[i + 1] for i in range(len(x_coords) - 1))


class TestGeodesicInterpolation:
    """Test geodesic interpolation strategy."""

    def test_geodesic_interpolation_basic(self) -> None:
        """Test basic geodesic interpolation."""
        interp = GeodesicInterpolation()
        start = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        end = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        path = interp.interpolate(start, end, npoints=5)

        assert len(path) == 5
        assert np.allclose(path[0], start, atol=1e-2)
        assert np.allclose(path[-1], end, atol=1e-2)

    def test_geodesic_distance_matrix(self) -> None:
        """Test distance matrix calculation."""
        interp = GeodesicInterpolation()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        dist_matrix = interp._get_distance_matrix(coords)

        assert dist_matrix.shape == (3, 3)
        assert np.allclose(dist_matrix[0, 1], 1.0)
        assert np.allclose(dist_matrix[0, 2], 1.0)
        assert np.allclose(dist_matrix[1, 2], np.sqrt(2.0))

    def test_refine_coordinates(self) -> None:
        """Test coordinate refinement."""
        interp = GeodesicInterpolation()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        target_dists = np.array([[0.0, 2.0], [2.0, 0.0]])

        refined = interp._refine_coordinates(coords.copy(), target_dists, max_iter=5)

        assert refined.shape == coords.shape
        # Distance should be closer to target
        refined_dist = np.linalg.norm(refined[1] - refined[0])
        assert abs(refined_dist - 2.0) < abs(np.linalg.norm(coords[1] - coords[0]) - 2.0)


class TestIDPPInterpolation:
    """Test IDPP interpolation strategy."""

    def test_idpp_interpolation_basic(self) -> None:
        """Test basic IDPP interpolation."""
        interp = IDPPInterpolation()
        start = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        end = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        path = interp.interpolate(start, end, npoints=5)

        assert len(path) == 5
        assert np.allclose(path[0], start, atol=1e-2)
        assert np.allclose(path[-1], end, atol=1e-2)

    def test_idpp_preserves_endpoints(self) -> None:
        """Test that IDPP preserves endpoints exactly."""
        interp = IDPPInterpolation()
        start = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        end = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        path = interp.interpolate(start, end, npoints=5)

        # First and last should be exact (not refined)
        assert np.allclose(path[0], start)
        assert np.allclose(path[-1], end)

    def test_get_distance_matrix(self) -> None:
        """Test IDPP distance matrix calculation."""
        interp = IDPPInterpolation()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        dist_matrix = interp._get_distance_matrix(coords)

        assert dist_matrix.shape == (2, 2)
        assert np.allclose(dist_matrix[0, 1], 1.0)


class TestQuadraticInterpolation:
    """Test quadratic interpolation strategy."""

    def test_quadratic_interpolation_basic(self) -> None:
        """Test basic quadratic interpolation."""
        interp = QuadraticInterpolation()
        start = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        end = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        path = interp.interpolate(start, end, npoints=5)

        assert len(path) == 5
        assert np.allclose(path[0], start)
        assert np.allclose(path[-1], end)

    def test_quadratic_interpolate_function(self) -> None:
        """Test quadratic interpolation function."""
        interp = QuadraticInterpolation()
        start = np.array([[0.0, 0.0, 0.0]])
        end = np.array([[1.0, 0.0, 0.0]])

        # At t=0, should be start
        result = interp._quadratic_interpolate(start, end, 0.0)
        assert np.allclose(result, start)

        # At t=1, should be end
        result = interp._quadratic_interpolate(start, end, 1.0)
        assert np.allclose(result, end)


class TestCubicSplineInterpolation:
    """Test cubic spline interpolation strategy."""

    def test_cubic_spline_interpolation_basic(self) -> None:
        """Test basic cubic spline interpolation."""
        interp = CubicSplineInterpolation()
        start = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        end = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        path = interp.interpolate(start, end, npoints=5)

        assert len(path) == 5
        assert np.allclose(path[0], start)
        assert np.allclose(path[-1], end)

    def test_cubic_spline_interpolate_function(self) -> None:
        """Test cubic spline interpolation function."""
        interp = CubicSplineInterpolation()
        control_points = [
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[2.0, 0.0, 0.0]]),
            np.array([[3.0, 0.0, 0.0]]),
        ]

        # At t=0, should be first control point
        result = interp._cubic_spline_interpolate(control_points, 0.0)
        assert np.allclose(result, control_points[0], atol=1e-6)

        # At t=1, should be last control point
        result = interp._cubic_spline_interpolate(control_points, 1.0)
        assert np.allclose(result, control_points[-1], atol=1e-6)

    def test_cubic_spline_wrong_number_points(self) -> None:
        """Test cubic spline with wrong number of control points."""
        interp = CubicSplineInterpolation()
        control_points = [
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0]]),
        ]

        with pytest.raises(ValueError, match="exactly 4 control points"):
            interp._cubic_spline_interpolate(control_points, 0.5)


class TestInterpolationRegistry:
    """Test interpolation strategy registry."""

    def test_get_interpolation_strategy_linear(self) -> None:
        """Test getting linear interpolation strategy."""
        strategy = get_interpolation_strategy("linear")
        assert isinstance(strategy, LinearInterpolation)

    def test_get_interpolation_strategy_geodesic(self) -> None:
        """Test getting geodesic interpolation strategy."""
        strategy = get_interpolation_strategy("geodesic")
        assert isinstance(strategy, GeodesicInterpolation)

    def test_get_interpolation_strategy_idpp(self) -> None:
        """Test getting IDPP interpolation strategy."""
        strategy = get_interpolation_strategy("idpp")
        assert isinstance(strategy, IDPPInterpolation)

    def test_get_interpolation_strategy_case_insensitive(self) -> None:
        """Test that strategy lookup is case-insensitive."""
        strategy1 = get_interpolation_strategy("LINEAR")
        strategy2 = get_interpolation_strategy("linear")
        strategy3 = get_interpolation_strategy("Linear")

        assert isinstance(strategy1, LinearInterpolation)
        assert isinstance(strategy2, LinearInterpolation)
        assert isinstance(strategy3, LinearInterpolation)

    def test_get_interpolation_strategy_unknown(self) -> None:
        """Test getting unknown interpolation strategy raises error."""
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            get_interpolation_strategy("unknown_method")

    def test_list_interpolation_methods(self) -> None:
        """Test listing available interpolation methods."""
        methods = list_interpolation_methods()

        assert isinstance(methods, dict)
        assert "linear" in methods
        assert "geodesic" in methods
        assert "idpp" in methods
        assert "quadratic" in methods
        assert "spline" in methods

        # Check that all methods have descriptions
        for _method, description in methods.items():
            assert isinstance(description, str)
            assert len(description) > 0


class TestInterpolationPathQuality:
    """Test path quality properties across interpolation methods."""

    @pytest.mark.parametrize(
        ("method", "tolerance"),
        [
            ("linear", 1e-10),
            ("geodesic", 1e-2),
            ("idpp", 1e-2),
            ("quadratic", 1e-6),
            ("spline", 1e-6),
        ],
    )
    def test_all_methods_preserve_endpoints(self, method: str, tolerance: float) -> None:
        """Test that all interpolation methods preserve endpoints."""
        strategy = get_interpolation_strategy(method)
        start = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        end = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        path = strategy.interpolate(start, end, npoints=7)

        assert np.allclose(path[0], start, atol=tolerance)
        assert np.allclose(path[-1], end, atol=tolerance)

    @pytest.mark.parametrize("method", ["linear", "geodesic", "idpp", "quadratic", "spline"])
    def test_all_methods_produce_correct_length(self, method: str) -> None:
        """Test that all methods produce paths of correct length."""
        strategy = get_interpolation_strategy(method)
        start = np.array([[0.0, 0.0, 0.0]])
        end = np.array([[1.0, 0.0, 0.0]])

        for npoints in [3, 5, 10]:
            path = strategy.interpolate(start, end, npoints=npoints)
            assert len(path) == npoints

    def test_path_smoothness_linear(self) -> None:
        """Test that linear interpolation produces smooth paths."""
        strategy = LinearInterpolation()
        start = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        end = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        path = strategy.interpolate(start, end, npoints=10)

        # Check that differences between consecutive points are similar
        # (indicating smoothness)
        for i in range(len(path) - 2):
            diff1 = np.linalg.norm(path[i + 1] - path[i])
            diff2 = np.linalg.norm(path[i + 2] - path[i + 1])

            # For linear interpolation, differences should be very similar
            assert abs(diff1 - diff2) < 1e-6

    def test_interpolation_with_different_system_sizes(self) -> None:
        """Test interpolation works with different system sizes."""
        strategy = LinearInterpolation()

        # Test with 1 atom
        start = np.array([[0.0, 0.0, 0.0]])
        end = np.array([[1.0, 0.0, 0.0]])
        path = strategy.interpolate(start, end, npoints=5)
        assert len(path) == 5
        assert path[0].shape == (1, 3)

        # Test with multiple atoms
        start = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        end = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        path = strategy.interpolate(start, end, npoints=5)
        assert len(path) == 5
        assert path[0].shape == (3, 3)
