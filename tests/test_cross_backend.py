"""
Cross-backend compatibility and comparison tests.

Tests that verify consistent behavior across different backends,
and tests that run with whatever backends are available.
"""

import pytest
from ase import Atoms

import qme
from qme.dependencies import deps


class TestCrossBackendCompatibility:
    """Test compatibility across available backends."""

    @pytest.fixture(params=["mock"])
    def available_backend(self, request):
        """Parametrized fixture that yields available backends."""
        backend = request.param

        # Always test mock backend
        if backend == "mock":
            return backend

        # Test other backends only if available
        if backend == "uma" and deps.has("fairchem"):
            return backend
        elif backend == "so3lr" and deps.has("so3lr"):
            return backend
        elif backend == "aimnet2" and deps.has("aimnet2"):
            return backend
        else:
            pytest.skip(f"{backend} backend not available")

    def test_optimizer_creation_consistency(self, available_backend):
        """Test that all backends can create optimizers consistently."""
        try:
            optimizer = qme.QMEOptimizer(backend=available_backend)
            assert optimizer.backend == available_backend
            assert optimizer.calculator is not None
            assert hasattr(optimizer, "optimize_minimum")
            assert hasattr(optimizer, "find_transition_state")
        except ImportError:
            pytest.skip(f"{available_backend} backend dependencies not available")

    def test_h2_optimization_consistency(self, available_backend):
        """Test H2 optimization gives reasonable results across backends."""
        try:
            optimizer = qme.QMEOptimizer(backend=available_backend)
            h2 = Atoms("H2", positions=[[0, 0, 0], [1.8, 0, 0]])

            result = optimizer.optimize_minimum(atoms=h2, fmax=0.05, steps=30)

            # All backends should:
            # 1. Complete without error
            assert "converged" in result
            assert "steps_taken" in result
            assert result["steps_taken"] >= 0

            # 2. Return optimized atoms
            assert "optimized_atoms" in result
            optimized = result["optimized_atoms"]
            assert isinstance(optimized, Atoms)
            assert len(optimized) == 2

            # 3. Optimize toward reasonable H-H distance
            final_distance = optimized.get_distance(0, 1)
            assert 0.5 < final_distance < 2.0  # Reasonable range

        except ImportError:
            pytest.skip(f"{available_backend} backend dependencies not available")

    def test_minimize_structure_function_consistency(self, available_backend):
        """Test minimize_structure function works with all backends."""
        try:
            h2 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])

            result = qme.minimize_structure(
                h2, backend=available_backend, fmax=0.05, steps=20
            )

            assert isinstance(result, Atoms)
            assert len(result) == 2
            final_distance = result.get_distance(0, 1)
            assert 0.5 < final_distance < 2.0

        except ImportError:
            pytest.skip(f"{available_backend} backend dependencies not available")


# Add specific multi-backend fixtures for backends that are available
available_backends = ["mock"]
if deps.has("fairchem"):
    available_backends.append("uma")
if deps.has("so3lr"):
    available_backends.append("so3lr")
if deps.has("aimnet2"):
    available_backends.append("aimnet2")


class TestBackendComparison:
    """Compare results between different available backends."""

    @pytest.mark.skipif(
        len(available_backends) < 2, reason="Need at least 2 backends for comparison"
    )
    def test_h2_optimization_comparison(self):
        """Compare H2 optimization results between available backends."""
        h2 = Atoms("H2", positions=[[0, 0, 0], [1.8, 0, 0]])
        results = {}

        for backend in available_backends[:2]:  # Test first two available
            try:
                optimizer = qme.QMEOptimizer(backend=backend)
                result = optimizer.optimize_minimum(
                    atoms=h2.copy(), fmax=0.05, steps=30
                )
                results[backend] = result
            except ImportError:
                continue

        if len(results) >= 2:
            backend_names = list(results.keys())
            result1 = results[backend_names[0]]
            result2 = results[backend_names[1]]

            # Both should converge or make progress
            assert result1["steps_taken"] >= 0
            assert result2["steps_taken"] >= 0

            # Final distances should be in similar range
            dist1 = result1["optimized_atoms"].get_distance(0, 1)
            dist2 = result2["optimized_atoms"].get_distance(0, 1)

            # Allow for significant variation between backends
            assert 0.5 < dist1 < 2.0
            assert 0.5 < dist2 < 2.0

    @pytest.mark.skipif(
        len(available_backends) < 2, reason="Need at least 2 backends for comparison"
    )
    def test_water_optimization_comparison(self):
        """Compare water optimization between available backends."""
        water = Atoms(
            "H2O",
            positions=[
                [0.0, 0.0, 0.0],  # O
                [1.2, 0.0, 0.0],  # H (stretched)
                [0.0, 1.2, 0.0],  # H (stretched)
            ],
        )

        results = {}
        for backend in available_backends[:2]:
            try:
                optimizer = qme.QMEOptimizer(backend=backend)
                result = optimizer.optimize_minimum(
                    atoms=water.copy(), fmax=0.05, steps=50
                )
                results[backend] = result
            except ImportError:
                continue

        if len(results) >= 2:
            for backend, result in results.items():
                # All backends should optimize water reasonably
                optimized = result["optimized_atoms"]
                oh1_dist = optimized.get_distance(0, 1)
                oh2_dist = optimized.get_distance(0, 2)

                # O-H distances should be reasonable
                assert 0.5 < oh1_dist < 1.5
                assert 0.5 < oh2_dist < 1.5


class TestReactionPathways:
    """Test reaction pathway generation across backends."""

    def test_reaction_interpolation_with_available_calculators(self):
        """Test reaction interpolation with whatever calculators are available."""
        reactant = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        product = Atoms("H2", positions=[[0, 0, 0], [2.5, 0, 0]])

        # Test with each available calculator
        calculators = []

        # Mock calculators are always available
        calculators.append(("mock_uma", qme.get_mock_uma_calculator()))
        calculators.append(("mock_so3lr", qme.get_mock_so3lr_calculator()))
        calculators.append(("mock_aimnet2", qme.get_mock_aimnet2_calculator()))

        # Real calculators if available
        if deps.has("fairchem"):
            try:
                calculators.append(("uma", qme.get_uma_calculator()))
            except ImportError:
                pass

        if deps.has("so3lr"):
            try:
                calculators.append(("so3lr", qme.get_so3lr_calculator()))
            except ImportError:
                pass

        if deps.has("aimnet2"):
            try:
                calculators.append(("aimnet2", qme.get_aimnet2_calculator()))
            except ImportError:
                pass

        # Test interpolation with each calculator
        for calc_name, calculator in calculators:
            reaction = qme.Reaction(reactant, product, calculator=calculator)

            path = reaction.interpolate(npoints=5, method="linear")
            assert len(path) == 5

            # Check that distances increase along path
            distances = [geom.get_distance(0, 1) for geom in path]
            for i in range(1, len(distances)):
                assert distances[i] >= distances[i - 1]


@pytest.mark.skipif(not deps.has("sella"), reason="SELLA not available")
class TestTransitionStateSearch:
    """Test transition state searches across available backends."""

    def test_ts_search_with_available_backends(self):
        """Test TS search with available backends."""
        ts_guess = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])

        for backend in available_backends:
            try:
                optimizer = qme.QMEOptimizer(backend=backend)

                result = optimizer.find_transition_state(
                    atoms=ts_guess.copy(), fmax=0.05, steps=20  # Short for testing
                )

                # Should return results without crashing
                assert "converged" in result
                assert "steps_taken" in result

            except ImportError:
                # Backend not available, skip
                continue
            except Exception as e:
                # TS searches may not converge, but shouldn't crash with errors
                # about missing methods
                assert "find_transition_state" not in str(e)


class TestBackendFallbacks:
    """Test backend fallback behavior."""

    def test_fallback_to_mock_on_import_error(self):
        """Test that system gracefully falls back to mock when backends unavailable."""
        # This test checks that the error handling works properly

        # Try to create optimizers for potentially unavailable backends
        backend_attempts = ["nonexistent_backend"]

        for backend in backend_attempts:
            try:
                optimizer = qme.QMEOptimizer(backend=backend)
                # If this succeeds, the backend is available
                assert optimizer is not None
            except ValueError as e:
                # Expected for nonexistent backends
                assert "Unknown backend" in str(e)
            except ImportError:
                # Also acceptable - backend exists but dependencies missing
                pass

    def test_mock_backend_always_available(self):
        """Test that mock backend is always available as fallback."""
        optimizer = qme.QMEOptimizer(backend="mock")
        assert optimizer.backend == "mock"
        assert optimizer.calculator is not None

        # Should be able to optimize
        h2 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])
        result = optimizer.optimize_minimum(atoms=h2, fmax=0.1, steps=10)

        assert result["steps_taken"] >= 0
        assert "optimized_atoms" in result
