"""
CI/CD integration tests for QME.

These tests specifically validate behavior in CI/CD environments where
some backends may not be available, ensuring graceful fallbacks and
proper test skipping.
"""

import pytest
from ase import Atoms

import qme
from qme.dependencies import deps


class TestCICDIntegration:
    """Test CI/CD specific scenarios."""

    def test_basic_functionality_always_available(self):
        """Test that basic QME functionality is always available."""
        # These should never fail in CI/CD
        assert hasattr(qme, "QMEOptimizer")
        assert hasattr(qme, "minimize_structure")
        assert hasattr(qme, "Geometry")
        assert hasattr(qme, "Reaction")

    def test_mock_backend_always_works(self):
        """Test that mock backend always works for CI/CD."""
        optimizer = qme.QMEOptimizer(backend="mock")
        assert optimizer.backend == "mock"
        assert optimizer.calculator is not None

        # Should be able to optimize
        h2 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])
        result = optimizer.optimize_minimum(atoms=h2, fmax=0.1, steps=5)

        assert "optimized_atoms" in result
        assert result["steps_taken"] >= 0

    def test_minimize_structure_function_fallback(self):
        """Test minimize_structure works with available backends."""
        h2 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])

        # Should work with mock backend (always available)
        result = qme.minimize_structure(h2, backend="mock", steps=5)
        assert isinstance(result, Atoms)

    def test_backend_availability_detection(self):
        """Test that backend availability is correctly detected."""
        # Mock backend should always be available
        assert "mock" in qme.QMEOptimizer.AVAILABLE_BACKENDS

        # Other backends may or may not be available
        for backend in ["uma", "so3lr", "aimnet2", "mace"]:
            try:
                optimizer = qme.QMEOptimizer(backend=backend)
                # If creation succeeds, backend is available
                assert optimizer.backend == backend
            except ImportError:
                # Backend not available - this is expected in some CI environments
                pass
            except ValueError as e:
                # Backend not recognized - should not happen with valid backends
                if "Unknown backend" in str(e):
                    pytest.fail(
                        f"Backend {backend} should be recognized even if not available"
                    )

    def test_graceful_degradation_to_mock(self):
        """Test that applications can gracefully degrade to mock calculators."""
        # This simulates how user code should handle missing backends
        backends_to_try = ["uma", "so3lr", "aimnet2", "mace", "mock"]

        working_backend = None
        working_optimizer = None
        for backend in backends_to_try:
            try:
                optimizer = qme.QMEOptimizer(backend=backend)
                working_backend = backend
                working_optimizer = optimizer
                break
            except ImportError:
                continue

        # At minimum, mock should work
        assert working_backend is not None
        assert working_backend in backends_to_try
        assert working_optimizer is not None

    def test_dependency_manager_consistency(self):
        """Test that dependency manager provides consistent information."""
        # Test known dependencies
        torch_available = deps.has("torch")
        sella_available = deps.has("sella")

        # These should be boolean
        assert isinstance(torch_available, bool)
        assert isinstance(sella_available, bool)

        # Test that require() works correctly
        try:
            sella_module = deps.require("sella", "testing")
            assert sella_module is not None
            assert sella_available  # Should be True if require() succeeded
        except ImportError:
            assert not sella_available  # Should be False if require() failed

    def test_calculator_creation_robustness(self):
        """Test that calculator creation is robust in CI/CD."""
        # Mock calculators should always work
        mock_calculators = [
            qme.MockCalculator(backend="uma"),
            qme.MockCalculator(backend="so3lr"),
            qme.MockCalculator(backend="aimnet2"),
            qme.MockCalculator(backend="mace"),
        ]

        for calc in mock_calculators:
            assert calc is not None

            # Should be able to calculate energy and forces
            h2 = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
            h2.calc = calc

            energy = h2.get_potential_energy()
            forces = h2.get_forces()

            assert isinstance(energy, (int, float))
            assert forces.shape == (2, 3)

    def test_reaction_pathways_work_with_mock(self):
        """Test that reaction pathways work with mock calculators."""
        from ase import Atoms

        reactant = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        product = Atoms("H2", positions=[[0, 0, 0], [2.5, 0, 0]])

        mock_calc = qme.MockCalculator(backend="uma")
        reaction = qme.Reaction(reactant, product, calculator=mock_calc)

        # Should be able to generate pathway
        path = reaction.interpolate(npoints=5, method="linear")
        assert len(path) == 5

        # All points should have same number of atoms
        for geom in path:
            assert len(geom) == 2

    @pytest.mark.skipif(not deps.has("sella"), reason="SELLA not available")
    def test_transition_state_search_conditional(self):
        """Test transition state search when SELLA is available."""
        optimizer = qme.QMEOptimizer(backend="mock")
        ts_guess = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])

        # Should not crash even if it doesn't converge
        result = optimizer.find_transition_state(atoms=ts_guess, fmax=0.05, steps=10)

        assert "converged" in result
        assert "steps_taken" in result

    def test_cli_integration_availability(self):
        """Test that CLI integration works in CI/CD."""
        # CLI should be importable
        from qme import cli

        assert hasattr(cli, "main")

        # Test setup command should work with mock backend
        # (This is tested more thoroughly in CLI integration tests)

    def test_comprehensive_backend_matrix(self):
        """Test comprehensive matrix of backend availability."""
        backend_tests = {
            "mock": True,  # Always available
            "uma": deps.has("fairchem"),
            "so3lr": deps.has("so3lr"),
            "aimnet2": deps.has("aimnet2"),
        }

        for backend, should_be_available in backend_tests.items():
            if should_be_available:
                # Should be able to create optimizer
                optimizer = qme.QMEOptimizer(backend=backend)
                assert optimizer.backend == backend

                # Should be able to do basic optimization
                h2 = Atoms("H2", positions=[[0, 0, 0], [1.8, 0, 0]])
                result = optimizer.optimize_minimum(atoms=h2, fmax=0.1, steps=5)
                assert "optimized_atoms" in result
            else:
                # Should gracefully handle unavailable backends
                try:
                    optimizer = qme.QMEOptimizer(backend=backend)
                    # If this succeeds, the backend became available
                    pass
                except ImportError:
                    # Expected for unavailable backends
                    pass


class TestDocumentationExamples:
    """Test examples that should work in documentation."""

    def test_basic_usage_example(self):
        """Test basic usage example from documentation."""
        # This should always work
        h2 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])

        result = qme.minimize_structure(
            h2, backend="mock", fmax=0.05, steps=50  # Use mock for CI/CD
        )

        assert isinstance(result, Atoms)
        final_distance = result.get_distance(0, 1)
        assert 0.5 < final_distance < 2.0

    def test_optimizer_class_example(self):
        """Test optimizer class usage example."""
        optimizer = qme.QMEOptimizer(backend="mock")

        h2 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])
        result = optimizer.optimize_minimum(atoms=h2, fmax=0.05, steps=50)

        assert result["converged"] or result["steps_taken"] > 0
        assert "optimized_atoms" in result

    def test_reaction_pathway_example(self):
        """Test reaction pathway example."""
        reactant = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        product = Atoms("H2", positions=[[0, 0, 0], [3.0, 0, 0]])

        reaction = qme.Reaction(reactant, product)
        path = reaction.interpolate(npoints=10, method="linear")

        assert len(path) == 10
        # Distances should increase along path
        distances = [geom.get_distance(0, 1) for geom in path]
        assert distances[0] < distances[-1]
