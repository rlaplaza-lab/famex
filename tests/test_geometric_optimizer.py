"""Tests for geomeTRIC optimizer integration."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from qme.core.explorer import Explorer
from qme.dependencies import deps
from tests.backend_utils import AVAILABLE_ML_BACKENDS


class TestGeometricOptimizer:
    """Test geomeTRIC optimizer functionality."""

    @pytest.fixture
    def water_molecule(self):
        """Create a water molecule for testing."""
        return molecule("H2O")

    @pytest.fixture
    def methane_molecule(self):
        """Create a methane molecule for testing."""
        return molecule("CH4")

    def test_geometric_availability(self):
        """Test that geomeTRIC is available."""
        assert deps.has("geometric"), "geomeTRIC should be available"

    def test_geometric_minima_optimization(self, water_molecule):
        """Test geomeTRIC minima optimization."""
        if not deps.has("geometric"):
            pytest.skip("geomeTRIC not available")

        # Test with mock backend to avoid ML model dependencies
        optimizer = Explorer(
            atoms=water_molecule, backend="mock", local_optimizer="geometric"
        )

        result = optimizer.optimize_minima(
            fmax=0.1, steps=10  # Small number for testing
        )

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1  # Should return list with one atoms object
        final_atoms = result[0]  # Get first (and only) atoms object
        assert hasattr(final_atoms, "get_distance")  # Should be Atoms object
        assert len(final_atoms) == 3  # H2O has 3 atoms

    def test_geometric_ts_optimization(self, water_molecule):
        """Test geomeTRIC transition state optimization."""
        if not deps.has("geometric"):
            pytest.skip("geomeTRIC not available")

        # Skip if no ML backends available for TS optimization
        if not AVAILABLE_ML_BACKENDS:
            pytest.skip("No ML backends available for TS optimization")

        # Create a TS guess (stretched H2O)
        ts_guess = water_molecule.copy()
        # Stretch one O-H bond to create TS-like geometry
        pos = ts_guess.get_positions()
        pos[1] += [0.5, 0.0, 0.0]  # Move H away from O
        ts_guess.set_positions(pos)

        optimizer = Explorer(
            atoms=ts_guess,
            backend=AVAILABLE_ML_BACKENDS[0],  # Use first available ML backend
            local_optimizer="geometric",
        )

        result = optimizer.optimize_ts(fmax=0.1, steps=10)  # Small number for testing

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1  # Should return list with one atoms object
        final_atoms = result[0]  # Get first (and only) atoms object
        assert hasattr(final_atoms, "get_distance")  # Should be Atoms object
        assert len(final_atoms) == 3  # H2O has 3 atoms

    def test_geometric_vs_sella_comparison(self, water_molecule):
        """Compare geomeTRIC and Sella optimizers on the same system."""
        if not deps.has("geometric") or not deps.has("sella"):
            pytest.skip("Both geomeTRIC and Sella must be available")

        # Test minima optimization
        geometric_opt = Explorer(
            atoms=water_molecule.copy(), backend="mock", local_optimizer="geometric"
        )

        sella_opt = Explorer(
            atoms=water_molecule.copy(), backend="mock", local_optimizer="sella"
        )

        # Run both optimizations
        geometric_result = geometric_opt.optimize_minima(fmax=0.1, steps=10)
        sella_result = sella_opt.optimize_minima(fmax=0.1, steps=10)

        # Both should produce valid results
        assert geometric_result is not None
        assert sella_result is not None
        assert isinstance(geometric_result, list)
        assert isinstance(sella_result, list)
        assert len(geometric_result) == 1
        assert len(sella_result) == 1
        geometric_atoms = geometric_result[0]
        sella_atoms = sella_result[0]
        assert hasattr(geometric_atoms, "get_distance")  # Should be Atoms object
        assert hasattr(sella_atoms, "get_distance")  # Should be Atoms object

        # Both should have the same number of atoms
        assert len(geometric_atoms) == len(sella_atoms)

    def test_geometric_with_different_backends(self, water_molecule):
        """Test geomeTRIC with different ML backends if available."""
        if not deps.has("geometric"):
            pytest.skip("geomeTRIC not available")

        # Test with available backends
        available_backends = []
        for backend in ["uma", "aimnet2", "mace", "so3lr"]:
            if deps.has(backend):
                available_backends.append(backend)

        if not available_backends:
            pytest.skip("No ML backends available for testing")

        # Test only first available to keep tests fast
        for backend in available_backends[:1]:
            optimizer = Explorer(
                atoms=water_molecule.copy(),
                backend=backend,
                local_optimizer="geometric",
            )

            result = optimizer.optimize_minima(
                fmax=0.1, steps=5  # Very small number for testing
            )

            assert result is not None
            assert isinstance(result, list)
            assert len(result) == 1
            final_atoms = result[0]
            assert hasattr(final_atoms, "get_distance")  # Should be Atoms object

    def test_geometric_optimizer_parameters(self, water_molecule):
        """Test geomeTRIC with custom optimizer parameters."""
        if not deps.has("geometric"):
            pytest.skip("geomeTRIC not available")

        # Test with custom parameters
        optimizer = Explorer(
            atoms=water_molecule,
            backend="mock",
            local_optimizer="geometric",
            optimizer_kwargs={
                "trust": 0.05,  # Smaller trust radius
                "convergence": {"energy": 1e-5, "gradient": 0.05, "step": 1e-4},
            },
        )

        result = optimizer.optimize_minima(fmax=0.05, steps=10)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        final_atoms = result[0]
        assert hasattr(final_atoms, "get_distance")  # Should be Atoms object

    def test_geometric_ts_parameters(self, water_molecule):
        """Test geomeTRIC TS optimization with custom parameters."""
        if not deps.has("geometric"):
            pytest.skip("geomeTRIC not available")

        # Skip if no ML backends available for TS optimization
        if not AVAILABLE_ML_BACKENDS:
            pytest.skip("No ML backends available for TS optimization")

        # Create TS guess
        ts_guess = water_molecule.copy()
        pos = ts_guess.get_positions()
        pos[1] += [0.3, 0.0, 0.0]  # Stretch O-H bond
        ts_guess.set_positions(pos)

        optimizer = Explorer(
            atoms=ts_guess,
            backend=AVAILABLE_ML_BACKENDS[0],  # Use first available ML backend
            local_optimizer="geometric",
            ts_kwargs={
                "trust": 0.05,
                "convergence": {"energy": 1e-5, "gradient": 0.05, "step": 1e-4},
            },
        )

        result = optimizer.optimize_ts(fmax=0.05, steps=10)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        final_atoms = result[0]
        assert hasattr(final_atoms, "get_distance")  # Should be Atoms object

    def test_geometric_with_hessian_input(self, water_molecule):
        """Test geomeTRIC optimization with initial Hessian matrix."""
        if not deps.has("geometric"):
            pytest.skip("geomeTRIC not available")

        # Create a simple Hessian matrix (3 atoms = 9x9 matrix)
        hessian = np.eye(9) * 0.1  # Simple diagonal Hessian

        optimizer = Explorer(
            atoms=water_molecule,
            backend="mock",
            local_optimizer="geometric",
            initial_hessian=hessian,
        )

        result = optimizer.optimize_minima(fmax=0.1, steps=10)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        final_atoms = result[0]
        assert hasattr(final_atoms, "get_distance")  # Should be Atoms object
        assert len(final_atoms) == 3  # H2O has 3 atoms

    def test_geometric_ts_with_hessian_input(self, water_molecule):
        """Test geomeTRIC TS optimization with initial Hessian matrix."""
        if not deps.has("geometric"):
            pytest.skip("geomeTRIC not available")

        # Skip if no ML backends available for TS optimization
        if not AVAILABLE_ML_BACKENDS:
            pytest.skip("No ML backends available for TS optimization")

        # Create TS guess (stretched H2O)
        ts_guess = water_molecule.copy()
        pos = ts_guess.get_positions()
        pos[1] += [0.3, 0.0, 0.0]  # Stretch O-H bond
        ts_guess.set_positions(pos)

        # Create a simple Hessian matrix (3 atoms = 9x9 matrix)
        hessian = np.eye(9) * 0.1  # Simple diagonal Hessian

        optimizer = Explorer(
            atoms=ts_guess,
            backend=AVAILABLE_ML_BACKENDS[0],  # Use first available ML backend
            local_optimizer="geometric",
            initial_hessian=hessian,
        )

        result = optimizer.optimize_ts(fmax=0.1, steps=10)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        final_atoms = result[0]
        assert hasattr(final_atoms, "get_distance")  # Should be Atoms object
        assert len(final_atoms) == 3  # H2O has 3 atoms
