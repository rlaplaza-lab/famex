"""Unit tests for geometric_interface module.

This module tests the GeometricOptimizer and related classes to ensure they
behave consistently with ASE optimizers and handle edge cases properly.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from ase import Atoms
from ase.optimize.optimize import Optimizer

from qme.backend_availability import is_backend_available
from qme.core.geometric_interface import (
    GeometricOptimizer,
    GeometricTSOptimizer,
    create_geometric_optimizer,
    geometric_minima_optimizer,
    geometric_ts_optimizer,
)
from qme.potentials.mock_potential import MockCalculator


class TestGeometricOptimizerInit:
    """Test GeometricOptimizer initialization and parameter handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        self.atoms.calc = Mock()

    def test_basic_initialization(self):
        """Test basic GeometricOptimizer initialization."""
        optimizer = GeometricOptimizer(self.atoms)

        assert optimizer.order == 0
        assert optimizer.initial_hessian is None
        assert optimizer.step_count == 0
        assert optimizer.converged is False
        assert isinstance(optimizer.geometric_kwargs, dict)

    def test_initialization_with_order(self):
        """Test initialization with different order values."""
        # Test minima optimization (order=0)
        opt_min = GeometricOptimizer(self.atoms, order=0)
        assert opt_min.order == 0

        # Test TS optimization (order=1)
        opt_ts = GeometricOptimizer(self.atoms, order=1)
        assert opt_ts.order == 1

    def test_initialization_with_hessian(self):
        """Test initialization with Hessian matrix."""
        hessian = np.eye(9) * 0.1
        optimizer = GeometricOptimizer(self.atoms, hessian=hessian)

        assert optimizer.initial_hessian is not None
        np.testing.assert_array_equal(optimizer.initial_hessian, hessian)

    def test_parameter_filtering_ase_vs_geometric(self):
        """Test that ASE and geomeTRIC parameters are properly separated."""
        kwargs = {
            "trust": 0.05,  # geomeTRIC
            "convergence": {"energy": 1e-6, "gradient": 1e-3},  # geomeTRIC
            "maxiter": 500,  # geomeTRIC
            "logfile": "test.log",  # ASE
            "restart": "test.pckl",  # ASE
        }

        optimizer = GeometricOptimizer(self.atoms, **kwargs)

        # Check that geomeTRIC parameters are stored
        assert optimizer.geometric_kwargs["trust"] == 0.05
        assert optimizer.geometric_kwargs["convergence"]["energy"] == 1e-6
        assert optimizer.geometric_kwargs["maxiter"] == 500

        # Check that ASE-specific parameters are removed from geomeTRIC kwargs
        assert "logfile" not in optimizer.geometric_kwargs
        assert "restart" not in optimizer.geometric_kwargs

    def test_default_parameter_setting(self):
        """Test that default parameters are set correctly."""
        optimizer = GeometricOptimizer(self.atoms)

        # Check default convergence criteria
        expected_convergence = {"energy": 1e-6, "gradient": 1e-3, "step": 1e-3}
        assert optimizer.geometric_kwargs["convergence"] == expected_convergence
        assert optimizer.geometric_kwargs["maxiter"] == 1000
        assert optimizer.geometric_kwargs["trust"] == 0.1

    def test_ase_optimizer_inheritance(self):
        """Test that GeometricOptimizer properly inherits from ASE Optimizer."""
        optimizer = GeometricOptimizer(self.atoms)

        assert isinstance(optimizer, Optimizer)
        assert hasattr(optimizer, "atoms")
        assert hasattr(optimizer, "run")
        assert optimizer.atoms is self.atoms


class TestGeometricOptimizerRun:
    """Test the run method of GeometricOptimizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        self.atoms.calc = Mock()
        self.optimizer = GeometricOptimizer(self.atoms)

    def test_run_missing_calculator(self):
        """Test behavior when atoms has no calculator."""
        atoms_no_calc = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        optimizer_no_calc = GeometricOptimizer(atoms_no_calc)

        with pytest.raises(RuntimeError, match="Optimization failed"):
            optimizer_no_calc.run()

    def test_run_parameter_setup(self):
        """Test that run method sets up parameters correctly."""
        with patch.object(self.optimizer, "_run_optimization") as mock_run_opt:
            # Mock successful optimization
            def mock_run_optimization(optimizer, step_engine):
                self.optimizer.converged = True
                self.optimizer.step_count = 5
                final_positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
                self.optimizer.atoms.set_positions(final_positions)

            mock_run_opt.side_effect = mock_run_optimization

            result = self.optimizer.run(fmax=0.1, steps=50)

            # Check that the run method completed successfully
            assert result is True
            assert self.optimizer.converged is True
            assert self.optimizer.step_count == 5

            # Check that convergence criteria were updated
            assert self.optimizer.geometric_kwargs["convergence"]["gradient"] == 0.1
            assert self.optimizer.geometric_kwargs["maxiter"] == 50

    def test_run_with_hessian(self):
        """Test run method with initial Hessian."""
        hessian = np.eye(9) * 0.1
        optimizer = GeometricOptimizer(self.atoms, hessian=hessian)

        with patch.object(optimizer, "_run_optimization") as mock_run_opt:

            def mock_run_optimization(optimizer_obj, step_engine):
                optimizer.converged = True
                optimizer.step_count = 3
                final_positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
                optimizer.atoms.set_positions(final_positions)

            mock_run_opt.side_effect = mock_run_optimization

            result = optimizer.run()
            assert result is True
            assert optimizer.converged is True
            assert optimizer.step_count == 3

    def test_run_hessian_shape_validation(self):
        """Test Hessian shape validation during run."""
        wrong_hessian = np.eye(6) * 0.1  # Wrong shape: 6x6 instead of 9x9
        optimizer = GeometricOptimizer(self.atoms, hessian=wrong_hessian)

        with patch("geometric.ase_engine"), patch("geometric.optimize"):
            with patch("geometric.optimize.DelocalizedInternalCoordinates"):
                with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                    with patch.object(optimizer, "_create_molecule_from_atoms") as mock_create_mol:
                        mock_molecule = Mock()
                        mock_molecule.na = 3
                        mock_create_mol.return_value = mock_molecule

                        mock_tmpdir.return_value.__enter__.return_value = "/tmp/test"

                        with pytest.raises(
                            ValueError, match="Hessian must be \\(9, 9\\) but got \\(6, 6\\)"
                        ):
                            optimizer.run()

    def test_run_not_converged(self):
        """Test behavior when optimization does not converge."""
        with patch.object(self.optimizer, "_run_optimization") as mock_run_opt:

            def mock_run_optimization(optimizer, step_engine):
                self.optimizer.converged = False
                self.optimizer.step_count = 5
                final_positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
                self.optimizer.atoms.set_positions(final_positions)

            mock_run_opt.side_effect = mock_run_optimization

            result = self.optimizer.run()
            assert result is False
            assert self.optimizer.converged is False

    def test_run_exception_handling(self):
        """Test handling of optimization exceptions."""
        with patch.object(self.optimizer, "_run_optimization") as mock_run_opt:
            mock_run_opt.side_effect = ValueError("Optimization failed")

            with pytest.raises(ValueError, match="Optimization failed"):
                self.optimizer.run()


class TestGeometricTSOptimizer:
    """Test GeometricTSOptimizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        self.atoms.calc = Mock()

    def test_ts_optimizer_initialization(self):
        """Test GeometricTSOptimizer initialization."""
        ts_optimizer = GeometricTSOptimizer(self.atoms)

        assert ts_optimizer.order == 1
        assert isinstance(ts_optimizer, GeometricOptimizer)

    def test_ts_optimizer_with_parameters(self):
        """Test GeometricTSOptimizer with additional parameters."""
        hessian = np.eye(9) * 0.1
        ts_optimizer = GeometricTSOptimizer(self.atoms, hessian=hessian, trust=0.05)

        assert ts_optimizer.order == 1
        assert ts_optimizer.initial_hessian is not None
        assert ts_optimizer.geometric_kwargs["trust"] == 0.05


class TestFactoryFunctions:
    """Test factory functions for creating optimizers."""

    def setup_method(self):
        """Set up test fixtures."""
        self.atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        self.atoms.calc = Mock()

    def test_create_geometric_optimizer_minima(self):
        """Test create_geometric_optimizer for minima (order=0)."""
        optimizer = create_geometric_optimizer(self.atoms, order=0)

        assert isinstance(optimizer, GeometricOptimizer)
        assert optimizer.order == 0

    def test_create_geometric_optimizer_ts(self):
        """Test create_geometric_optimizer for TS (order=1)."""
        optimizer = create_geometric_optimizer(self.atoms, order=1)

        assert isinstance(optimizer, GeometricTSOptimizer)
        assert optimizer.order == 1

    def test_create_geometric_optimizer_with_hessian(self):
        """Test create_geometric_optimizer with Hessian."""
        hessian = np.eye(9) * 0.1
        optimizer = create_geometric_optimizer(self.atoms, order=0, hessian=hessian)

        assert optimizer.initial_hessian is not None
        np.testing.assert_array_equal(optimizer.initial_hessian, hessian)

    def test_geometric_minima_optimizer(self):
        """Test geometric_minima_optimizer factory function."""
        optimizer = geometric_minima_optimizer(self.atoms)

        assert isinstance(optimizer, GeometricOptimizer)
        assert optimizer.order == 0

    def test_geometric_ts_optimizer(self):
        """Test geometric_ts_optimizer factory function."""
        optimizer = geometric_ts_optimizer(self.atoms)

        assert isinstance(optimizer, GeometricTSOptimizer)
        assert optimizer.order == 1

    def test_factory_functions_with_parameters(self):
        """Test factory functions with additional parameters."""
        hessian = np.eye(9) * 0.1

        # Test minima optimizer with parameters
        min_opt = geometric_minima_optimizer(self.atoms, hessian=hessian, trust=0.05)
        assert min_opt.order == 0
        assert min_opt.geometric_kwargs["trust"] == 0.05

        # Test TS optimizer with parameters
        ts_opt = geometric_ts_optimizer(self.atoms, hessian=hessian, trust=0.05)
        assert ts_opt.order == 1
        assert ts_opt.geometric_kwargs["trust"] == 0.05


class TestASEOptimizerCompatibility:
    """Test that GeometricOptimizer behaves consistently with ASE optimizers."""

    def setup_method(self):
        """Set up test fixtures."""
        self.atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        self.atoms.calc = Mock()

    def test_ase_optimizer_interface(self):
        """Test that GeometricOptimizer implements ASE optimizer interface."""
        optimizer = GeometricOptimizer(self.atoms)

        # Check required ASE optimizer attributes
        assert hasattr(optimizer, "atoms")
        assert hasattr(optimizer, "run")
        assert optimizer.atoms is self.atoms

        # Check that run method has correct signature
        import inspect

        run_signature = inspect.signature(optimizer.run)
        assert "fmax" in run_signature.parameters
        assert "steps" in run_signature.parameters

    def test_run_method_return_type(self):
        """Test that run method returns boolean like ASE optimizers."""
        optimizer = GeometricOptimizer(self.atoms)

        with patch.object(optimizer, "_run_optimization") as mock_run_opt:

            def mock_run_optimization(optimizer_obj, step_engine):
                optimizer.converged = True
                optimizer.step_count = 5
                final_positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
                optimizer.atoms.set_positions(final_positions)

            mock_run_opt.side_effect = mock_run_optimization

            result = optimizer.run()
            assert isinstance(result, bool)

    def test_convergence_tracking(self):
        """Test that convergence is tracked like ASE optimizers."""
        optimizer = GeometricOptimizer(self.atoms)

        # Initially not converged
        assert optimizer.converged is False

        # Test that the converged attribute exists and can be set
        optimizer.converged = True
        assert optimizer.converged is True

        optimizer.converged = False
        assert optimizer.converged is False

    def test_step_count_tracking(self):
        """Test that step count is tracked like ASE optimizers."""
        optimizer = GeometricOptimizer(self.atoms)

        # Initially no steps
        assert optimizer.step_count == 0

        with patch.object(optimizer, "_run_optimization") as mock_run_opt:

            def mock_run_optimization(optimizer_obj, step_engine):
                optimizer.converged = True
                optimizer.step_count = 7
                final_positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
                optimizer.atoms.set_positions(final_positions)

            mock_run_opt.side_effect = mock_run_optimization

            optimizer.run()
            assert optimizer.step_count == 7

    def test_parameter_consistency(self):
        """Test that parameters are handled consistently with ASE optimizers."""
        # Test with ASE-style parameters
        optimizer = GeometricOptimizer(self.atoms, logfile="test.log", restart="test.pckl")

        # ASE parameters should be passed to parent class
        assert hasattr(optimizer, "logfile")
        assert hasattr(optimizer, "restart")

        # But should not be in geomeTRIC kwargs
        assert "logfile" not in optimizer.geometric_kwargs
        assert "restart" not in optimizer.geometric_kwargs

    def test_atoms_modification(self):
        """Test that atoms are modified in-place like ASE optimizers."""
        optimizer = GeometricOptimizer(self.atoms)

        # Test that we can modify atoms positions directly
        original_positions = self.atoms.get_positions().copy()
        new_positions = original_positions + 0.1
        self.atoms.set_positions(new_positions)

        # Verify the atoms were modified
        assert not np.array_equal(self.atoms.get_positions(), original_positions)
        np.testing.assert_array_almost_equal(self.atoms.get_positions(), new_positions)

        # Test that the optimizer has access to the same atoms object
        assert optimizer.atoms is self.atoms


class TestGeometricCoordinateHandling:
    """Test geometric optimizer coordinate handling and Hessian compatibility."""

    @pytest.fixture
    def test_molecule(self):
        """Create a test molecule for optimization tests."""
        symbols = ["O", "H", "H"]
        positions = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
        return Atoms(symbols=symbols, positions=positions)

    @pytest.fixture
    def test_benzene(self):
        """Create a benzene molecule for more complex tests."""
        symbols = ["C"] * 6 + ["H"] * 6
        positions = np.array(
            [
                # Carbon atoms (distorted ring)
                [1.39, 0.0, 0.0],
                [0.69, 1.20, 0.0],
                [-0.69, 1.20, 0.0],
                [-1.39, 0.0, 0.0],
                [-0.69, -1.20, 0.0],
                [0.69, -1.20, 0.0],
                # Hydrogen atoms
                [2.47, 0.0, 0.0],
                [1.23, 2.13, 0.0],
                [-1.23, 2.13, 0.0],
                [-2.47, 0.0, 0.0],
                [-1.23, -2.13, 0.0],
                [1.23, -2.13, 0.0],
            ]
        )
        return Atoms(symbols=symbols, positions=positions)

    def test_geometric_optimizer_preserves_original_atoms(self, test_molecule):
        """Test that geometric optimizer preserves original atoms object.

        Uses MockCalculator for fast unit testing of coordinate handling.
        """
        # Use mock calculator for fast unit testing (minima only)
        calculator = MockCalculator()

        # Create original atoms object
        original_atoms = test_molecule.copy()
        original_positions = original_atoms.get_positions().copy()
        original_id = id(original_atoms)

        # Attach calculator
        original_atoms.calc = calculator

        # Create geometric optimizer with original atoms
        optimizer = GeometricOptimizer(original_atoms, order=0)

        # Run optimization
        optimizer.run(fmax=0.05, steps=10)

        # Check that original atoms object was preserved
        assert id(original_atoms) == original_id, "Original atoms object ID should be preserved"

        # Check that original positions were modified (optimization happened)
        optimized_positions = original_atoms.get_positions()
        assert not np.allclose(
            original_positions, optimized_positions
        ), "Optimization should have changed the positions"

        # Verify the optimized structure is reasonable
        assert len(optimized_positions) == len(
            original_positions
        ), "Number of atoms should be preserved"
        assert optimized_positions.shape == (3, 3), "Position array shape should be preserved"

    def test_geometric_optimizer_coordinate_consistency(self, test_benzene):
        """Test that geometric optimizer produces consistent coordinates.

        Uses MockCalculator for fast unit testing of coordinate handling.
        """
        # Use mock calculator for fast unit testing (minima only)
        calculator = MockCalculator()

        # Create atoms object
        atoms = test_benzene.copy()
        atoms.calc = calculator

        # Store initial state
        initial_positions = atoms.get_positions().copy()
        initial_symbols = atoms.get_chemical_symbols().copy()

        # Run optimization
        optimizer = GeometricOptimizer(atoms, order=0)
        optimizer.run(fmax=0.1, steps=50)

        # Get optimized coordinates
        optimized_positions = atoms.get_positions()
        optimized_symbols = atoms.get_chemical_symbols()

        # Verify coordinate consistency
        assert len(optimized_positions) == len(
            initial_positions
        ), "Number of atoms should be preserved"
        assert optimized_symbols == initial_symbols, "Atomic symbols should be preserved"
        assert (
            optimized_positions.shape == initial_positions.shape
        ), "Position array shape should be preserved"

        # Verify coordinates are finite and reasonable
        assert np.all(
            np.isfinite(optimized_positions)
        ), "All optimized coordinates should be finite"
        assert np.all(
            np.abs(optimized_positions) < 100
        ), "Coordinates should be within reasonable range"

        # Verify some optimization occurred
        position_change = np.linalg.norm(optimized_positions - initial_positions)
        assert position_change > 0.001, "Optimization should have moved atoms significantly"

    @pytest.mark.skipif(
        not is_backend_available("uma"),
        reason="UMA backend required for realistic Hessian calculation tests",
    )
    def test_geometric_optimized_atoms_hessian_compatibility(self, test_benzene):
        """Test that geometric-optimized atoms work with Hessian calculations.

        Requires real ML backend (UMA) because MockCalculator doesn't produce
        realistic frequency spectra. Uses minimal structure for faster testing.
        """
        from qme.analysis.frequency import FrequencyAnalysis
        from qme.potentials.uma_potential import UMAPotential

        # Use smaller molecule for faster testing - water instead of benzene
        # Benzene (12 atoms) requires 72 force calls for Hessian
        # Water (3 atoms) requires only 18 force calls
        atoms = Atoms(
            ["O", "H", "H"],
            positions=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        )

        # Must use real ML backend for Hessian/frequency tests
        calculator = UMAPotential()
        atoms.calc = calculator

        # Run geometric optimization with fewer steps
        optimizer = GeometricOptimizer(atoms, order=0)
        optimizer.run(fmax=0.1, steps=20)  # Reduced from 50 to 20

        # Test Hessian calculation on optimized atoms
        try:
            freq_analysis = FrequencyAnalysis(atoms=atoms, calculator=calculator, delta=0.01)
            hessian = freq_analysis.calculate_hessian(method="finite_differences")
            frequencies, normal_modes = freq_analysis.diagonalize_hessian()

            # Verify Hessian is valid
            assert hessian is not None, "Hessian should be calculated successfully"
            assert hessian.shape == (
                9,
                9,
            ), f"Hessian should be 9x9 for 3 atoms, got {hessian.shape}"
            assert np.all(np.isfinite(hessian)), "Hessian should contain only finite values"

            # Verify frequencies are reasonable
            expected_frequencies = 3 * len(atoms)  # 3N degrees of freedom
            freq_msg = (
                f"Should have {expected_frequencies} frequencies for {len(atoms)} atoms, "
                f"got {len(frequencies)}"
            )
            assert len(frequencies) == expected_frequencies, freq_msg
            assert np.all(np.isfinite(frequencies)), "All frequencies should be finite"

            # Check for reasonable frequency range (should have some real positive frequencies)
            real_frequencies = frequencies[frequencies > 0]
            assert len(real_frequencies) > 0, "Should have some real positive frequencies"

        except Exception as e:
            pytest.fail(f"Hessian calculation failed on geometric-optimized atoms: {e}")
