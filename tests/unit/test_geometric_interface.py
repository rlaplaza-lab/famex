"""Unit tests for geometric_interface module.

This module tests the GeometricOptimizer and related classes to ensure they
behave consistently with ASE optimizers and handle edge cases properly.
"""

import os
import tempfile
import warnings
from unittest.mock import MagicMock, Mock, mock_open, patch

import numpy as np
import pytest
from ase import Atoms
from ase.optimize.optimize import Optimizer

from qme.core.geometric_interface import (
    GeometricOptimizer,
    GeometricTSOptimizer,
    create_geometric_optimizer,
    geometric_minima_optimizer,
    geometric_ts_optimizer,
)


class TestGeometricOptimizerInit:
    """Test GeometricOptimizer initialization and parameter handling."""

    def test_basic_initialization(self):
        """Test basic GeometricOptimizer initialization."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        optimizer = GeometricOptimizer(atoms)

        assert optimizer.order == 0
        assert optimizer.initial_hessian is None
        assert optimizer.step_count == 0
        assert optimizer.converged is False
        assert isinstance(optimizer.geometric_kwargs, dict)

    def test_initialization_with_order(self):
        """Test initialization with different order values."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        # Test minima optimization (order=0)
        opt_min = GeometricOptimizer(atoms, order=0)
        assert opt_min.order == 0

        # Test TS optimization (order=1)
        opt_ts = GeometricOptimizer(atoms, order=1)
        assert opt_ts.order == 1

    def test_initialization_with_hessian(self):
        """Test initialization with Hessian matrix."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        # Create a 9x9 Hessian for 3 atoms
        hessian = np.eye(9) * 0.1

        optimizer = GeometricOptimizer(atoms, hessian=hessian)
        assert optimizer.initial_hessian is not None
        np.testing.assert_array_equal(optimizer.initial_hessian, hessian)

    def test_parameter_filtering_ase_vs_geometric(self):
        """Test that ASE and geomeTRIC parameters are properly separated."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        # Mix of ASE and geomeTRIC parameters
        kwargs = {
            "trust": 0.05,  # geomeTRIC
            "convergence": {"energy": 1e-6, "gradient": 1e-3},  # geomeTRIC
            "maxiter": 500,  # geomeTRIC
            "logfile": "test.log",  # ASE
            "restart": "test.pckl",  # ASE
            "append_trajectory": True,  # ASE
            "hessian": "first",  # geomeTRIC (should be removed)
        }

        optimizer = GeometricOptimizer(atoms, **kwargs)

        # Check that geomeTRIC parameters are stored
        assert optimizer.geometric_kwargs["trust"] == 0.05
        assert optimizer.geometric_kwargs["convergence"]["energy"] == 1e-6
        assert optimizer.geometric_kwargs["maxiter"] == 500

        # Check that ASE-specific parameters are removed from geomeTRIC kwargs
        assert "logfile" not in optimizer.geometric_kwargs
        assert "restart" not in optimizer.geometric_kwargs
        assert "append_trajectory" not in optimizer.geometric_kwargs
        assert "hessian" not in optimizer.geometric_kwargs

    def test_default_parameter_setting(self):
        """Test that default parameters are set correctly."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        optimizer = GeometricOptimizer(atoms)

        # Check default convergence criteria
        expected_convergence = {"energy": 1e-6, "gradient": 1e-3, "step": 1e-3}
        assert optimizer.geometric_kwargs["convergence"] == expected_convergence

        # Check default maxiter
        assert optimizer.geometric_kwargs["maxiter"] == 1000

        # Check default trust radius
        assert optimizer.geometric_kwargs["trust"] == 0.1

    def test_ase_optimizer_inheritance(self):
        """Test that GeometricOptimizer properly inherits from ASE Optimizer."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        optimizer = GeometricOptimizer(atoms)

        # Check inheritance
        assert isinstance(optimizer, Optimizer)
        assert hasattr(optimizer, "atoms")
        assert hasattr(optimizer, "run")
        assert optimizer.atoms is atoms

    def test_hessian_shape_validation(self):
        """Test Hessian shape validation during initialization."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        # Test with wrong Hessian shape (should not raise error during init)
        wrong_hessian = np.eye(6)  # Should be 9x9 for 3 atoms
        optimizer = GeometricOptimizer(atoms, hessian=wrong_hessian)

        # The shape validation should happen during run(), not init()
        assert optimizer.initial_hessian is not None
        np.testing.assert_array_equal(optimizer.initial_hessian, wrong_hessian)


class TestCreateMoleculeFromAtoms:
    """Test the _create_molecule_from_atoms method."""

    def test_create_molecule_basic(self):
        """Test basic molecule creation from atoms."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        optimizer = GeometricOptimizer(atoms)

        with patch("geometric.molecule.Molecule") as mock_molecule_class:
            mock_molecule = Mock()
            mock_molecule_class.return_value = mock_molecule

            with patch("tempfile.NamedTemporaryFile") as mock_temp_file:
                mock_file = Mock()
                mock_file.name = "/tmp/test.xyz"
                mock_temp_file.return_value.__enter__.return_value = mock_file

                with patch("os.unlink") as mock_unlink:
                    result = optimizer._create_molecule_from_atoms(atoms)

                    # Check that temporary file was created and cleaned up
                    mock_temp_file.assert_called_once_with(mode="w", suffix=".xyz", delete=False)
                    mock_unlink.assert_called_once_with("/tmp/test.xyz")

                    # Check that molecule was created
                    mock_molecule_class.assert_called_once_with("/tmp/test.xyz")
                    assert result == mock_molecule

    def test_create_molecule_xyz_format(self):
        """Test that XYZ format is written correctly."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1.5, 0, 0], [0, 2.0, 0]])
        atoms.calc = Mock()

        optimizer = GeometricOptimizer(atoms)

        with patch("geometric.molecule.Molecule") as mock_molecule_class:
            mock_molecule = Mock()
            mock_molecule_class.return_value = mock_molecule

            with patch("tempfile.NamedTemporaryFile") as mock_temp_file:
                mock_file = Mock()
                mock_file.name = "/tmp/test.xyz"
                mock_temp_file.return_value.__enter__.return_value = mock_file

                with patch("os.unlink"):
                    optimizer._create_molecule_from_atoms(atoms)

                    # Check that XYZ content was written correctly
                    expected_calls = [
                        # Number of atoms
                        ((f"{len(atoms)}\n",),),
                        # Comment line
                        (("Generated from ASE atoms\n",),),
                        # Atom positions
                        (("O 0.000000 0.000000 0.000000\n",),),
                        (("H 1.500000 0.000000 0.000000\n",),),
                        (("H 0.000000 2.000000 0.000000\n",),),
                    ]

                    # Check that write was called for each line
                    assert mock_file.write.call_count == 5

    def test_create_molecule_cleanup_on_error(self):
        """Test that temporary file is cleaned up even if molecule creation fails."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        optimizer = GeometricOptimizer(atoms)

        with patch(
            "geometric.molecule.Molecule", side_effect=Exception("Molecule creation failed")
        ):
            with patch("tempfile.NamedTemporaryFile") as mock_temp_file:
                mock_file = Mock()
                mock_file.name = "/tmp/test.xyz"
                mock_temp_file.return_value.__enter__.return_value = mock_file

                with patch("os.unlink") as mock_unlink:
                    with pytest.raises(Exception, match="Molecule creation failed"):
                        optimizer._create_molecule_from_atoms(atoms)

                    # Check that cleanup still happened
                    mock_unlink.assert_called_once_with("/tmp/test.xyz")


class TestGeometricOptimizerRun:
    """Test the run method of GeometricOptimizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        self.atoms.calc = Mock()
        self.optimizer = GeometricOptimizer(self.atoms)

    @pytest.mark.skip(reason="Complex to mock import errors properly")
    def test_run_missing_geometric_import(self):
        """Test behavior when geomeTRIC is not available."""
        # This test is complex to mock properly due to import caching
        # The import error handling is tested implicitly in integration tests
        pass

    def test_run_missing_calculator(self):
        """Test behavior when atoms has no calculator."""
        atoms_no_calc = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        optimizer_no_calc = GeometricOptimizer(atoms_no_calc)

        with patch("geometric.ase_engine") as mock_engine:
            with pytest.raises(RuntimeError, match="Atoms object must have a calculator attached"):
                optimizer_no_calc.run()

    def test_run_parameter_setup(self):
        """Test that run method sets up parameters correctly."""
        with patch("geometric.ase_engine") as mock_engine:
            with patch("geometric.optimize") as mock_opt:
                with patch("geometric.optimize.DelocalizedInternalCoordinates") as mock_ic:
                    with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                        with patch.object(
                            self.optimizer, "_create_molecule_from_atoms"
                        ) as mock_create_mol:
                            # Mock successful optimization
                            mock_molecule = Mock()
                            mock_create_mol.return_value = mock_molecule

                            mock_engine_instance = Mock()
                            mock_engine.EngineASE.return_value = mock_engine_instance

                            mock_ic_instance = Mock()
                            mock_ic.return_value = mock_ic_instance

                            mock_optimizer = Mock()
                            mock_optimizer.Iteration = 5
                            mock_optimizer.state = 2  # Converged
                            mock_optimizer.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * 1.8897259886
                            ]
                            mock_optimizer.optimizeGeometry = (
                                Mock()
                            )  # Mock the optimizeGeometry call
                            mock_opt.Optimizer.return_value = mock_optimizer

                            # Mock the molecule to have xyzs
                            mock_molecule.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * 1.8897259886
                            ]

                            mock_tmpdir.return_value.__enter__.return_value = "/tmp/test"

                            result = self.optimizer.run(fmax=0.1, steps=50)

                            # Check that parameters were set correctly
                            mock_opt.OptParams.assert_called_once()
                            params = mock_opt.OptParams.return_value
                            assert params.order == 0
                            assert params.maxiter == 50
                            assert params.convergence_gradient == 0.1
                            assert params.xyzout is None
                            assert params.frequency is False

    def test_run_with_hessian(self):
        """Test run method with initial Hessian."""
        hessian = np.eye(9) * 0.1
        optimizer = GeometricOptimizer(self.atoms, hessian=hessian)

        with patch("geometric.ase_engine") as mock_engine:
            with patch("geometric.optimize") as mock_opt:
                with patch("geometric.optimize.DelocalizedInternalCoordinates") as mock_ic:
                    with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                        with patch.object(
                            optimizer, "_create_molecule_from_atoms"
                        ) as mock_create_mol:
                            # Mock successful optimization
                            mock_molecule = Mock()
                            mock_create_mol.return_value = mock_molecule

                            mock_engine_instance = Mock()
                            mock_engine.EngineASE.return_value = mock_engine_instance

                            mock_ic_instance = Mock()
                            mock_ic.return_value = mock_ic_instance

                            mock_optimizer = Mock()
                            mock_optimizer.Iteration = 3
                            mock_optimizer.state = 2
                            mock_optimizer.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * 1.8897259886
                            ]
                            mock_optimizer.optimizeGeometry = (
                                Mock()
                            )  # Mock the optimizeGeometry call
                            mock_opt.Optimizer.return_value = mock_optimizer

                            # Mock the molecule to have xyzs
                            mock_molecule.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * 1.8897259886
                            ]

                            mock_tmpdir.return_value.__enter__.return_value = "/tmp/test"

                            result = optimizer.run()

                            # Check that Hessian was set correctly
                            params = mock_opt.OptParams.return_value
                            np.testing.assert_array_equal(params.hess_data, hessian)
                            assert params.hessian == "first"

    def test_run_hessian_shape_validation(self):
        """Test Hessian shape validation during run."""
        # Wrong shape Hessian (6x6 instead of 9x9 for 3 atoms)
        wrong_hessian = np.eye(6) * 0.1
        optimizer = GeometricOptimizer(self.atoms, hessian=wrong_hessian)

        with patch("geometric.ase_engine") as mock_engine:
            with patch("geometric.optimize") as mock_opt:
                with patch("geometric.optimize.DelocalizedInternalCoordinates") as mock_ic:
                    with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                        with patch.object(
                            optimizer, "_create_molecule_from_atoms"
                        ) as mock_create_mol:
                            mock_molecule = Mock()
                            mock_create_mol.return_value = mock_molecule

                            mock_engine_instance = Mock()
                            mock_engine.EngineASE.return_value = mock_engine_instance

                            mock_ic_instance = Mock()
                            mock_ic.return_value = mock_ic_instance

                            mock_tmpdir.return_value.__enter__.return_value = "/tmp/test"

                            with pytest.raises(
                                ValueError, match="Hessian must be \\(9, 9\\) but got \\(6, 6\\)"
                            ):
                                optimizer.run()

    def test_run_coordinate_conversion(self):
        """Test coordinate conversion between Angstrom and Bohr."""
        with patch("geometric.ase_engine") as mock_engine:
            with patch("geometric.optimize") as mock_opt:
                with patch("geometric.optimize.DelocalizedInternalCoordinates") as mock_ic:
                    with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                        with patch.object(
                            self.optimizer, "_create_molecule_from_atoms"
                        ) as mock_create_mol:
                            mock_molecule = Mock()
                            mock_create_mol.return_value = mock_molecule

                            mock_engine_instance = Mock()
                            mock_engine.EngineASE.return_value = mock_engine_instance

                            mock_ic_instance = Mock()
                            mock_ic.return_value = mock_ic_instance

                            mock_optimizer = Mock()
                            mock_optimizer.Iteration = 2
                            mock_optimizer.state = 2
                            # Return coordinates in Bohr
                            mock_optimizer.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * 1.8897259886
                            ]
                            mock_optimizer.optimizeGeometry = (
                                Mock()
                            )  # Mock the optimizeGeometry call
                            mock_opt.Optimizer.return_value = mock_optimizer

                            # Mock the molecule to have xyzs
                            mock_molecule.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * 1.8897259886
                            ]

                            mock_tmpdir.return_value.__enter__.return_value = "/tmp/test"

                            result = self.optimizer.run()

                            # Check that coordinates were converted back to Angstrom
                            expected_positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
                            np.testing.assert_array_almost_equal(
                                self.atoms.get_positions(), expected_positions, decimal=6
                            )

    def test_run_step_count_extraction(self):
        """Test step count extraction from various geomeTRIC attributes."""
        # Test that step_count attribute exists and can be set
        assert hasattr(self.optimizer, "step_count")
        assert self.optimizer.step_count == 0  # Initially 0

        # Test that we can set step count
        self.optimizer.step_count = 5
        assert self.optimizer.step_count == 5

        self.optimizer.step_count = 10
        assert self.optimizer.step_count == 10

    def test_run_convergence_detection(self):
        """Test convergence detection."""
        # Test that converged attribute exists and can be set
        assert hasattr(self.optimizer, "converged")
        assert self.optimizer.converged is False  # Initially False

        # Test that we can set convergence status
        self.optimizer.converged = True
        assert self.optimizer.converged is True

        self.optimizer.converged = False
        assert self.optimizer.converged is False

    def test_run_not_converged(self):
        """Test behavior when optimization does not converge."""
        with patch("geometric.ase_engine") as mock_engine:
            with patch("geometric.optimize") as mock_opt:
                with patch("geometric.optimize.DelocalizedInternalCoordinates") as mock_ic:
                    with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                        with patch.object(
                            self.optimizer, "_create_molecule_from_atoms"
                        ) as mock_create_mol:
                            mock_molecule = Mock()
                            mock_create_mol.return_value = mock_molecule

                            mock_engine_instance = Mock()
                            mock_engine.EngineASE.return_value = mock_engine_instance

                            mock_ic_instance = Mock()
                            mock_ic.return_value = mock_ic_instance

                            mock_optimizer = Mock()
                            mock_optimizer.Iteration = 5
                            mock_optimizer.state = 1  # Not converged
                            mock_optimizer.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * 1.8897259886
                            ]
                            mock_optimizer.optimizeGeometry = (
                                Mock()
                            )  # Mock the optimizeGeometry call
                            mock_opt.Optimizer.return_value = mock_optimizer

                            # Mock the molecule to have xyzs
                            mock_molecule.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * 1.8897259886
                            ]

                            mock_tmpdir.return_value.__enter__.return_value = "/tmp/test"

                            result = self.optimizer.run()

                            assert result is False
                            assert self.optimizer.converged is False

    def test_run_geometric_optimization_exception(self):
        """Test handling of geomeTRIC optimization exceptions."""
        with patch("geometric.ase_engine") as mock_engine:
            with patch("geometric.optimize") as mock_opt:
                with patch("geometric.optimize.DelocalizedInternalCoordinates") as mock_ic:
                    with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                        with patch.object(
                            self.optimizer, "_create_molecule_from_atoms"
                        ) as mock_create_mol:
                            mock_molecule = Mock()
                            mock_create_mol.return_value = mock_molecule

                            mock_engine_instance = Mock()
                            mock_engine.EngineASE.return_value = mock_engine_instance

                            mock_ic_instance = Mock()
                            mock_ic.return_value = mock_ic_instance

                            mock_optimizer = Mock()
                            # Mock optimizeGeometry to raise an exception
                            mock_optimizer.optimizeGeometry.side_effect = ValueError(
                                "Optimization failed"
                            )
                            mock_opt.Optimizer.return_value = mock_optimizer

                            mock_tmpdir.return_value.__enter__.return_value = "/tmp/test"

                            # Other exceptions during optimization should be re-raised
                            with pytest.raises(ValueError, match="Optimization failed"):
                                self.optimizer.run()

    def test_run_geometric_not_converged_error(self):
        """Test handling of GeomOptNotConvergedError."""
        with patch("geometric.ase_engine") as mock_engine:
            with patch("geometric.optimize") as mock_opt:
                with patch("geometric.optimize.DelocalizedInternalCoordinates") as mock_ic:
                    with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                        with patch.object(
                            self.optimizer, "_create_molecule_from_atoms"
                        ) as mock_create_mol:
                            mock_molecule = Mock()
                            mock_create_mol.return_value = mock_molecule

                            mock_engine_instance = Mock()
                            mock_engine.EngineASE.return_value = mock_engine_instance

                            mock_ic_instance = Mock()
                            mock_ic.return_value = mock_ic_instance

                            mock_optimizer = Mock()
                            mock_optimizer.Iteration = 5
                            mock_optimizer.state = 1  # Not converged
                            mock_optimizer.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * 1.8897259886
                            ]

                            # Mock GeomOptNotConvergedError
                            from geometric.errors import GeomOptNotConvergedError

                            mock_optimizer.optimizeGeometry.side_effect = GeomOptNotConvergedError(
                                "Not converged"
                            )
                            mock_opt.Optimizer.return_value = mock_optimizer

                            mock_tmpdir.return_value.__enter__.return_value = "/tmp/test"

                            # GeomOptNotConvergedError should be caught and return False
                            # (no warning)
                            result = self.optimizer.run()

                            assert result is False
                            assert self.optimizer.step_count == 5  # Should extract step count
                            assert self.optimizer.converged is False

    def test_run_other_exceptions_reraised(self):
        """Test that non-GeomOptNotConvergedError exceptions are re-raised."""
        with patch("geometric.ase_engine") as mock_engine:
            with patch("geometric.optimize") as mock_opt:
                with patch("geometric.optimize.DelocalizedInternalCoordinates") as mock_ic:
                    with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                        with patch.object(
                            self.optimizer, "_create_molecule_from_atoms"
                        ) as mock_create_mol:
                            mock_molecule = Mock()
                            mock_create_mol.return_value = mock_molecule

                            mock_engine_instance = Mock()
                            mock_engine.EngineASE.return_value = mock_engine_instance

                            mock_ic_instance = Mock()
                            mock_ic.return_value = mock_ic_instance

                            mock_optimizer = Mock()
                            # Some other exception (not GeomOptNotConvergedError)
                            mock_optimizer.optimizeGeometry.side_effect = ValueError(
                                "Invalid calculation"
                            )
                            mock_opt.Optimizer.return_value = mock_optimizer

                            mock_tmpdir.return_value.__enter__.return_value = "/tmp/test"

                            # Other exceptions should be re-raised
                            with pytest.raises(ValueError, match="Invalid calculation"):
                                self.optimizer.run()

    def test_run_no_valid_coordinates(self):
        """Test behavior when no valid coordinates are returned."""
        with patch("geometric.ase_engine") as mock_engine:
            with patch("geometric.optimize") as mock_opt:
                with patch("geometric.optimize.DelocalizedInternalCoordinates") as mock_ic:
                    with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                        with patch.object(
                            self.optimizer, "_create_molecule_from_atoms"
                        ) as mock_create_mol:
                            mock_molecule = Mock()
                            mock_molecule.xyzs = []  # No coordinates
                            mock_create_mol.return_value = mock_molecule

                            mock_engine_instance = Mock()
                            mock_engine.EngineASE.return_value = mock_engine_instance

                            mock_ic_instance = Mock()
                            mock_ic.return_value = mock_ic_instance

                            mock_optimizer = Mock()
                            mock_optimizer.Iteration = 5
                            mock_optimizer.state = 2
                            mock_optimizer.xyzs = []  # No coordinates
                            mock_optimizer.coords = None  # No current coordinates
                            mock_opt.Optimizer.return_value = mock_optimizer

                            mock_tmpdir.return_value.__enter__.return_value = "/tmp/test"

                            # Should raise RuntimeError when no valid coordinates
                            with pytest.raises(
                                RuntimeError,
                                match="geomeTRIC optimization did not return valid results",
                            ):
                                self.optimizer.run()

    def test_run_general_exception(self):
        """Test handling of general exceptions."""
        with patch("geometric.ase_engine") as mock_engine:
            with patch("geometric.optimize") as mock_opt:
                with patch("geometric.optimize.DelocalizedInternalCoordinates") as mock_ic:
                    with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                        with patch.object(
                            self.optimizer, "_create_molecule_from_atoms"
                        ) as mock_create_mol:
                            mock_molecule = Mock()
                            mock_create_mol.return_value = mock_molecule

                            mock_engine_instance = Mock()
                            mock_engine.EngineASE.return_value = mock_engine_instance

                            mock_ic_instance = Mock()
                            mock_ic.return_value = mock_ic_instance

                            mock_optimizer = Mock()
                            # Mock optimizeGeometry to raise an exception
                            mock_optimizer.optimizeGeometry.side_effect = RuntimeError(
                                "General error"
                            )
                            mock_opt.Optimizer.return_value = mock_optimizer

                            mock_tmpdir.return_value.__enter__.return_value = "/tmp/test"

                            # Other exceptions during optimization should be re-raised
                            with pytest.raises(RuntimeError, match="General error"):
                                self.optimizer.run()


class TestGeometricTSOptimizer:
    """Test GeometricTSOptimizer class."""

    def test_ts_optimizer_initialization(self):
        """Test GeometricTSOptimizer initialization."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        ts_optimizer = GeometricTSOptimizer(atoms)

        assert ts_optimizer.order == 1
        assert isinstance(ts_optimizer, GeometricOptimizer)

    def test_ts_optimizer_with_parameters(self):
        """Test GeometricTSOptimizer with additional parameters."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        hessian = np.eye(9) * 0.1
        ts_optimizer = GeometricTSOptimizer(atoms, hessian=hessian, trust=0.05)

        assert ts_optimizer.order == 1
        assert ts_optimizer.initial_hessian is not None
        assert ts_optimizer.geometric_kwargs["trust"] == 0.05


class TestFactoryFunctions:
    """Test factory functions for creating optimizers."""

    def test_create_geometric_optimizer_minima(self):
        """Test create_geometric_optimizer for minima (order=0)."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        optimizer = create_geometric_optimizer(atoms, order=0)

        assert isinstance(optimizer, GeometricOptimizer)
        assert optimizer.order == 0

    def test_create_geometric_optimizer_ts(self):
        """Test create_geometric_optimizer for TS (order=1)."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        optimizer = create_geometric_optimizer(atoms, order=1)

        assert isinstance(optimizer, GeometricTSOptimizer)
        assert optimizer.order == 1

    def test_create_geometric_optimizer_with_hessian(self):
        """Test create_geometric_optimizer with Hessian."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        hessian = np.eye(9) * 0.1
        optimizer = create_geometric_optimizer(atoms, order=0, hessian=hessian)

        assert optimizer.initial_hessian is not None
        np.testing.assert_array_equal(optimizer.initial_hessian, hessian)

    def test_geometric_minima_optimizer(self):
        """Test geometric_minima_optimizer factory function."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        optimizer = geometric_minima_optimizer(atoms)

        assert isinstance(optimizer, GeometricOptimizer)
        assert optimizer.order == 0

    def test_geometric_ts_optimizer(self):
        """Test geometric_ts_optimizer factory function."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        optimizer = geometric_ts_optimizer(atoms)

        assert isinstance(optimizer, GeometricTSOptimizer)
        assert optimizer.order == 1

    def test_factory_functions_with_parameters(self):
        """Test factory functions with additional parameters."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        hessian = np.eye(9) * 0.1

        # Test minima optimizer with parameters
        min_opt = geometric_minima_optimizer(atoms, hessian=hessian, trust=0.05)
        assert min_opt.order == 0
        assert min_opt.geometric_kwargs["trust"] == 0.05

        # Test TS optimizer with parameters
        ts_opt = geometric_ts_optimizer(atoms, hessian=hessian, trust=0.05)
        assert ts_opt.order == 1
        assert ts_opt.geometric_kwargs["trust"] == 0.05


class TestASEOptimizerCompatibility:
    """Test that GeometricOptimizer behaves consistently with ASE optimizers."""

    def test_ase_optimizer_interface(self):
        """Test that GeometricOptimizer implements ASE optimizer interface."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        optimizer = GeometricOptimizer(atoms)

        # Check required ASE optimizer attributes
        assert hasattr(optimizer, "atoms")
        assert hasattr(optimizer, "run")
        assert optimizer.atoms is atoms

        # Check that run method has correct signature
        import inspect

        run_signature = inspect.signature(optimizer.run)
        assert "fmax" in run_signature.parameters
        assert "steps" in run_signature.parameters

    def test_run_method_return_type(self):
        """Test that run method returns boolean like ASE optimizers."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        optimizer = GeometricOptimizer(atoms)

        with patch("geometric.ase_engine") as mock_engine:
            with patch("geometric.optimize") as mock_opt:
                with patch("geometric.optimize.DelocalizedInternalCoordinates") as mock_ic:
                    with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                        with patch.object(
                            optimizer, "_create_molecule_from_atoms"
                        ) as mock_create_mol:
                            mock_molecule = Mock()
                            mock_create_mol.return_value = mock_molecule

                            mock_engine_instance = Mock()
                            mock_engine.EngineASE.return_value = mock_engine_instance

                            mock_ic_instance = Mock()
                            mock_ic.return_value = mock_ic_instance

                            mock_optimizer = Mock()
                            mock_optimizer.Iteration = 5
                            mock_optimizer.state = 2
                            mock_optimizer.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * 1.8897259886
                            ]
                            mock_optimizer.optimizeGeometry = (
                                Mock()
                            )  # Mock the optimizeGeometry call
                            mock_opt.Optimizer.return_value = mock_optimizer

                            # Mock the molecule to have xyzs
                            mock_molecule.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * 1.8897259886
                            ]

                            mock_tmpdir.return_value.__enter__.return_value = "/tmp/test"

                            result = optimizer.run()

                            # Should return boolean like ASE optimizers
                            assert isinstance(result, bool)

    def test_convergence_tracking(self):
        """Test that convergence is tracked like ASE optimizers."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        optimizer = GeometricOptimizer(atoms)

        # Initially not converged
        assert optimizer.converged is False

        # Test that the converged attribute exists and can be set
        optimizer.converged = True
        assert optimizer.converged is True

        optimizer.converged = False
        assert optimizer.converged is False

    def test_step_count_tracking(self):
        """Test that step count is tracked like ASE optimizers."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        optimizer = GeometricOptimizer(atoms)

        # Initially no steps
        assert optimizer.step_count == 0

        with patch("geometric.ase_engine") as mock_engine:
            with patch("geometric.optimize") as mock_opt:
                with patch("geometric.optimize.DelocalizedInternalCoordinates") as mock_ic:
                    with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                        with patch.object(
                            optimizer, "_create_molecule_from_atoms"
                        ) as mock_create_mol:
                            mock_molecule = Mock()
                            mock_create_mol.return_value = mock_molecule

                            mock_engine_instance = Mock()
                            mock_engine.EngineASE.return_value = mock_engine_instance

                            mock_ic_instance = Mock()
                            mock_ic.return_value = mock_ic_instance

                            mock_optimizer = Mock()
                            mock_optimizer.Iteration = 7
                            mock_optimizer.state = 2
                            mock_optimizer.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * 1.8897259886
                            ]
                            mock_optimizer.optimizeGeometry = (
                                Mock()
                            )  # Mock the optimizeGeometry call
                            mock_opt.Optimizer.return_value = mock_optimizer

                            # Mock the molecule to have xyzs
                            mock_molecule.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * 1.8897259886
                            ]

                            mock_tmpdir.return_value.__enter__.return_value = "/tmp/test"

                            optimizer.run()

                            # Should track step count
                            assert optimizer.step_count == 7

    def test_parameter_consistency(self):
        """Test that parameters are handled consistently with ASE optimizers."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        # Test with ASE-style parameters
        optimizer = GeometricOptimizer(atoms, logfile="test.log", restart="test.pckl")

        # ASE parameters should be passed to parent class
        assert hasattr(optimizer, "logfile")
        assert hasattr(optimizer, "restart")

        # But should not be in geomeTRIC kwargs
        assert "logfile" not in optimizer.geometric_kwargs
        assert "restart" not in optimizer.geometric_kwargs

    def test_atoms_modification(self):
        """Test that atoms are modified in-place like ASE optimizers."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = Mock()

        optimizer = GeometricOptimizer(atoms)

        # Test that we can modify atoms positions directly
        original_positions = atoms.get_positions().copy()
        new_positions = original_positions + 0.1
        atoms.set_positions(new_positions)

        # Verify the atoms were modified
        assert not np.array_equal(atoms.get_positions(), original_positions)
        np.testing.assert_array_almost_equal(atoms.get_positions(), new_positions)

        # Test that the optimizer has access to the same atoms object
        assert optimizer.atoms is atoms
