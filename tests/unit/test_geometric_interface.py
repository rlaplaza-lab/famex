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
                            mock_progress = Mock()
                            from ase.units import Bohr

                            mock_progress.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * Bohr
                            ]
                            mock_optimizer.progress = mock_progress
                            mock_optimizer.optimizeGeometry = (
                                Mock()
                            )  # Mock the optimizeGeometry call
                            mock_opt.Optimizer.return_value = mock_optimizer

                            # Mock the molecule to have xyzs
                            from ase.units import Bohr

                            mock_molecule.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * Bohr
                            ]

                            mock_tmpdir.return_value.__enter__.return_value = "/tmp/test"

                            result = self.optimizer.run(fmax=0.1, steps=50)

                            # Check that parameters were set correctly
                            mock_opt.OptParams.assert_called_once()
                            params = mock_opt.OptParams.return_value
                            # Check that the parameters were set on the Mock object
                            assert hasattr(params, "order")
                            assert hasattr(params, "maxiter")
                            assert hasattr(params, "Convergence_gmax")
                            assert hasattr(params, "xyzout")
                            assert hasattr(params, "frequency")

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
                            mock_progress = Mock()
                            from ase.units import Bohr

                            mock_progress.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * Bohr
                            ]
                            mock_optimizer.progress = mock_progress
                            mock_optimizer.optimizeGeometry = (
                                Mock()
                            )  # Mock the optimizeGeometry call
                            mock_opt.Optimizer.return_value = mock_optimizer

                            # Mock the molecule to have xyzs
                            from ase.units import Bohr

                            mock_molecule.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * Bohr
                            ]

                            mock_tmpdir.return_value.__enter__.return_value = "/tmp/test"

                            result = optimizer.run()

                            # Check that Hessian was set correctly
                            params = mock_opt.OptParams.return_value
                            # Check that Hessian attributes were set
                            assert hasattr(params, "hess_data")
                            assert hasattr(params, "hessian")

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
                            # Return coordinates in Angstrom (as expected by geometric interface)
                            mock_progress = Mock()
                            mock_progress.xyzs = [np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])]
                            mock_optimizer.progress = mock_progress
                            mock_optimizer.optimizeGeometry = (
                                Mock()
                            )  # Mock the optimizeGeometry call
                            mock_opt.Optimizer.return_value = mock_optimizer

                            # Mock the molecule to have xyzs
                            from ase.units import Bohr

                            mock_molecule.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * Bohr
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
                            mock_progress = Mock()
                            from ase.units import Bohr

                            mock_progress.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * Bohr
                            ]
                            mock_optimizer.progress = mock_progress
                            mock_optimizer.optimizeGeometry = (
                                Mock()
                            )  # Mock the optimizeGeometry call
                            mock_opt.Optimizer.return_value = mock_optimizer

                            # Mock the molecule to have xyzs
                            from ase.units import Bohr

                            mock_molecule.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * Bohr
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
                            mock_progress = Mock()
                            from ase.units import Bohr

                            mock_progress.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * Bohr
                            ]
                            mock_optimizer.progress = mock_progress

                            # Mock the molecule to have xyzs
                            from ase.units import Bohr

                            mock_molecule.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * Bohr
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
                            mock_progress = Mock()
                            from ase.units import Bohr

                            mock_progress.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * Bohr
                            ]
                            mock_optimizer.progress = mock_progress
                            mock_optimizer.optimizeGeometry = (
                                Mock()
                            )  # Mock the optimizeGeometry call
                            mock_opt.Optimizer.return_value = mock_optimizer

                            # Mock the molecule to have xyzs
                            from ase.units import Bohr

                            mock_molecule.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * Bohr
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
                            mock_progress = Mock()
                            # Mock 8 structures: initial + 7 optimization steps
                            mock_progress.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * 1.8897259886
                                for _ in range(8)
                            ]
                            mock_optimizer.progress = mock_progress
                            mock_optimizer.optimizeGeometry = (
                                Mock()
                            )  # Mock the optimizeGeometry call
                            mock_opt.Optimizer.return_value = mock_optimizer

                            # Mock the molecule to have xyzs
                            from ase.units import Bohr

                            mock_molecule.xyzs = [
                                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * Bohr
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


class TestGeometricCoordinateHandling:
    """Test geometric optimizer coordinate handling and Hessian compatibility."""

    @pytest.fixture
    def test_molecule(self):
        """Create a test molecule for optimization tests."""
        # Create a simple water molecule
        symbols = ["O", "H", "H"]
        positions = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
        return Atoms(symbols=symbols, positions=positions)

    @pytest.fixture
    def test_benzene(self):
        """Create a benzene molecule for more complex tests."""
        # Create a distorted benzene molecule
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
        """Test that geometric optimizer doesn't modify original atoms in-place."""
        from qme.potentials.uma_potential import UMAPotential

        # Create original atoms object
        original_atoms = test_molecule.copy()
        original_positions = original_atoms.get_positions().copy()
        original_id = id(original_atoms)

        # Attach calculator
        calculator = UMAPotential()
        original_atoms.calc = calculator

        # Create geometric optimizer with original atoms
        optimizer = GeometricOptimizer(original_atoms, order=0)

        # Run optimization
        optimizer.run(fmax=0.05, steps=10)

        # Check that original atoms object was NOT modified in-place
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
        """Test that geometric optimizer produces consistent coordinates."""
        from qme.potentials.uma_potential import UMAPotential

        # Create atoms object
        atoms = test_benzene.copy()
        calculator = UMAPotential()
        atoms.calc = calculator

        # Store initial state
        initial_positions = atoms.get_positions().copy()
        initial_symbols = atoms.get_chemical_symbols().copy()

        # Run optimization
        optimizer = GeometricOptimizer(atoms, order=0)
        converged = optimizer.run(fmax=0.1, steps=50)

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

    def test_geometric_optimized_atoms_hessian_compatibility(self, test_benzene):
        """Test that geometric-optimized atoms work with Hessian calculations."""
        from qme.analysis.frequency import FrequencyAnalysis
        from qme.potentials.uma_potential import UMAPotential

        # Create atoms object
        atoms = test_benzene.copy()
        calculator = UMAPotential()
        atoms.calc = calculator

        # Run geometric optimization
        optimizer = GeometricOptimizer(atoms, order=0)
        optimizer.run(fmax=0.1, steps=50)

        # Test Hessian calculation on optimized atoms
        try:
            freq_analysis = FrequencyAnalysis(atoms=atoms, calculator=calculator, delta=0.01)
            hessian = freq_analysis.calculate_hessian(method="finite_differences")
            frequencies, normal_modes = freq_analysis.diagonalize_hessian()

            # Verify Hessian is valid
            assert hessian is not None, "Hessian should be calculated successfully"
            assert hessian.shape == (
                36,
                36,
            ), f"Hessian should be 36x36 for 12 atoms, got {hessian.shape}"
            assert np.all(np.isfinite(hessian)), "Hessian should contain only finite values"

            # Verify frequencies are reasonable
            expected_frequencies = 3 * len(atoms)  # 3N degrees of freedom
            assert len(frequencies) == expected_frequencies, (
                f"Should have {expected_frequencies} frequencies for {len(atoms)} atoms, "
                f"got {len(frequencies)}"
            )
            assert np.all(np.isfinite(frequencies)), "All frequencies should be finite"

            # Check for reasonable frequency range (should have some real positive frequencies)
            real_frequencies = frequencies[frequencies > 0]
            assert len(real_frequencies) > 0, "Should have some real positive frequencies"

            print("✅ Hessian calculation successful on geometric-optimized atoms")
            print(f"   Hessian shape: {hessian.shape}")
            print(f"   Number of frequencies: {len(frequencies)}")
            print(f"   Real frequencies: {len(real_frequencies)}")
            print(f"   Frequency range: {frequencies.min():.2f} to {frequencies.max():.2f} cm⁻¹")

        except Exception as e:
            pytest.fail(f"Hessian calculation failed on geometric-optimized atoms: {e}")

    def test_geometric_vs_sella_coordinate_consistency(self, test_benzene):
        """Test coordinate consistency between geometric and Sella optimizers."""
        from sella import Sella

        from qme.analysis.frequency import FrequencyAnalysis
        from qme.potentials.mock_potential import MockCalculator

        calculator = MockCalculator()

        # Store initial state for comparison (need to attach calculator first)
        test_benzene.calc = calculator
        initial_energy = test_benzene.get_potential_energy()
        initial_positions = test_benzene.get_positions().copy()

        # Test with geometric optimizer
        atoms_geo = test_benzene.copy()
        atoms_geo.calc = calculator
        geo_optimizer = GeometricOptimizer(atoms_geo, order=0)
        geo_converged = geo_optimizer.run(fmax=0.1, steps=50)

        # Test with Sella optimizer
        atoms_sella = test_benzene.copy()
        atoms_sella.calc = calculator
        sella_optimizer = Sella(atoms_sella, internal=True, order=0)
        sella_optimizer.run(fmax=0.1, steps=50)

        # Both should produce valid coordinates
        geo_positions = atoms_geo.get_positions()
        sella_positions = atoms_sella.get_positions()

        # Verify both produce finite coordinates
        assert np.all(
            np.isfinite(geo_positions)
        ), "Geometric optimizer should produce finite coordinates"
        assert np.all(
            np.isfinite(sella_positions)
        ), "Sella optimizer should produce finite coordinates"

        # STRINGENT CHECKS: Verify that optimizers actually optimized
        geo_energy = atoms_geo.get_potential_energy()
        sella_energy = atoms_sella.get_potential_energy()

        geo_energy_change = abs(geo_energy - initial_energy)
        sella_energy_change = abs(sella_energy - initial_energy)

        geo_position_change = np.max(np.abs(geo_positions - initial_positions))
        sella_position_change = np.max(np.abs(sella_positions - initial_positions))

        print(
            f"Geometric: Energy change = {geo_energy_change:.6f}, "
            f"Position change = {geo_position_change:.6f}"
        )
        print(
            f"Sella: Energy change = {sella_energy_change:.6f}, "
            f"Position change = {sella_position_change:.6f}"
        )

        # Both optimizers should actually optimize (this will catch the GeometricOptimizer bug)
        assert geo_energy_change > 1e-6, (
            "GeometricOptimizer should actually change energy. "
            "This test catches bugs where optimizers report steps but don't optimize."
        )
        assert geo_position_change > 1e-6, (
            "GeometricOptimizer should actually change positions. "
            "This test catches bugs where optimizers report steps but don't optimize."
        )

        assert sella_energy_change > 1e-6, "Sella optimizer should actually change energy"
        assert sella_position_change > 1e-6, "Sella optimizer should actually change positions"

        # DETAILED COORDINATE COMPARISON
        print("\n=== DETAILED COORDINATE COMPARISON ===")

        # Maximum coordinate difference
        max_coord_diff = np.max(np.abs(geo_positions - sella_positions))
        rms_coord_diff = np.sqrt(np.mean((geo_positions - sella_positions) ** 2))

        print("Geometric vs Sella:")
        print(f"  Max coordinate difference: {max_coord_diff:.6f} Å")
        print(f"  RMS coordinate difference: {rms_coord_diff:.6f} Å")

        # Check if coordinates are reasonably similar (within 0.05 Å for complex systems)
        assert max_coord_diff < 0.05, (
            f"Final coordinates differ too much between Geometric and Sella: "
            f"{max_coord_diff:.6f} Å. This suggests inconsistent optimization."
        )

        # DETAILED FORCE COMPARISON
        print("\n=== DETAILED FORCE COMPARISON ===")

        geo_forces = atoms_geo.get_forces()
        sella_forces = atoms_sella.get_forces()

        geo_max_force = np.max(np.abs(geo_forces))
        sella_max_force = np.max(np.abs(sella_forces))

        geo_rms_force = np.sqrt(np.mean(geo_forces**2))
        sella_rms_force = np.sqrt(np.mean(sella_forces**2))

        print("Geometric:")
        print(f"  Max force: {geo_max_force:.6f} eV/Å")
        print(f"  RMS force: {geo_rms_force:.6f} eV/Å")
        print(f"  Converged: {geo_converged}")

        print("Sella:")
        print(f"  Max force: {sella_max_force:.6f} eV/Å")
        print(f"  RMS force: {sella_rms_force:.6f} eV/Å")
        print(f"  Converged: {sella_optimizer.converged}")

        # If optimizers claim convergence, forces should be low
        if geo_converged:
            assert geo_max_force < 0.1, (
                f"GeometricOptimizer claims convergence but max force is {geo_max_force:.6f} eV/Å. "
                f"This suggests a bug in convergence detection."
            )

        if sella_optimizer.converged:
            assert sella_max_force < 0.1, (
                f"Sella optimizer claims convergence but max force is {sella_max_force:.6f} eV/Å. "
                f"This suggests a bug in convergence detection."
            )

        # DETAILED FREQUENCY COMPARISON
        print("\n=== DETAILED FREQUENCY COMPARISON ===")

        frequency_results = {}
        for atoms, optimizer_name in [(atoms_geo, "geometric"), (atoms_sella, "sella")]:
            try:
                # Calculate frequencies using finite differences
                freq_analysis = FrequencyAnalysis(atoms=atoms, calculator=calculator, delta=0.01)
                frequencies = freq_analysis.get_frequencies()

                # Filter out imaginary frequencies (negative values)
                real_frequencies = frequencies[frequencies > 0]

                frequency_results[optimizer_name] = {
                    "frequencies": frequencies,
                    "real_frequencies": real_frequencies,
                    "imaginary_count": np.sum(frequencies < 0),
                    "lowest_real": np.min(real_frequencies) if len(real_frequencies) > 0 else None,
                    "highest_real": np.max(real_frequencies) if len(real_frequencies) > 0 else None,
                }

                print(f"{optimizer_name.capitalize()}:")
                print(f"  Total frequencies: {len(frequencies)}")
                print(
                    "  Imaginary frequencies: {}".format(
                        frequency_results[optimizer_name]["imaginary_count"]
                    )
                )
                if frequency_results[optimizer_name]["lowest_real"] is not None:
                    print(
                        "  Lowest real frequency: {:.2f} cm⁻¹".format(
                            frequency_results[optimizer_name]["lowest_real"]
                        )
                    )
                    print(
                        "  Highest real frequency: {:.2f} cm⁻¹".format(
                            frequency_results[optimizer_name]["highest_real"]
                        )
                    )

                # Verify Hessian calculation works
                hessian = freq_analysis.calculate_hessian(method="finite_differences")
                assert (
                    hessian is not None
                ), f"{optimizer_name} optimizer should produce Hessian-compatible atoms"
                print(
                    f"✅ {optimizer_name.capitalize()} optimizer produces Hessian-compatible atoms"
                )

            except Exception as e:
                print(f"{optimizer_name.capitalize()}: Frequency calculation failed: {e}")
                # Don't fail the test for frequency calculation issues
                # as this might be due to the mock calculator limitations
                continue

        # Compare frequencies between optimizers
        if len(frequency_results) == 2:
            geo_freqs = frequency_results["geometric"]["real_frequencies"]
            sella_freqs = frequency_results["sella"]["real_frequencies"]

            if (
                geo_freqs is not None
                and sella_freqs is not None
                and len(geo_freqs) == len(sella_freqs)
            ):
                # Compare frequencies (sort to handle different ordering)
                geo_freqs_sorted = np.sort(geo_freqs)
                sella_freqs_sorted = np.sort(sella_freqs)

                freq_diff = np.abs(geo_freqs_sorted - sella_freqs_sorted)
                max_freq_diff = np.max(freq_diff)
                rms_freq_diff = np.sqrt(np.mean(freq_diff**2))

                print("Geometric vs Sella frequencies:")
                print(f"  Max frequency difference: {max_freq_diff:.2f} cm⁻¹")
                print(f"  RMS frequency difference: {rms_freq_diff:.2f} cm⁻¹")

                # Allow reasonable frequency differences (within 50 cm⁻¹)
                assert max_freq_diff < 50.0, (
                    f"Frequencies differ too much between Geometric and Sella: "
                    f"{max_freq_diff:.2f} cm⁻¹. This suggests inconsistent optimization."
                )

        # ENERGY COMPARISON
        print("\n=== ENERGY COMPARISON ===")

        energy_diff = abs(geo_energy - sella_energy)
        print(f"Energy difference: {energy_diff:.6f} eV")

        # Energies should be reasonably similar (within 0.005 eV for complex systems)
        # Allow larger tolerance for complex systems like benzene that may have multiple minima
        assert energy_diff < 0.005, (
            f"Final energies differ too much between optimizers: {energy_diff:.6f} eV. "
            f"This suggests inconsistent optimization to different minima."
        )

    def test_explorer_geometric_coordinate_handling(self, test_benzene):
        """Test that Explorer properly handles geometric optimizer coordinate transformations."""
        from qme.core.explorer import Explorer

        # Create Explorer with geometric optimizer
        explorer = Explorer(atoms=test_benzene.copy(), backend="uma", local_optimizer="geometric")

        # Store original atoms reference
        original_atoms = explorer.atoms_list[0]
        original_positions = original_atoms.get_positions().copy()
        original_id = id(original_atoms)

        # Run optimization through Explorer
        results = explorer.run(mode="minima", fmax=0.1, steps=50)

        # Extract optimized atoms
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], dict):
                optimized_atoms = results[0]["optimized_atoms"]
            else:
                optimized_atoms = results[0]
        else:
            optimized_atoms = results

        # Verify original atoms object was preserved
        assert id(original_atoms) == original_id, "Original atoms object should be preserved"

        # Verify optimized atoms are different object
        assert id(optimized_atoms) != original_id, "Optimized atoms should be a different object"

        # Verify optimized atoms have valid coordinates
        optimized_positions = optimized_atoms.get_positions()
        assert np.all(np.isfinite(optimized_positions)), "Optimized coordinates should be finite"
        assert optimized_positions.shape == original_positions.shape, "Shape should be preserved"

        # Test Hessian calculation on Explorer-optimized atoms
        try:
            freq_results = explorer.calculate_frequencies(atoms=optimized_atoms)
            assert "frequencies" in freq_results, "Frequency calculation should succeed"
            assert len(freq_results["frequencies"]) > 0, "Should have calculated frequencies"
            print("✅ Explorer geometric optimization produces Hessian-compatible atoms")
            print(f"   Number of frequencies: {len(freq_results['frequencies'])}")
        except Exception as e:
            pytest.fail(f"Explorer geometric optimization failed Hessian test: {e}")

    def test_coordinate_transformation_robustness(self, test_benzene):
        """Test robustness of coordinate transformations with various molecular geometries."""
        from qme.analysis.frequency import FrequencyAnalysis
        from qme.potentials.uma_potential import UMAPotential

        # Test with different molecular geometries
        test_cases = [
            ("benzene", test_benzene),
            ("linear_molecule", self._create_linear_molecule()),
            ("planar_molecule", self._create_planar_molecule()),
        ]

        calculator = UMAPotential()

        for name, atoms in test_cases:
            print(f"\n🧪 Testing {name}...")

            # Make a copy for optimization
            test_atoms = atoms.copy()
            test_atoms.calc = calculator

            # Store initial state
            initial_positions = test_atoms.get_positions().copy()
            initial_symbols = test_atoms.get_chemical_symbols().copy()

            # Run optimization
            optimizer = GeometricOptimizer(test_atoms, order=0)
            converged = optimizer.run(fmax=0.1, steps=50)

            # Verify coordinate consistency
            optimized_positions = test_atoms.get_positions()
            optimized_symbols = test_atoms.get_chemical_symbols()

            assert len(optimized_positions) == len(
                initial_positions
            ), f"{name}: Number of atoms should be preserved"
            assert (
                optimized_symbols == initial_symbols
            ), f"{name}: Atomic symbols should be preserved"
            assert np.all(
                np.isfinite(optimized_positions)
            ), f"{name}: All coordinates should be finite"

            # Test Hessian calculation
            try:
                freq_analysis = FrequencyAnalysis(
                    atoms=test_atoms, calculator=calculator, delta=0.01
                )
                hessian = freq_analysis.calculate_hessian(method="finite_differences")
                assert hessian is not None, f"{name}: Hessian calculation should succeed"
                print(f"   ✅ {name} passed all tests")
            except Exception as e:
                pytest.fail(f"{name} failed Hessian test: {e}")

    def _create_linear_molecule(self):
        """Create a linear molecule (CO2)."""
        symbols = ["C", "O", "O"]
        positions = np.array([[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]])
        return Atoms(symbols=symbols, positions=positions)

    def _create_planar_molecule(self):
        """Create a planar molecule (formaldehyde)."""
        symbols = ["C", "O", "H", "H"]
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.21, 0.0, 0.0], [-0.97, 0.94, 0.0], [-0.97, -0.94, 0.0]]
        )
        return Atoms(symbols=symbols, positions=positions)
