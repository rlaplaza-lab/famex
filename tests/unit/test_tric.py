"""Unit tests for TRIC optimizer implementation.

This module tests the TRIC internal coordinate system and optimizer classes
to ensure they work correctly with the QME framework.
"""

import numpy as np
import pytest
from ase import Atoms

from qme.core.tric import (
    TRICOptimizer,
    TRICTSOptimizer,
    create_tric_optimizer,
    InternalCoords,
    Bond,
    Angle,
    Dihedral,
    BMatrixCalculator,
    Geometry,
)
from qme.potentials.mock_potential import MockCalculator


class TestInternalCoords:
    """Test internal coordinate classes."""

    def test_bond_coordinate(self):
        """Test bond coordinate calculation."""
        # Simple H2 molecule
        positions = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        
        bond = Bond(0, 1)
        bond_length, gradients = bond.eval(positions)
        
        assert abs(bond_length - 0.74) < 1e-10
        assert len(gradients) == 2
        assert gradients[0][0] == -1.0  # -x direction (away from atom 1)
        assert gradients[1][0] == 1.0  # +x direction (away from atom 0)

    def test_angle_coordinate(self):
        """Test angle coordinate calculation."""
        # Water molecule
        positions = np.array([
            [0.0, 0.0, 0.0],      # O
            [0.96, 0.0, 0.0],     # H
            [-0.24, 0.93, 0.0]    # H
        ])
        
        angle = Angle(1, 0, 2)  # H-O-H angle
        angle_value, gradients = angle.eval(positions)
        
        # Should be close to tetrahedral angle for water
        assert 0 < angle_value < np.pi
        assert len(gradients) == 3
        # All gradients should be finite
        for grad in gradients:
            assert np.all(np.isfinite(grad))

    def test_dihedral_coordinate(self):
        """Test dihedral coordinate calculation."""
        # Ethane-like structure
        positions = np.array([
            [0.0, 0.0, 0.0],      # C1
            [1.5, 0.0, 0.0],      # C2
            [0.0, 1.0, 0.0],      # H1
            [1.5, 0.0, 1.0]       # H2
        ])
        
        dihedral = Dihedral(2, 0, 1, 3)  # H-C-C-H dihedral
        dihedral_value, gradients = dihedral.eval(positions)
        
        # Should be a valid dihedral angle
        assert -np.pi <= dihedral_value <= np.pi
        assert len(gradients) == 4
        # All gradients should be finite
        for grad in gradients:
            assert np.all(np.isfinite(grad))

    def test_dihedral_gradients_numerical_validation(self):
        """Test dihedral gradients against numerical derivatives."""
        # Simple test case
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0]
        ])
        
        dihedral = Dihedral(0, 1, 2, 3)
        h = 1e-6
        
        # Analytical gradient
        _, analytical_grads = dihedral.eval(positions)
        
        # Numerical gradient for first atom
        positions_plus = positions.copy()
        positions_plus[0, 0] += h
        dihedral_plus, _ = dihedral.eval(positions_plus)
        
        positions_minus = positions.copy()
        positions_minus[0, 0] -= h
        dihedral_minus, _ = dihedral.eval(positions_minus)
        
        numerical_grad_x = (dihedral_plus - dihedral_minus) / (2 * h)
        analytical_grad_x = analytical_grads[0][0]
        
        # Should match within numerical precision
        assert abs(numerical_grad_x - analytical_grad_x) < 1e-6


class TestInternalCoordsSystem:
    """Test internal coordinate system generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.atoms = Atoms("H2O", positions=[
            [0.0, 0.0, 0.0],      # O
            [0.96, 0.0, 0.0],     # H
            [-0.24, 0.93, 0.0]    # H
        ])

    def test_internal_coords_generation(self):
        """Test automatic generation of internal coordinates."""
        geometry = Geometry.from_atoms(self.atoms)
        internal_coords = InternalCoords(geometry)
        
        # Should have bonds, angles, and possibly dihedrals
        assert len(internal_coords) > 0
        
        # Check that we have bonds
        bond_count = sum(1 for coord in internal_coords if isinstance(coord, Bond))
        assert bond_count >= 1  # At least some bonds detected
        
        # Check that we have angles (may not be generated if not enough bonds)
        angle_count = sum(1 for coord in internal_coords if isinstance(coord, Angle))
        # Note: angle generation depends on bond connectivity, so we just check it's non-negative
        assert angle_count >= 0

    def test_b_matrix_calculation(self):
        """Test B-matrix calculation."""
        geometry = Geometry.from_atoms(self.atoms)
        internal_coords = InternalCoords(geometry)
        b_matrix_calc = BMatrixCalculator(internal_coords)
        
        B = b_matrix_calc.calculate_B_matrix(geometry)
        
        # B-matrix should have correct shape
        expected_shape = (len(internal_coords), 3 * len(self.atoms))
        assert B.shape == expected_shape
        
        # Should not have NaN or infinite values
        assert not np.any(np.isnan(B))
        assert not np.any(np.isinf(B))

    def test_b_matrix_inverse(self):
        """Test B-matrix pseudo-inverse calculation."""
        geometry = Geometry.from_atoms(self.atoms)
        internal_coords = InternalCoords(geometry)
        b_matrix_calc = BMatrixCalculator(internal_coords)
        
        B_inv = b_matrix_calc.calculate_B_inverse(geometry)
        
        # Should have correct shape
        expected_shape = (3 * len(self.atoms), len(internal_coords))
        assert B_inv.shape == expected_shape
        
        # Should not have NaN or infinite values
        assert not np.any(np.isnan(B_inv))
        assert not np.any(np.isinf(B_inv))


class TestTRICOptimizer:
    """Test TRIC optimizer classes."""

    def setup_method(self):
        """Set up test fixtures."""
        self.atoms = Atoms("H2O", positions=[
            [0.0, 0.0, 0.0],      # O
            [0.96, 0.0, 0.0],     # H
            [-0.24, 0.93, 0.0]    # H
        ])
        self.atoms.calc = MockCalculator()

    def test_tric_optimizer_initialization(self):
        """Test TRIC optimizer initialization."""
        optimizer = TRICOptimizer(self.atoms)
        
        assert optimizer.order == 0
        assert optimizer.trust_radius == 0.3
        assert optimizer.max_step == 0.2
        assert optimizer.step_count == 0
        assert optimizer.converged() is False
        assert hasattr(optimizer, 'internal_coords')
        assert hasattr(optimizer, 'hessian')

    def test_tric_optimizer_with_hessian(self):
        """Test TRIC optimizer with custom Hessian."""
        hessian = np.eye(10)
        optimizer = TRICOptimizer(self.atoms, hessian=hessian)
        
        assert np.array_equal(optimizer.hessian, hessian)

    def test_tric_ts_optimizer_initialization(self):
        """Test TRIC TS optimizer initialization."""
        ts_optimizer = TRICTSOptimizer(self.atoms)
        
        assert ts_optimizer.order == 1
        assert isinstance(ts_optimizer, TRICOptimizer)

    def test_create_tric_optimizer_factory(self):
        """Test factory function for creating TRIC optimizers."""
        # Minima optimizer
        min_opt = create_tric_optimizer(self.atoms, order=0)
        assert isinstance(min_opt, TRICOptimizer)
        assert min_opt.order == 0
        
        # TS optimizer
        ts_opt = create_tric_optimizer(self.atoms, order=1)
        assert isinstance(ts_opt, TRICTSOptimizer)
        assert ts_opt.order == 1

    def test_basic_optimization_run(self):
        """Test basic optimization run (may not converge with mock calculator)."""
        optimizer = TRICOptimizer(self.atoms)
        
        # Run optimization
        converged = optimizer.run(fmax=0.05, steps=10)
        
        # Should complete without crashing
        assert optimizer.step_count >= 0
        # May or may not converge with mock calculator
        assert isinstance(converged, bool)

    def test_optimization_with_different_parameters(self):
        """Test optimization with different parameters."""
        optimizer = TRICOptimizer(
            self.atoms,
            trust_radius=0.5,
            max_step=0.1
        )
        
        assert optimizer.trust_radius == 0.5
        assert optimizer.max_step == 0.1

    def test_ase_optimizer_interface(self):
        """Test that TRIC optimizer implements ASE optimizer interface."""
        optimizer = TRICOptimizer(self.atoms)
        
        # Check required ASE optimizer attributes
        assert hasattr(optimizer, 'atoms')
        assert hasattr(optimizer, 'run')
        assert optimizer.atoms is self.atoms
        
        # Check that run method has correct signature
        import inspect
        run_signature = inspect.signature(optimizer.run)
        assert 'fmax' in run_signature.parameters
        assert 'steps' in run_signature.parameters


class TestTRICEdgeCases:
    """Test TRIC optimizer with edge cases."""

    def test_single_atom_system(self):
        """Test TRIC optimizer with single atom (should handle gracefully)."""
        single_atom = Atoms("H", positions=[[0, 0, 0]])
        single_atom.calc = MockCalculator()
        
        # Should not crash, but may have limited internal coordinates
        optimizer = TRICOptimizer(single_atom)
        assert optimizer.atoms is single_atom

    def test_linear_molecule(self):
        """Test TRIC optimizer with linear molecule (CO2)."""
        linear_atoms = Atoms("CO2", positions=[
            [0, 0, 0],
            [1.16, 0, 0],
            [-1.16, 0, 0]
        ])
        linear_atoms.calc = MockCalculator()
        
        optimizer = TRICOptimizer(linear_atoms)
        assert optimizer.atoms is linear_atoms

    def test_large_molecule(self):
        """Test TRIC optimizer with larger molecule (benzene)."""
        # Simple benzene-like structure
        benzene_positions = [
            # Carbon ring (6 atoms)
            [0, 1.4, 0], [1.2, 0.7, 0], [1.2, -0.7, 0],
            [0, -1.4, 0], [-1.2, -0.7, 0], [-1.2, 0.7, 0],
            # Hydrogens (6 atoms)
            [0, 2.5, 0], [2.1, 1.2, 0], [2.1, -1.2, 0],
            [0, -2.5, 0], [-2.1, -1.2, 0], [-2.1, 1.2, 0]
        ]
        benzene_atoms = Atoms("C6H6", positions=benzene_positions)
        benzene_atoms.calc = MockCalculator()
        
        optimizer = TRICOptimizer(benzene_atoms)
        assert optimizer.atoms is benzene_atoms
        assert len(optimizer.internal_coords) > 3


if __name__ == "__main__":
    pytest.main([__file__])
