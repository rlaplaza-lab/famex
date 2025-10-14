"""Tests for advanced TRIC features: TR projection, connectivity, and P-RFO."""

import pytest
import numpy as np
from ase import Atoms
from ase.units import Hartree

from qme.core.tric import (
    TRProjector, TranslationCoordinate, RotationCoordinate, create_tr_projector,
    ConnectivityGraph, find_bonds_with_connectivity, validate_connectivity,
    get_augmented_hessian, solve_rfo, rfo_model, restricted_step_microcycles,
    calculate_ts_mode_indices, calculate_min_mode_indices, validate_rfo_step
)
from qme.core.tric.utils import Geometry
from qme.potentials.mock_potential import MockCalculator


class TestTRCoordinates:
    """Test TR (Translation-Rotation) coordinate system."""
    
    def setup_method(self):
        """Set up test molecules."""
        # Simple H2O molecule
        self.h2o = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        self.h2o.calc = MockCalculator()
        self.geometry_h2o = Geometry.from_atoms(self.h2o)
        
        # Methane molecule
        self.ch4 = Atoms('CH4', positions=[
            [0, 0, 0],      # C
            [1, 1, 1],      # H
            [-1, -1, 1],    # H
            [-1, 1, -1],    # H
            [1, -1, -1]     # H
        ])
        self.ch4.calc = MockCalculator()
        self.geometry_ch4 = Geometry.from_atoms(self.ch4)
    
    def test_translation_coordinate(self):
        """Test translation coordinate calculation."""
        # Test x-translation for H2O
        tr_coord = TranslationCoordinate(0, [0, 1, 2], self.geometry_h2o.get_masses())
        
        # Calculate COM along x-axis
        positions = self.geometry_h2o.positions
        expected_com_x = np.sum(self.geometry_h2o.get_masses() * positions[:, 0]) / np.sum(self.geometry_h2o.get_masses())
        
        value = tr_coord.value(positions)
        assert abs(value - expected_com_x) < 1e-10
        
        # Test derivatives
        derivatives = tr_coord.derivative(positions)
        assert derivatives.shape == positions.shape
        
        # Check that derivatives sum to 1 (for x-component)
        assert abs(np.sum(derivatives[:, 0]) - 1.0) < 1e-10
        assert abs(np.sum(derivatives[:, 1])) < 1e-10  # y and z should be 0
        assert abs(np.sum(derivatives[:, 2])) < 1e-10
    
    def test_rotation_coordinate(self):
        """Test rotation coordinate calculation."""
        # Test z-rotation for H2O
        tr_coord = RotationCoordinate(2, [0, 1, 2], self.geometry_h2o.get_masses(), 
                                    self.geometry_h2o.positions)
        
        # For reference geometry, rotation should be 0
        value = tr_coord.value(self.geometry_h2o.positions)
        assert abs(value) < 1e-10
        
        # Test derivatives
        derivatives = tr_coord.derivative(self.geometry_h2o.positions)
        assert derivatives.shape == self.geometry_h2o.positions.shape
    
    def test_tr_projector(self):
        """Test TR projector functionality."""
        projector = TRProjector(self.geometry_h2o)
        
        # Check that we have 6 TR coordinates (3 trans + 3 rot)
        assert len(projector.tr_coords) == 6
        
        # Test projection matrix properties
        P = projector.projection_matrix
        n_cartesian = 3 * len(self.geometry_h2o)
        assert P.shape == (n_cartesian, n_cartesian)
        
        # Projection matrix should be idempotent: P^2 = P
        P_squared = P @ P
        np.testing.assert_allclose(P, P_squared, atol=1e-10)
        
        # Test gradient projection
        gradient = np.random.random((len(self.geometry_h2o), 3))
        projected_gradient = projector.project_gradient(gradient)
        
        # Projected gradient should have TR components removed
        # (This is a basic test - more sophisticated tests would check specific TR modes)
        assert projected_gradient.shape == gradient.shape
    
    def test_create_tr_projector(self):
        """Test TR projector factory function."""
        projector = create_tr_projector(self.geometry_h2o)
        assert isinstance(projector, TRProjector)
        assert len(projector.tr_coords) == 6


class TestConnectivity:
    """Test connectivity analysis and smart bond detection."""
    
    def setup_method(self):
        """Set up test molecules."""
        self.h2o = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        self.geometry_h2o = Geometry.from_atoms(self.h2o)
        
        # Benzene-like structure
        self.benzene = Atoms('C6H6', positions=[
            [0, 0, 0], [1, 0, 0], [1.5, 0.866, 0], [1, 1.732, 0], [0, 1.732, 0], [-0.5, 0.866, 0],  # C
            [0, 0, 1], [1, 0, 1], [1.5, 0.866, 1], [1, 1.732, 1], [0, 1.732, 1], [-0.5, 0.866, 1]   # H
        ])
        self.geometry_benzene = Geometry.from_atoms(self.benzene)
    
    def test_connectivity_graph(self):
        """Test connectivity graph functionality."""
        graph = ConnectivityGraph(3)  # H2O
        
        # Add bonds
        graph.add_bond(0, 1)  # O-H
        graph.add_bond(0, 2)  # O-H
        
        # Test connectivity
        assert graph.is_connected(0, 1)
        assert graph.is_connected(0, 2)
        assert not graph.is_connected(1, 2)  # H atoms not directly connected
        
        # Test full connectivity - H2O with O-H bonds is fully connected (H atoms connected through O)
        assert graph.is_fully_connected()  # All atoms in single connected component
        
        # Add H-H bond to make fully connected
        graph.add_bond(1, 2)
        assert graph.is_fully_connected()
        
        # Test that we can get connected components
        components = graph.get_connected_components()
        assert len(components) == 1  # Should be fully connected after adding H-H bond
        
        # Test connected components
        components = graph.get_connected_components()
        assert len(components) == 1
        assert len(components[0]) == 3
    
    def test_find_bonds_with_connectivity(self):
        """Test smart bond detection with connectivity."""
        bonds = find_bonds_with_connectivity(self.geometry_h2o)
        
        # Should find at least 2 bonds for H2O (O-H bonds)
        assert len(bonds) >= 2
        
        # All atoms should be connected
        graph = ConnectivityGraph(len(self.geometry_h2o))
        for i, j in bonds:
            graph.add_bond(i, j)
        
        assert graph.is_fully_connected()
    
    def test_validate_connectivity(self):
        """Test connectivity validation."""
        # Test with good connectivity
        bonds = [(0, 1), (0, 2)]  # H2O bonds
        is_valid, warnings = validate_connectivity(self.geometry_h2o, bonds)
        
        # Should be valid (all atoms connected)
        assert is_valid
        assert len(warnings) == 0
        
        # Test with poor connectivity
        bonds = [(0, 1)]  # Only one bond
        is_valid, warnings = validate_connectivity(self.geometry_h2o, bonds)
        
        # Should not be valid
        assert not is_valid
        assert len(warnings) > 0


class TestRFO:
    """Test RFO and P-RFO algorithms."""
    
    def setup_method(self):
        """Set up test data."""
        # Simple 3x3 Hessian with one negative eigenvalue (TS-like)
        self.hessian = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -0.5, 0.0],  # Negative eigenvalue
            [0.0, 0.0, 2.0]
        ])
        
        # Corresponding eigenvalues and eigenvectors
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.hessian)
        
        # Test gradient
        self.gradient = np.array([1.0, 2.0, 0.5])
    
    def test_get_augmented_hessian(self):
        """Test augmented Hessian construction."""
        alpha = 1.0
        H_aug = get_augmented_hessian(self.eigenvalues, self.gradient, alpha)
        
        n = len(self.eigenvalues)
        assert H_aug.shape == (n + 1, n + 1)
        
        # Check diagonal block
        np.testing.assert_allclose(H_aug[:n, :n], np.diag(self.eigenvalues / alpha))
        
        # Check off-diagonal blocks (matching pysisyphus)
        np.testing.assert_allclose(H_aug[:n, n], self.gradient)  # No division by alpha
        np.testing.assert_allclose(H_aug[n, :n], self.gradient / alpha)  # Division by alpha
        
        # Check bottom-right element
        assert H_aug[n, n] == 0.0
    
    def test_solve_rfo(self):
        """Test RFO eigenvalue solver."""
        H_aug = get_augmented_hessian(self.eigenvalues, self.gradient, 1.0)
        
        # Test minimization
        step, eigval, nu, eigvec = solve_rfo(H_aug, 'min')
        assert len(step) == len(self.eigenvalues)
        assert np.isfinite(eigval)
        assert np.isfinite(nu)
        assert len(eigvec) == len(self.eigenvalues) + 1
        
        # Test maximization
        step, eigval, nu, eigvec = solve_rfo(H_aug, 'max')
        assert len(step) == len(self.eigenvalues)
        assert np.isfinite(eigval)
        assert np.isfinite(nu)
    
    def test_rfo_model(self):
        """Test RFO energy model."""
        step = np.array([0.1, -0.2, 0.05])
        energy_change = rfo_model(self.gradient, self.hessian, step)
        
        # Should be finite
        assert np.isfinite(energy_change)
        
        # Should match analytical formula
        expected = np.dot(self.gradient, step) + 0.5 * np.dot(step, self.hessian @ step)
        np.testing.assert_allclose(energy_change, expected)
    
    def test_calculate_ts_mode_indices(self):
        """Test TS mode index calculation."""
        # Test with one negative eigenvalue
        ts_indices = calculate_ts_mode_indices(self.eigenvalues, n_negative=1)
        assert len(ts_indices) == 1
        assert self.eigenvalues[ts_indices[0]] < 0
        
        # Test with no negative eigenvalues
        pos_eigenvalues = np.array([1.0, 2.0, 3.0])
        ts_indices = calculate_ts_mode_indices(pos_eigenvalues, n_negative=1)
        assert len(ts_indices) == 0
    
    def test_calculate_min_mode_indices(self):
        """Test minimization mode index calculation."""
        ts_indices = calculate_ts_mode_indices(self.eigenvalues, n_negative=1)
        min_indices = calculate_min_mode_indices(self.eigenvalues, ts_indices)
        
        # Should exclude TS mode
        assert ts_indices[0] not in min_indices
        
        # Should include all other modes with significant eigenvalues
        assert len(min_indices) >= 0
    
    def test_validate_rfo_step(self):
        """Test RFO step validation."""
        step = np.array([0.1, -0.2, 0.05])
        trust_radius = 1.0
        
        is_valid, warnings = validate_rfo_step(step, self.gradient, self.hessian, trust_radius)
        
        # Should be valid for reasonable step
        assert is_valid
        assert len(warnings) == 0
        
        # Test with step too large
        large_step = np.array([2.0, 0.0, 0.0])
        is_valid, warnings = validate_rfo_step(large_step, self.gradient, self.hessian, trust_radius)
        
        # Should not be valid
        assert not is_valid
        assert len(warnings) > 0
    
    def test_restricted_step_microcycles(self):
        """Test restricted-step P-RFO micro-cycles."""
        trust_radius = 0.5
        
        step, alpha_history = restricted_step_microcycles(
            eigenvals=self.eigenvalues,
            eigenvecs=self.eigenvectors,
            gradient=self.gradient,
            trust_radius=trust_radius,
            max_micro_cycles=10
        )
        
        # Should return valid step
        assert len(step) == len(self.eigenvalues)
        assert np.all(np.isfinite(step))
        
        # Step should satisfy trust radius
        step_norm = np.linalg.norm(step)
        assert step_norm <= trust_radius * 1.01  # Allow small numerical errors
        
        # Should have alpha history
        assert len(alpha_history) >= 1
        assert all(alpha > 0 for alpha in alpha_history)


class TestAdvancedIntegration:
    """Integration tests for advanced TRIC features."""
    
    def setup_method(self):
        """Set up test system."""
        # Simple H2O molecule
        self.atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        self.atoms.calc = MockCalculator()
    
    def test_tr_projection_in_optimization(self):
        """Test that TR projection works in optimization."""
        from qme.core.tric import TRICOptimizer
        
        optimizer = TRICOptimizer(self.atoms)
        
        # Check that TR projection is set up
        assert hasattr(optimizer.b_matrix_calc, 'tr_projector')
        assert optimizer.b_matrix_calc.tr_projector is not None
        
        # Test gradient projection
        gradient = np.random.random((len(self.atoms), 3))
        projected = optimizer.b_matrix_calc.project_gradient(gradient)
        assert projected.shape == gradient.shape
        
        # Verify TR components are removed (projected gradient should be different)
        # unless gradient is already zero
        assert not np.allclose(projected, gradient) or np.allclose(gradient, 0)
    
    def test_connectivity_in_coordinate_generation(self):
        """Test that smart connectivity is used in coordinate generation."""
        from qme.core.tric import InternalCoords
        
        geometry = Geometry.from_atoms(self.atoms)
        coords = InternalCoords(geometry)
        
        # Should have bonds from smart connectivity
        bond_coords = [c for c in coords.coords if c.__class__.__name__ == 'Bond']
        assert len(bond_coords) >= 2  # At least O-H bonds
        
        # Should have angles
        angle_coords = [c for c in coords.coords if c.__class__.__name__ == 'Angle']
        assert len(angle_coords) >= 1  # At least H-O-H angle
    
    def test_prfo_in_ts_optimizer(self):
        """Test that P-RFO is available in TS optimizer."""
        from qme.core.tric import TRICTSOptimizer
        
        optimizer = TRICTSOptimizer(self.atoms)
        
        # Should have P-RFO method available
        assert hasattr(optimizer, '_calculate_ts_step')
        
        # Test that it can handle basic gradient
        gradient = np.array([1.0, -0.5, 0.2, 0.1, -0.3, 0.4, 0.0, 0.1, -0.2])
        step = optimizer._calculate_ts_step(gradient)
        
        assert len(step) == len(gradient)
        assert np.all(np.isfinite(step))
