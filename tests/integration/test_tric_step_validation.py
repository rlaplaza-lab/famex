"""Test TRIC step validation and coordinate transformations.

This module tests the critical coordinate transformation steps in TRIC optimization.
"""

import numpy as np
import pytest
from ase import Atoms

from qme.core.tric.utils import Geometry
from qme.core.tric.internal_coords import InternalCoords
from qme.core.tric.b_matrix import BMatrixCalculator
from qme.backend_availability import get_available_backends


def create_eclipsed_ethane():
    """Create eclipsed ethane geometry (C2H6)."""
    cc_bond = 1.54
    ch_bond = 1.09
    
    # Carbon positions
    c1_pos = np.array([0.0, 0.0, 0.0])
    c2_pos = np.array([cc_bond, 0.0, 0.0])
    
    # Hydrogen positions around C1 (tetrahedral)
    h1_c1 = ch_bond * np.array([1, 1, 1]) / np.sqrt(3)
    h2_c1 = ch_bond * np.array([1, -1, -1]) / np.sqrt(3)
    h3_c1 = ch_bond * np.array([-1, 1, -1]) / np.sqrt(3)
    h4_c1 = ch_bond * np.array([-1, -1, 1]) / np.sqrt(3)
    
    # Hydrogen positions around C2 (eclipsed with C1)
    theta = np.pi / 3  # 60 degrees
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    h1_c2 = ch_bond * np.array([1, cos_theta, sin_theta]) / np.sqrt(3) + c2_pos
    h2_c2 = ch_bond * np.array([1, -cos_theta, -sin_theta]) / np.sqrt(3) + c2_pos
    h3_c2 = ch_bond * np.array([-1, cos_theta, -sin_theta]) / np.sqrt(3) + c2_pos
    h4_c2 = ch_bond * np.array([-1, -cos_theta, sin_theta]) / np.sqrt(3) + c2_pos
    
    # Add C1 hydrogens
    h1_c1 += c1_pos
    h2_c1 += c1_pos
    h3_c1 += c1_pos
    h4_c1 += c1_pos
    
    # Combine all positions
    positions = np.array([
        c1_pos,  # C1
        h1_c1,   # H1
        h2_c1,   # H2
        h3_c1,   # H3
        h4_c1,   # H4
        c2_pos,  # C2
        h1_c2,   # H5
        h2_c2,   # H6
        h3_c2,   # H7
        h4_c2,   # H8
    ])
    
    symbols = ['C', 'H', 'H', 'H', 'H', 'C', 'H', 'H', 'H', 'H']
    
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.info['charge'] = 0
    atoms.info['spin'] = 1
    
    return atoms


class TestTRICStepValidation:
    """Test TRIC step validation and coordinate transformations."""
    
    def setup_method(self):
        """Set up test."""
        self.atoms = create_eclipsed_ethane()
        self.geometry = Geometry.from_atoms(self.atoms)
        self.internal_coords = InternalCoords(self.geometry)
        self.b_matrix_calc = BMatrixCalculator(self.internal_coords)
        
    def test_small_step_round_trip(self):
        """Test round-trip consistency with small steps."""
        print("Testing small step round-trip consistency:")
        
        # Create a small test step in internal coordinates
        n_internal = len(self.internal_coords)
        small_step = np.random.randn(n_internal) * 0.001  # Very small step
        
        print(f"  Internal step norm: {np.linalg.norm(small_step):.6f}")
        
        # Project to Cartesian coordinates
        cartesian_step = self.b_matrix_calc.project_internal_step(self.geometry, small_step)
        print(f"  Cartesian step norm: {np.linalg.norm(cartesian_step):.6f}")
        
        # Update geometry
        new_positions = self.geometry.positions + cartesian_step
        new_geometry = Geometry(self.geometry.symbols, new_positions)
        
        # Calculate new internal coordinates
        new_internal_coords = InternalCoords(new_geometry)
        new_values = new_internal_coords.eval_geometry(new_geometry)
        old_values = self.internal_coords.eval_geometry(self.geometry)
        
        # Check round-trip consistency
        actual_step = new_values - old_values
        step_error = np.linalg.norm(actual_step - small_step)
        
        print(f"  Expected internal step: {small_step[:5]}...")
        print(f"  Actual internal step: {actual_step[:5]}...")
        print(f"  Round-trip error: {step_error:.6f}")
        
        # Round-trip error should be small for small steps
        assert step_error < 0.01, f"Round-trip error too large: {step_error:.6f}"
        
    def test_gradient_projection_consistency(self):
        """Test gradient projection consistency."""
        print("Testing gradient projection consistency:")
        
        # Create test Cartesian forces
        n_atoms = len(self.geometry)
        cartesian_forces = np.random.randn(n_atoms, 3) * 0.1
        
        print(f"  Cartesian forces norm: {np.linalg.norm(cartesian_forces):.6f}")
        
        # Project to internal coordinates
        internal_forces = self.b_matrix_calc.project_cartesian_forces(self.geometry, cartesian_forces)
        print(f"  Internal forces norm: {np.linalg.norm(internal_forces):.6f}")
        
        # Check for NaN or Inf
        assert not np.any(np.isnan(internal_forces)), "Internal forces contain NaN"
        assert not np.any(np.isinf(internal_forces)), "Internal forces contain Inf"
        
        # Test the relationship: internal_forces = G^(-1) @ B @ cartesian_forces
        B = self.b_matrix_calc.calculate_B_matrix(self.geometry)
        G = B @ B.T
        
        # Use pseudo-inverse for singular G
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            G_inv = np.linalg.pinv(G)
        
        cartesian_forces_flat = cartesian_forces.flatten()
        expected_internal_forces = G_inv @ B @ cartesian_forces_flat
        
        projection_error = np.linalg.norm(internal_forces - expected_internal_forces)
        print(f"  Projection consistency error: {projection_error:.6f}")
        
        assert projection_error < 1e-10, f"Projection inconsistency: {projection_error:.6f}"
        
    def test_step_size_preservation(self):
        """Test that step sizes are preserved appropriately."""
        print("Testing step size preservation:")
        
        # Test different step sizes
        step_sizes = [0.001, 0.01, 0.1]
        
        for step_size in step_sizes:
            print(f"  Testing step size: {step_size}")
            
            # Create step in internal coordinates
            n_internal = len(self.internal_coords)
            direction = np.random.randn(n_internal)
            direction = direction / np.linalg.norm(direction)  # Normalize
            internal_step = direction * step_size
            
            # Project to Cartesian
            cartesian_step = self.b_matrix_calc.project_internal_step(self.geometry, internal_step)
            cartesian_norm = np.linalg.norm(cartesian_step)
            
            print(f"    Internal step norm: {np.linalg.norm(internal_step):.6f}")
            print(f"    Cartesian step norm: {cartesian_norm:.6f}")
            print(f"    Ratio: {cartesian_norm / np.linalg.norm(internal_step):.6f}")
            
            # Check that Cartesian step is reasonable
            assert cartesian_norm > 0, "Cartesian step should have non-zero norm"
            assert cartesian_norm < 10.0, f"Cartesian step too large: {cartesian_norm:.6f}"
            
            # Check for NaN or Inf
            assert not np.any(np.isnan(cartesian_step)), "Cartesian step contains NaN"
            assert not np.any(np.isinf(cartesian_step)), "Cartesian step contains Inf"
            
    def test_energy_improvement_direction(self):
        """Test that forces point in the right direction for energy improvement."""
        print("Testing energy improvement direction:")
        
        # This test requires a real calculator, so we'll create a simple test
        # Create a mock force that should decrease energy
        n_atoms = len(self.geometry)
        
        # Create forces that point toward a more compact geometry
        # (this is a simplified test - in reality we'd use a real calculator)
        center = np.mean(self.geometry.positions, axis=0)
        cartesian_forces = center - self.geometry.positions
        cartesian_forces = cartesian_forces * 0.1  # Scale down
        
        print(f"  Cartesian forces norm: {np.linalg.norm(cartesian_forces):.6f}")
        
        # Project to internal coordinates
        internal_forces = self.b_matrix_calc.project_cartesian_forces(self.geometry, cartesian_forces)
        
        print(f"  Internal forces norm: {np.linalg.norm(internal_forces):.6f}")
        print(f"  Internal forces: {internal_forces[:5]}...")
        
        # Check that internal forces are reasonable
        assert not np.any(np.isnan(internal_forces)), "Internal forces contain NaN"
        assert not np.any(np.isinf(internal_forces)), "Internal forces contain Inf"
        
        # Internal forces should have finite magnitude
        assert np.linalg.norm(internal_forces) > 0, "Internal forces should be non-zero"
        assert np.linalg.norm(internal_forces) < 100.0, "Internal forces too large"
        
    def test_optimization_step_simulation(self):
        """Simulate a single optimization step to check for issues."""
        print("Simulating optimization step:")
        
        # Create test forces (simulating a real calculator)
        n_atoms = len(self.geometry)
        cartesian_forces = np.random.randn(n_atoms, 3) * 0.5  # Moderate forces
        
        print(f"  Initial Cartesian forces norm: {np.linalg.norm(cartesian_forces):.6f}")
        
        # Project to internal coordinates
        internal_forces = self.b_matrix_calc.project_cartesian_forces(self.geometry, cartesian_forces)
        print(f"  Internal forces norm: {np.linalg.norm(internal_forces):.6f}")
        
        # Calculate G-matrix and its inverse
        B = self.b_matrix_calc.calculate_B_matrix(self.geometry)
        G = B @ B.T
        
        try:
            G_inv = np.linalg.inv(G)
            print("  Using regular inverse")
        except np.linalg.LinAlgError:
            G_inv = np.linalg.pinv(G)
            print("  Using pseudo-inverse")
        
        # Calculate step in internal coordinates
        internal_step = G_inv @ internal_forces
        print(f"  Internal step norm: {np.linalg.norm(internal_step):.6f}")
        
        # Project to Cartesian coordinates
        cartesian_step = self.b_matrix_calc.project_internal_step(self.geometry, internal_step)
        print(f"  Cartesian step norm: {np.linalg.norm(cartesian_step):.6f}")
        
        # Check for numerical issues
        assert not np.any(np.isnan(cartesian_step)), "Cartesian step contains NaN"
        assert not np.any(np.isinf(cartesian_step)), "Cartesian step contains Inf"
        
        # Update geometry
        new_positions = self.geometry.positions + cartesian_step
        new_geometry = Geometry(self.geometry.symbols, new_positions)
        
        # Check that new geometry is reasonable
        position_changes = np.abs(new_positions - self.geometry.positions)
        max_change = np.max(position_changes)
        
        print(f"  Maximum position change: {max_change:.6f} Å")
        
        # Position changes should be reasonable
        assert max_change > 0, "Should have some position change"
        assert max_change < 1.0, f"Position change too large: {max_change:.6f} Å"
        
        # Check that new geometry doesn't have overlapping atoms
        distances = []
        for i in range(len(new_positions)):
            for j in range(i+1, len(new_positions)):
                dist = np.linalg.norm(new_positions[i] - new_positions[j])
                distances.append(dist)
        
        min_distance = min(distances)
        print(f"  Minimum interatomic distance: {min_distance:.6f} Å")
        
        # Minimum distance should be reasonable (not too small)
        assert min_distance > 0.5, f"Atoms too close: {min_distance:.6f} Å"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
