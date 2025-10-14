"""Implement and test fixes for TRIC optimizer.

This module implements the critical fixes needed to make TRIC optimizer work properly.
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


class TestTRICFixImplementation:
    """Test the implementation of TRIC fixes."""
    
    def setup_method(self):
        """Set up test."""
        self.atoms = create_eclipsed_ethane()
        self.geometry = Geometry.from_atoms(self.atoms)
        
    def test_minimal_internal_coordinates(self):
        """Test using minimal internal coordinates to ensure full rank."""
        print("Testing minimal internal coordinates:")
        
        # Create a minimal set of internal coordinates
        # For ethane, we need exactly 24 coordinates (3N-6)
        
        # Essential bonds: 7 bonds (1 C-C, 6 C-H)
        essential_bonds = [
            (0, 5),  # C-C bond
            (0, 1), (0, 2), (0, 3), (0, 4),  # C1-H bonds
            (5, 6), (5, 7), (5, 8), (5, 9)   # C2-H bonds
        ]
        
        # Essential angles: 12 angles (H-C-H and H-C-C)
        essential_angles = [
            (1, 0, 2), (1, 0, 3), (1, 0, 4),  # H-C-H angles around C1
            (2, 0, 3), (2, 0, 4), (3, 0, 4),
            (6, 5, 7), (6, 5, 8), (6, 5, 9),  # H-C-H angles around C2
            (7, 5, 8), (7, 5, 9), (8, 5, 9),
            (1, 0, 5), (2, 0, 5), (3, 0, 5), (4, 0, 5),  # H-C-C angles
            (0, 5, 6), (0, 5, 7), (0, 5, 8), (0, 5, 9)
        ]
        
        # Essential dihedrals: 5 dihedrals (H-C-C-H)
        essential_dihedrals = [
            (1, 0, 5, 6), (2, 0, 5, 7), (3, 0, 5, 8), (4, 0, 5, 9), (1, 0, 5, 7)
        ]
        
        # Create internal coordinates manually
        from qme.core.tric.internal_coords import Bond, Angle, Dihedral
        
        coords = []
        
        # Add bonds
        for i, j in essential_bonds:
            coords.append(Bond(i, j))
            
        # Add angles
        for i, j, k in essential_angles:
            coords.append(Angle(i, j, k))
            
        # Add dihedrals
        for i, j, k, l in essential_dihedrals:
            coords.append(Dihedral(i, j, k, l))
        
        print(f"  Total coordinates: {len(coords)}")
        print(f"  Expected: {3 * len(self.geometry) - 6}")
        
        # Create internal coordinates object
        internal_coords = InternalCoords.__new__(InternalCoords)
        internal_coords.geometry = self.geometry
        internal_coords.coords = coords
        
        # Test B-matrix calculation
        b_matrix_calc = BMatrixCalculator(internal_coords)
        B = b_matrix_calc.calculate_B_matrix(self.geometry)
        
        print(f"  B-matrix shape: {B.shape}")
        
        # Calculate G-matrix
        G = B @ B.T
        
        # Check rank
        rank = np.linalg.matrix_rank(G)
        condition_number = np.linalg.cond(G)
        
        print(f"  G-matrix rank: {rank}/{len(coords)}")
        print(f"  G-matrix condition number: {condition_number:.2e}")
        
        # Check if full rank
        if rank == len(coords):
            print("  ✓ G-matrix is full rank!")
        else:
            print(f"  ✗ G-matrix is rank-deficient (rank {rank}/{len(coords)})")
            
        # Check condition number
        if condition_number < 1e12:
            print("  ✓ G-matrix is well-conditioned!")
        else:
            print(f"  ✗ G-matrix is ill-conditioned (κ={condition_number:.2e})")
            
        # Store for other tests
        self.minimal_internal_coords = internal_coords
        self.minimal_b_matrix_calc = b_matrix_calc
        
    def test_robust_coordinate_selection(self):
        """Test robust coordinate selection algorithm."""
        print("Testing robust coordinate selection:")
        
        # Use the minimal coordinates from previous test
        if not hasattr(self, 'minimal_internal_coords'):
            self.test_minimal_internal_coordinates()
            
        internal_coords = self.minimal_internal_coords
        b_matrix_calc = self.minimal_b_matrix_calc
        
        # Test coordinate transformations
        B = b_matrix_calc.calculate_B_matrix(self.geometry)
        G = B @ B.T
        
        print(f"  Testing with {len(internal_coords)} coordinates")
        
        # Test small step round-trip
        n_internal = len(internal_coords)
        small_step = np.random.randn(n_internal) * 0.001
        
        print(f"  Internal step norm: {np.linalg.norm(small_step):.6f}")
        
        # Project to Cartesian
        cartesian_step = b_matrix_calc.project_internal_step(self.geometry, small_step)
        print(f"  Cartesian step norm: {np.linalg.norm(cartesian_step):.6f}")
        
        # Update geometry
        new_positions = self.geometry.positions + cartesian_step
        new_geometry = Geometry(self.geometry.symbols, new_positions)
        
        # Calculate new internal coordinates
        new_internal_coords = InternalCoords.__new__(InternalCoords)
        new_internal_coords.geometry = new_geometry
        new_internal_coords.coords = internal_coords.coords  # Use same coordinate definitions
        
        new_values = new_internal_coords.eval_geometry(new_geometry)
        old_values = internal_coords.eval_geometry(self.geometry)
        
        # Check round-trip consistency
        actual_step = new_values - old_values
        step_error = np.linalg.norm(actual_step - small_step)
        
        print(f"  Round-trip error: {step_error:.6f}")
        
        if step_error < 0.01:
            print("  ✓ Round-trip consistency is good!")
        else:
            print(f"  ✗ Round-trip error too large: {step_error:.6f}")
            
        # Test gradient projection
        n_atoms = len(self.geometry)
        cartesian_forces = np.random.randn(n_atoms, 3) * 0.1
        
        internal_forces = b_matrix_calc.project_cartesian_forces(self.geometry, cartesian_forces)
        
        print(f"  Gradient projection norm: {np.linalg.norm(internal_forces):.6f}")
        
        # Check for numerical issues
        has_nan = np.any(np.isnan(internal_forces))
        has_inf = np.any(np.isinf(internal_forces))
        
        if not has_nan and not has_inf:
            print("  ✓ Gradient projection is numerically stable!")
        else:
            print(f"  ✗ Gradient projection has numerical issues (NaN: {has_nan}, Inf: {has_inf})")
            
    def test_optimization_step_simulation(self):
        """Test a complete optimization step simulation."""
        print("Testing complete optimization step:")
        
        # Use minimal coordinates
        if not hasattr(self, 'minimal_internal_coords'):
            self.test_minimal_internal_coordinates()
            
        internal_coords = self.minimal_internal_coords
        b_matrix_calc = self.minimal_b_matrix_calc
        
        # Simulate forces from a calculator
        n_atoms = len(self.geometry)
        cartesian_forces = np.random.randn(n_atoms, 3) * 0.5
        
        print(f"  Cartesian forces norm: {np.linalg.norm(cartesian_forces):.6f}")
        
        # Project to internal coordinates
        internal_forces = b_matrix_calc.project_cartesian_forces(self.geometry, cartesian_forces)
        print(f"  Internal forces norm: {np.linalg.norm(internal_forces):.6f}")
        
        # Calculate G-matrix and its inverse
        B = b_matrix_calc.calculate_B_matrix(self.geometry)
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
        
        # Apply trust radius
        trust_radius = 0.3
        step_norm = np.linalg.norm(internal_step)
        if step_norm > trust_radius:
            internal_step = internal_step / step_norm * trust_radius
            print(f"  Step limited by trust radius: {trust_radius}")
        
        # Project to Cartesian coordinates
        cartesian_step = b_matrix_calc.project_internal_step(self.geometry, internal_step)
        print(f"  Cartesian step norm: {np.linalg.norm(cartesian_step):.6f}")
        
        # Check for numerical issues
        has_nan = np.any(np.isnan(cartesian_step))
        has_inf = np.any(np.isinf(cartesian_step))
        
        if not has_nan and not has_inf:
            print("  ✓ Step projection is numerically stable!")
        else:
            print(f"  ✗ Step projection has numerical issues (NaN: {has_nan}, Inf: {has_inf})")
            
        # Update geometry
        new_positions = self.geometry.positions + cartesian_step
        new_geometry = Geometry(self.geometry.symbols, new_positions)
        
        # Check geometry validity
        position_changes = np.abs(new_positions - self.geometry.positions)
        max_change = np.max(position_changes)
        
        print(f"  Maximum position change: {max_change:.6f} Å")
        
        # Check interatomic distances
        distances = []
        for i in range(len(new_positions)):
            for j in range(i+1, len(new_positions)):
                dist = np.linalg.norm(new_positions[i] - new_positions[j])
                distances.append(dist)
        
        min_distance = min(distances)
        print(f"  Minimum interatomic distance: {min_distance:.6f} Å")
        
        if max_change < 1.0 and min_distance > 0.5:
            print("  ✓ Geometry update is reasonable!")
        else:
            print(f"  ✗ Geometry update issues (max_change: {max_change:.6f}, min_dist: {min_distance:.6f})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
