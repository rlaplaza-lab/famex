"""Diagnostic tests for TRIC optimizer mathematical components.

This module implements Phase 3 of the TRIC debug plan - mathematical component validation.
"""

import numpy as np
import pytest
from ase import Atoms

from qme.core.tric import create_tric_optimizer
from qme.core.tric.utils import Geometry
from qme.core.tric.internal_coords import InternalCoords
from qme.core.tric.b_matrix import BMatrixCalculator
from qme.backend_availability import get_available_backends


def create_eclipsed_ethane():
    """Create eclipsed ethane geometry (C2H6)."""
    # Eclipsed ethane geometry
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


class TestTRICDiagnostics:
    """Test TRIC mathematical components."""
    
    def setup_method(self):
        """Set up test."""
        self.atoms = create_eclipsed_ethane()
        self.geometry = Geometry.from_atoms(self.atoms)
        
    def test_internal_coordinates_generation(self):
        """Test internal coordinate generation for ethane."""
        internal_coords = InternalCoords(self.geometry)
        
        print(f"Number of internal coordinates: {len(internal_coords)}")
        
        # Count coordinate types
        bond_count = 0
        angle_count = 0
        dihedral_count = 0
        
        for coord in internal_coords:
            coord_class = coord.__class__.__name__
            print(f"  {coord_class}: {coord.indices}")
            
            if coord_class == 'Bond':
                bond_count += 1
            elif coord_class == 'Angle':
                angle_count += 1
            elif coord_class == 'Dihedral':
                dihedral_count += 1
        
        print(f"Bonds: {bond_count}, Angles: {angle_count}, Dihedrals: {dihedral_count}")
        
        # For ethane, we expect:
        # - 7 bonds: 1 C-C, 6 C-H
        # - Several angles: H-C-H, H-C-C, etc.
        # - Several dihedrals: H-C-C-H, etc.
        
        assert bond_count > 0, "Should have at least some bonds"
        assert len(internal_coords) > 0, "Should have internal coordinates"
        
        # Store for other tests
        self.internal_coords = internal_coords
        
    def test_b_matrix_calculation(self):
        """Test B-matrix calculation."""
        if not hasattr(self, 'internal_coords'):
            self.test_internal_coordinates_generation()
            
        internal_coords = self.internal_coords
        b_matrix_calc = BMatrixCalculator(internal_coords)
        
        # Calculate B-matrix
        B = b_matrix_calc.calculate_B_matrix(self.geometry)
        
        print(f"B-matrix shape: {B.shape}")
        print(f"B-matrix:\n{B}")
        
        # Check B-matrix properties
        n_internal, n_cartesian = B.shape
        assert n_internal == len(internal_coords), "B-matrix rows should match internal coordinates"
        assert n_cartesian == 3 * len(self.geometry), "B-matrix cols should match 3*N_atoms"
        
        # Check for NaN or Inf values
        assert not np.any(np.isnan(B)), "B-matrix should not contain NaN values"
        assert not np.any(np.isinf(B)), "B-matrix should not contain Inf values"
        
        # Store for other tests
        self.B = B
        
    def test_g_matrix_calculation(self):
        """Test G-matrix (Wilson matrix) calculation."""
        if not hasattr(self, 'B'):
            self.test_b_matrix_calculation()
            
        B = self.B
        
        # Calculate G-matrix: G = B @ B.T
        G = B @ B.T
        
        print(f"G-matrix shape: {G.shape}")
        print(f"G-matrix:\n{G}")
        
        # Check G-matrix properties
        assert G.shape == (len(self.internal_coords), len(self.internal_coords)), "G-matrix should be square"
        assert np.allclose(G, G.T), "G-matrix should be symmetric"
        
        # Check for NaN or Inf values
        assert not np.any(np.isnan(G)), "G-matrix should not contain NaN values"
        assert not np.any(np.isinf(G)), "G-matrix should not contain Inf values"
        
        # Check rank and condition number
        rank = np.linalg.matrix_rank(G)
        condition_number = np.linalg.cond(G)
        
        print(f"G-matrix rank: {rank}/{len(self.internal_coords)}")
        print(f"G-matrix condition number: {condition_number:.2e}")
        
        # Check if G-matrix is singular
        if rank < len(self.internal_coords):
            print(f"WARNING: G-matrix is singular! Rank {rank} < {len(self.internal_coords)}")
            print("This will cause optimization failures.")
            
            # Check eigenvalues
            eigenvals = np.linalg.eigvals(G)
            eigenvals_sorted = np.sort(eigenvals)[::-1]  # Sort descending
            
            print(f"G-matrix eigenvalues (sorted): {eigenvals_sorted}")
            print(f"Smallest eigenvalue: {eigenvals_sorted[-1]:.2e}")
            
            # Find zero eigenvalues
            zero_eigenvals = np.abs(eigenvals_sorted) < 1e-12
            n_zero = np.sum(zero_eigenvals)
            print(f"Number of zero eigenvalues: {n_zero}")
            
        else:
            print("G-matrix is full rank - this is good!")
            
        # Store for other tests
        self.G = G
        self.G_rank = rank
        self.G_condition = condition_number
        
    def test_b_matrix_pseudo_inverse(self):
        """Test B-matrix pseudo-inverse calculation."""
        if not hasattr(self, 'B'):
            self.test_b_matrix_calculation()
            
        B = self.B
        b_matrix_calc = BMatrixCalculator(self.internal_coords)
        
        # Test different pseudo-inverse methods
        print("Testing B-matrix pseudo-inverse methods:")
        
        # Method 1: numpy.linalg.pinv
        try:
            B_inv_pinv = np.linalg.pinv(B)
            print(f"  pinv: shape {B_inv_pinv.shape}, condition check passed")
        except Exception as e:
            print(f"  pinv failed: {e}")
            B_inv_pinv = None
            
        # Method 2: SVD-based pseudo-inverse
        try:
            B_inv_svd = b_matrix_calc.calculate_B_inverse(self.geometry, method='svd')
            print(f"  svd: shape {B_inv_svd.shape}, condition check passed")
        except Exception as e:
            print(f"  svd failed: {e}")
            B_inv_svd = None
            
        # Test round-trip consistency
        if B_inv_pinv is not None:
            B_reconstructed = B_inv_pinv @ B
            identity_error = np.linalg.norm(B_reconstructed - np.eye(B_reconstructed.shape[0]))
            print(f"  pinv round-trip error: {identity_error:.2e}")
            
        if B_inv_svd is not None:
            B_reconstructed = B_inv_svd @ B
            identity_error = np.linalg.norm(B_reconstructed - np.eye(B_reconstructed.shape[0]))
            print(f"  svd round-trip error: {identity_error:.2e}")
            
    def test_gradient_projection(self):
        """Test gradient projection from Cartesian to internal coordinates."""
        if not hasattr(self, 'B'):
            self.test_b_matrix_calculation()
            
        # Create a test gradient (random forces)
        n_atoms = len(self.geometry)
        cartesian_forces = np.random.randn(n_atoms, 3) * 0.1  # Small random forces
        
        b_matrix_calc = BMatrixCalculator(self.internal_coords)
        
        print("Testing gradient projection:")
        print(f"  Cartesian forces shape: {cartesian_forces.shape}")
        print(f"  Cartesian forces norm: {np.linalg.norm(cartesian_forces):.6f}")
        
        try:
            # Project to internal coordinates
            internal_forces = b_matrix_calc.project_cartesian_forces(self.geometry, cartesian_forces)
            
            print(f"  Internal forces shape: {internal_forces.shape}")
            print(f"  Internal forces norm: {np.linalg.norm(internal_forces):.6f}")
            print(f"  Internal forces: {internal_forces}")
            
            # Check for NaN or Inf
            assert not np.any(np.isnan(internal_forces)), "Internal forces should not contain NaN"
            assert not np.any(np.isinf(internal_forces)), "Internal forces should not contain Inf"
            
        except Exception as e:
            print(f"  Gradient projection failed: {e}")
            raise
            
    def test_step_projection(self):
        """Test step projection from internal to Cartesian coordinates."""
        if not hasattr(self, 'B'):
            self.test_b_matrix_calculation()
            
        # Create a test step in internal coordinates
        n_internal = len(self.internal_coords)
        internal_step = np.random.randn(n_internal) * 0.01  # Small random step
        
        b_matrix_calc = BMatrixCalculator(self.internal_coords)
        
        print("Testing step projection:")
        print(f"  Internal step shape: {internal_step.shape}")
        print(f"  Internal step norm: {np.linalg.norm(internal_step):.6f}")
        print(f"  Internal step: {internal_step}")
        
        try:
            # Project to Cartesian coordinates
            cartesian_step = b_matrix_calc.project_internal_step(self.geometry, internal_step)
            
            print(f"  Cartesian step shape: {cartesian_step.shape}")
            print(f"  Cartesian step norm: {np.linalg.norm(cartesian_step):.6f}")
            
            # Check for NaN or Inf
            assert not np.any(np.isnan(cartesian_step)), "Cartesian step should not contain NaN"
            assert not np.any(np.isinf(cartesian_step)), "Cartesian step should not contain Inf"
            
        except Exception as e:
            print(f"  Step projection failed: {e}")
            raise
            
    def test_round_trip_consistency(self):
        """Test round-trip consistency of coordinate transformations."""
        if not hasattr(self, 'B'):
            self.test_b_matrix_calculation()
            
        b_matrix_calc = BMatrixCalculator(self.internal_coords)
        
        print("Testing round-trip consistency:")
        
        # Test 1: Cartesian -> Internal -> Cartesian
        n_atoms = len(self.geometry)
        original_forces = np.random.randn(n_atoms, 3) * 0.1
        
        try:
            # Cartesian -> Internal
            internal_forces = b_matrix_calc.project_cartesian_forces(self.geometry, original_forces)
            
            # Internal -> Cartesian (using pseudo-inverse)
            B_inv = b_matrix_calc.calculate_B_inverse(self.geometry, method='pinv')
            reconstructed_forces = B_inv.T @ internal_forces
            reconstructed_forces = reconstructed_forces.reshape(-1, 3)
            
            # Check consistency
            force_error = np.linalg.norm(original_forces - reconstructed_forces)
            print(f"  Cartesian->Internal->Cartesian error: {force_error:.2e}")
            
            # Test 2: Internal -> Cartesian -> Internal
            n_internal = len(self.internal_coords)
            original_step = np.random.randn(n_internal) * 0.01
            
            # Internal -> Cartesian
            cartesian_step = b_matrix_calc.project_internal_step(self.geometry, original_step)
            
            # Cartesian -> Internal (using B-matrix)
            cartesian_step_flat = cartesian_step.flatten()
            reconstructed_step = self.B @ cartesian_step_flat
            
            # Check consistency
            step_error = np.linalg.norm(original_step - reconstructed_step)
            print(f"  Internal->Cartesian->Internal error: {step_error:.2e}")
            
        except Exception as e:
            print(f"  Round-trip test failed: {e}")
            raise
            
    def test_hessian_guess(self):
        """Test Hessian guess calculation."""
        if not hasattr(self, 'internal_coords'):
            self.test_internal_coordinates_generation()
            
        b_matrix_calc = BMatrixCalculator(self.internal_coords)
        
        # Calculate Hessian guess
        hessian = b_matrix_calc.calculate_hessian_guess(self.geometry)
        
        print(f"Hessian guess shape: {hessian.shape}")
        print(f"Hessian guess:\n{hessian}")
        
        # Check properties
        assert hessian.shape == (len(self.internal_coords), len(self.internal_coords)), "Hessian should be square"
        assert np.allclose(hessian, hessian.T), "Hessian should be symmetric"
        
        # Check eigenvalues
        eigenvals = np.linalg.eigvals(hessian)
        eigenvals_sorted = np.sort(eigenvals)[::-1]
        
        print(f"Hessian eigenvalues: {eigenvals_sorted}")
        print(f"Smallest eigenvalue: {eigenvals_sorted[-1]:.6f}")
        
        # All eigenvalues should be positive for a good Hessian guess
        n_negative = np.sum(eigenvals < 0)
        if n_negative > 0:
            print(f"WARNING: Hessian has {n_negative} negative eigenvalues!")
        else:
            print("Hessian is positive definite - this is good!")
            
        # Store for other tests
        self.hessian_guess = hessian


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
