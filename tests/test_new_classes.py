"""
Test the new Geometry, Reaction, and MLPCalculator classes.
"""

import pytest
import numpy as np
from qme import Geometry, Reaction, MLPCalculator


class TestNewClasses:
    """Test suite for new QME classes."""
    
    def test_geometry_creation(self):
        """Test Geometry class creation."""
        atoms = ["H", "H"]
        coords = np.array([0.0, 0.0, 0.0, 0.74, 0.0, 0.0])
        
        geom = Geometry(atoms, coords, charge=0, mult=1)
        
        assert len(geom) == 2
        assert geom.symbols == ["H", "H"]
        assert geom.charge == 0
        assert geom.mult == 1
        assert geom.coords3d.shape == (2, 3)
    
    def test_reaction_creation(self):
        """Test Reaction class creation."""
        atoms = ["H", "H"]
        reactant_coords = np.array([0.0, 0.0, 0.0, 0.74, 0.0, 0.0])
        product_coords = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0])
        
        reactant = Geometry(atoms, reactant_coords, charge=0, mult=1)
        product = Geometry(atoms, product_coords, charge=0, mult=3)
        
        reaction = Reaction(reactant, product, name="test_reaction")
        
        assert reaction.name == "test_reaction"
        assert len(reaction.reactant) == 2
        assert len(reaction.product) == 2
    
    def test_linear_interpolation(self):
        """Test linear interpolation."""
        atoms = ["H", "H"]
        reactant_coords = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        product_coords = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0])
        
        reactant = Geometry(atoms, reactant_coords)
        product = Geometry(atoms, product_coords)
        reaction = Reaction(reactant, product)
        
        path = reaction.interpolate(npoints=5, method="linear")
        
        assert len(path) == 5
        
        # Check distances along path
        distances = []
        for geom in path:
            dist = np.linalg.norm(geom.coords3d[0] - geom.coords3d[1])
            distances.append(dist)
        
        # Should be linearly spaced
        expected = np.linspace(1.0, 3.0, 5)
        np.testing.assert_allclose(distances, expected, rtol=1e-10)
    
    def test_geodesic_interpolation(self):
        """Test geodesic interpolation."""
        atoms = ["H", "H"]
        reactant_coords = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        product_coords = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0])
        
        reactant = Geometry(atoms, reactant_coords)
        product = Geometry(atoms, product_coords)
        reaction = Reaction(reactant, product)
        
        path = reaction.interpolate(npoints=5, method="geodesic")
        
        assert len(path) == 5
        
        # First and last points should match reactant and product
        first_dist = np.linalg.norm(path[0].coords3d[0] - path[0].coords3d[1])
        last_dist = np.linalg.norm(path[-1].coords3d[0] - path[-1].coords3d[1])
        
        assert abs(first_dist - 1.0) < 1e-6
        assert abs(last_dist - 3.0) < 1e-6
    
    def test_transition_state_guess(self):
        """Test transition state guess finding."""
        atoms = ["H", "H"]
        reactant_coords = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        product_coords = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0])
        
        reactant = Geometry(atoms, reactant_coords)
        product = Geometry(atoms, product_coords)
        reaction = Reaction(reactant, product)
        
        ts_guess = reaction.find_transition_state_guess(npoints=9, method="geodesic")
        
        # TS guess should be roughly in the middle
        dist = np.linalg.norm(ts_guess.coords3d[0] - ts_guess.coords3d[1])
        assert 1.8 < dist < 2.2  # Should be around 2.0
    
    def test_mlp_calculator(self):
        """Test MLPCalculator creation."""
        calc = MLPCalculator(model_type="mock")
        
        assert calc.backend == "so3lr"  # mock uses so3lr backend
        assert "Calculator" in calc.name
        
        # Test calculation
        atoms = ["H", "H"]
        coords = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        geom = Geometry(atoms, coords)
        
        calc.calculate(geom)
        
        assert geom.energy is not None
        assert geom.forces is not None
    
    def test_rmsd_analysis(self):
        """Test RMSD analysis."""
        atoms = ["H", "H"]
        reactant_coords = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        product_coords = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0])
        
        reactant = Geometry(atoms, reactant_coords)
        product = Geometry(atoms, product_coords)
        reaction = Reaction(reactant, product)
        
        path = reaction.interpolate(npoints=5, method="linear")
        rmsd_r, rmsd_p = reaction.get_rmsd_profile(path)
        
        assert len(rmsd_r) == 5
        assert len(rmsd_p) == 5
        
        # First point should have zero RMSD from reactant
        assert rmsd_r[0] < 1e-10
        # Last point should have zero RMSD from product  
        assert rmsd_p[-1] < 1e-10
    
    def test_xyz_trajectory_export(self):
        """Test XYZ trajectory export."""
        atoms = ["H", "H"]
        reactant_coords = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        product_coords = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0])
        
        reactant = Geometry(atoms, reactant_coords)
        product = Geometry(atoms, product_coords)
        reaction = Reaction(reactant, product)
        
        path = reaction.interpolate(npoints=3, method="linear")
        xyz_traj = reaction.to_xyz_trajectory(path)
        
        # Should have proper XYZ format
        lines = xyz_traj.split('\n')
        
        # Check structure: num_atoms, comment, coords, repeat...
        assert lines[0] == "2"  # Number of atoms
        assert "Frame 0" in lines[1]  # Comment
        assert "H" in lines[2]  # First atom
        assert "H" in lines[3]  # Second atom
        assert lines[4] == "2"  # Second frame