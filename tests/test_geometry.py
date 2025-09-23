"""Test geometry handling functionality."""

import pytest
import numpy as np
from qme import Geometry


class TestGeometry:
    """Test suite for Geometry class."""
    
    @pytest.fixture
    def water_molecule(self):
        """Water molecule geometry."""
        atoms = ["O", "H", "H"]
        coords = np.array([
            0.0, 0.0, 0.0,      # O
            0.96, 0.0, 0.0,     # H
            -0.24, 0.93, 0.0,   # H
        ])
        return Geometry(atoms=atoms, coords=coords, charge=0, mult=1)
    
    @pytest.fixture
    def methane_molecule(self):
        """Methane molecule geometry."""
        atoms = ["C", "H", "H", "H", "H"]
        coords = np.array([
            0.0, 0.0, 0.0,      # C
            1.09, 0.0, 0.0,     # H
            -0.36, 1.03, 0.0,   # H  
            -0.36, -0.51, 0.89, # H
            -0.36, -0.51, -0.89, # H
        ])
        return Geometry(atoms=atoms, coords=coords, charge=0, mult=1, energy=-40.0)
    
    def test_geometry_creation(self, water_molecule):
        """Test basic geometry creation."""
        assert water_molecule.natoms == 3
        assert len(water_molecule.atoms) == 3
        assert len(water_molecule.coords) == 9  # 3 * 3
        assert water_molecule.charge == 0
        assert water_molecule.mult == 1
    
    def test_coords3d_property(self, water_molecule):
        """Test 3D coordinates property."""
        coords3d = water_molecule.coords3d
        assert coords3d.shape == (3, 3)
        assert np.allclose(coords3d[0], [0.0, 0.0, 0.0])
        assert np.allclose(coords3d[1], [0.96, 0.0, 0.0])
        assert np.allclose(coords3d[2], [-0.24, 0.93, 0.0])
    
    def test_coords3d_setter(self, water_molecule):
        """Test setting coordinates via 3D array."""
        new_coords = np.array([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0], 
            [1.5, 1.0, 0.0]
        ])
        
        water_molecule.coords3d = new_coords
        
        assert np.allclose(water_molecule.coords3d, new_coords)
        assert len(water_molecule.coords) == 9
    
    def test_geometry_copy(self, methane_molecule):
        """Test geometry copying."""
        copied = methane_molecule.copy()
        
        assert copied.natoms == methane_molecule.natoms
        assert copied.atoms == methane_molecule.atoms
        assert np.allclose(copied.coords, methane_molecule.coords)
        assert copied.charge == methane_molecule.charge
        assert copied.mult == methane_molecule.mult
        assert copied.energy == methane_molecule.energy
        
        # Modifying copy shouldn't affect original
        copied.coords[0] += 1.0
        assert not np.allclose(copied.coords, methane_molecule.coords)
    
    def test_center_of_mass(self, water_molecule):
        """Test center of mass calculation."""
        com = water_molecule.center_of_mass()
        
        assert len(com) == 3
        # COM should be close to oxygen (heaviest atom)
        assert np.linalg.norm(com - water_molecule.coords3d[0]) < 0.5
    
    def test_rmsd_calculation(self, water_molecule):
        """Test RMSD calculation between geometries."""
        # Create slightly displaced geometry
        displaced = water_molecule.copy()
        displaced.coords += np.random.normal(0, 0.1, size=displaced.coords.shape)
        
        rmsd = water_molecule.rmsd(displaced)
        
        assert rmsd > 0.0
        assert rmsd < 1.0  # Should be small displacement
        
        # RMSD with self should be zero
        assert water_molecule.rmsd(water_molecule) == 0.0
    
    def test_xyz_format(self, methane_molecule):
        """Test XYZ format generation."""
        xyz_str = methane_molecule.as_xyz()
        
        lines = xyz_str.strip().split('\n')
        assert lines[0] == "5"  # Number of atoms
        assert "Energy:" in lines[1]  # Comment with energy
        
        # Check atom lines
        for i, atom in enumerate(methane_molecule.atoms):
            atom_line = lines[i + 2]
            assert atom_line.startswith(atom)
    
    def test_from_xyz(self):
        """Test creating geometry from XYZ string."""
        xyz_str = """3
Water molecule
O    0.000000    0.000000    0.000000
H    0.960000    0.000000    0.000000  
H   -0.240000    0.930000    0.000000"""
        
        geom = Geometry.from_xyz(xyz_str, charge=0, mult=1)
        
        assert geom.natoms == 3
        assert geom.atoms == ["O", "H", "H"]
        assert np.allclose(geom.coords3d[0], [0.0, 0.0, 0.0])
        assert np.allclose(geom.coords3d[1], [0.96, 0.0, 0.0])
        assert np.allclose(geom.coords3d[2], [-0.24, 0.93, 0.0])
    
    def test_energy_and_forces(self):
        """Test geometry with energy and forces."""
        atoms = ["H", "H"]
        coords = np.array([0.0, 0.0, 0.0, 0.74, 0.0, 0.0])
        energy = -1.1
        forces = np.array([0.1, 0.0, 0.0, -0.1, 0.0, 0.0])
        
        geom = Geometry(atoms=atoms, coords=coords, energy=energy, forces=forces)
        
        assert geom.energy == energy
        assert np.allclose(geom.forces, forces)
    
    def test_invalid_dimensions(self):
        """Test error handling for invalid dimensions."""
        atoms = ["H", "H"]
        
        # Wrong number of coordinates
        with pytest.raises(ValueError, match="Coordinates array must have 3\\*N elements"):
            coords = np.array([0.0, 0.0])  # Should be 6 elements for 2 atoms
            Geometry(atoms=atoms, coords=coords)
        
        # Wrong forces dimensions  
        with pytest.raises(ValueError, match="Forces array must match coordinates dimensions"):
            coords = np.array([0.0, 0.0, 0.0, 0.74, 0.0, 0.0])
            forces = np.array([0.1, 0.0, 0.0])  # Should be 6 elements
            Geometry(atoms=atoms, coords=coords, forces=forces)
    
    def test_atomic_masses(self, water_molecule):
        """Test atomic mass lookup."""
        # Test that atomic masses are reasonable
        h_mass = water_molecule._atomic_mass("H")
        o_mass = water_molecule._atomic_mass("O")
        
        assert h_mass == 1.008
        assert o_mass == 15.999
        assert o_mass > h_mass
        
        # Unknown element should default to 1.0
        unknown_mass = water_molecule._atomic_mass("X")
        assert unknown_mass == 1.0
    
    def test_geometry_repr(self, methane_molecule):
        """Test string representation."""
        repr_str = repr(methane_molecule)
        
        assert "Geometry" in repr_str
        assert "natoms=5" in repr_str
        assert "energy=-40.0" in repr_str