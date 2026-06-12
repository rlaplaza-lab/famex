"""Tests for molecular geometry property utilities."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from famex.analysis.molecular_properties import determine_degrees_of_freedom, is_linear_molecule


class TestIsLinearMolecule:
    """Tests for is_linear_molecule function."""

    def test_single_atom(self):
        """Test that single atom is considered linear."""
        atoms = Atoms("H", positions=[[0, 0, 0]])
        assert is_linear_molecule(atoms, [0]) is True

    def test_two_atoms(self):
        """Test that two atoms are always linear."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        assert is_linear_molecule(atoms, [0, 1]) is True

    def test_three_atoms_collinear(self):
        """Test that collinear three-atom system is linear."""
        # CO2-like: O-C-O collinear
        atoms = Atoms("OCO", positions=[[-1.16, 0, 0], [0, 0, 0], [1.16, 0, 0]])
        assert is_linear_molecule(atoms, [0, 1, 2]) is True

    def test_three_atoms_non_collinear(self):
        """Test that non-collinear three-atom system is not linear."""
        # H2O-like: bent
        atoms = Atoms("OHH", positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]])
        assert is_linear_molecule(atoms, [0, 1, 2]) is False

    def test_linear_multi_atom(self):
        """Test linear multi-atom molecule using inertia method."""
        # HCN-like: H-C-N linear
        atoms = Atoms(
            "HCN",
            positions=[[-1.06, 0, 0], [0, 0, 0], [1.15, 0, 0]],
        )
        assert is_linear_molecule(atoms, [0, 1, 2]) is True

    def test_non_linear_multi_atom(self):
        """Test non-linear multi-atom molecule."""
        # CH4-like: tetrahedral
        atoms = Atoms(
            "CHHHH",
            positions=[
                [0.0, 0.0, 0.0],
                [1.09, 0.0, 0.0],
                [-0.36, 1.03, 0.0],
                [-0.36, -0.51, 0.89],
                [-0.36, -0.51, -0.89],
            ],
        )
        assert is_linear_molecule(atoms, [0, 1, 2, 3, 4]) is False

    def test_almost_linear_but_not(self):
        """Test molecule that is almost but not quite linear."""
        # Slightly bent triatomic
        atoms = Atoms(
            "OCO",
            positions=[[-1.16, 0, 0], [0, 0, 0], [1.16, 0.1, 0]],
        )
        assert is_linear_molecule(atoms, [0, 1, 2]) is False

    def test_subset_indices(self):
        """Test that only specified indices are considered."""
        # Full molecule is non-linear, but subset is linear
        atoms = Atoms(
            "H2O",
            positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]],
        )
        # Just H-H bond (indices 1 and 2) should be linear
        assert is_linear_molecule(atoms, [1, 2]) is True
        # All three atoms should not be linear
        assert is_linear_molecule(atoms, [0, 1, 2]) is False


class TestDetermineDegreesOfFreedom:
    """Tests for determine_degrees_of_freedom function."""

    def test_single_atom(self):
        """Test that single atom has 3 DOF (translation only)."""
        atoms = Atoms("H", positions=[[0, 0, 0]])
        assert determine_degrees_of_freedom(atoms, [0]) == 3

    def test_two_atoms(self):
        """Test that two atoms have 5 DOF (3 translation + 2 rotation)."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        assert determine_degrees_of_freedom(atoms, [0, 1]) == 5

    def test_linear_molecule(self):
        """Test that linear molecule has 5 DOF (3 translation + 2 rotation)."""
        # CO2
        atoms = Atoms("OCO", positions=[[-1.16, 0, 0], [0, 0, 0], [1.16, 0, 0]])
        assert determine_degrees_of_freedom(atoms, [0, 1, 2]) == 5

    def test_linear_hcn(self):
        """Test HCN (linear) has 5 DOF."""
        atoms = Atoms(
            "HCN",
            positions=[[-1.06, 0, 0], [0, 0, 0], [1.15, 0, 0]],
        )
        assert determine_degrees_of_freedom(atoms, [0, 1, 2]) == 5

    def test_non_linear_molecule(self):
        """Test that non-linear molecule has 6 DOF (3 translation + 3 rotation)."""
        # H2O
        atoms = Atoms("OHH", positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]])
        assert determine_degrees_of_freedom(atoms, [0, 1, 2]) == 6

    def test_non_linear_methane(self):
        """Test CH4 (non-linear) has 6 DOF."""
        atoms = Atoms(
            "CHHHH",
            positions=[
                [0.0, 0.0, 0.0],
                [1.09, 0.0, 0.0],
                [-0.36, 1.03, 0.0],
                [-0.36, -0.51, 0.89],
                [-0.36, -0.51, -0.89],
            ],
        )
        assert determine_degrees_of_freedom(atoms, [0, 1, 2, 3, 4]) == 6

    def test_subset_indices(self):
        """Test that DOF is determined only for specified indices."""
        # Full molecule is non-linear, but subset is linear
        atoms = Atoms(
            "H2O",
            positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]],
        )
        # Just H-H bond should be 5 DOF (linear)
        assert determine_degrees_of_freedom(atoms, [1, 2]) == 5
        # All three atoms should be 6 DOF (non-linear)
        assert determine_degrees_of_freedom(atoms, [0, 1, 2]) == 6

    def test_large_linear_molecule(self):
        """Test large linear molecule still has 5 DOF."""
        # Long chain of atoms in a line
        positions = np.array([[i * 1.0, 0, 0] for i in range(10)])
        atoms = Atoms("H" * 10, positions=positions)
        assert determine_degrees_of_freedom(atoms, list(range(10))) == 5

    def test_large_non_linear_molecule(self):
        """Test large non-linear molecule has 6 DOF."""
        # Create a non-linear structure
        positions = np.array(
            [[i * 1.0, 0, 0] if i % 2 == 0 else [i * 1.0, 0.5, 0] for i in range(10)],
        )
        atoms = Atoms("H" * 10, positions=positions)
        assert determine_degrees_of_freedom(atoms, list(range(10))) == 6
