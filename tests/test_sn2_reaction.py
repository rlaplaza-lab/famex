"""Test SN2 reaction: CH3Cl + OH- → CH3OH + Cl-

This is a classic nucleophilic substitution reaction that demonstrates
the backside attack mechanism typical of SN2 reactions.

Inspired by pysisyphus organic reaction tests.
"""

import numpy as np
import pytest

from qme import Geometry, MLPCalculator, Reaction


class TestSN2Reaction:
    """Test suite for SN2 reaction pathway."""

    @pytest.fixture
    def sn2_reactant(self):
        """Methyl chloride + hydroxide ion reactant complex."""
        # Approximate starting geometry for CH3Cl + OH-
        atoms = ["C", "H", "H", "H", "Cl", "O", "H"]
        coords = np.array(
            [
                # CH3Cl part
                0.0,
                0.0,
                0.0,  # C
                1.09,
                0.0,
                0.0,  # H
                -0.36,
                1.03,
                0.0,  # H
                -0.36,
                -0.51,
                0.89,  # H
                -0.37,
                -0.52,
                -1.76,  # Cl
                # OH- approaching
                3.0,
                0.0,
                0.0,  # O (far away initially)
                3.97,
                0.0,
                0.0,  # H
            ]
        )
        return Geometry(atoms=atoms, coords=coords, charge=-1, mult=1)

    @pytest.fixture
    def sn2_product(self):
        """Methanol + chloride ion product complex."""
        # Approximate final geometry for CH3OH + Cl-
        atoms = ["C", "H", "H", "H", "Cl", "O", "H"]
        coords = np.array(
            [
                # CH3OH part
                0.0,
                0.0,
                0.0,  # C
                1.09,
                0.0,
                0.0,  # H
                -0.36,
                1.03,
                0.0,  # H
                -0.36,
                -0.51,
                0.89,  # H
                -3.0,
                0.0,
                0.0,  # Cl (far away)
                # OH bound to carbon
                -1.43,
                -0.52,
                -0.87,  # O
                -1.8,
                -1.42,
                -0.87,  # H
            ]
        )
        return Geometry(atoms=atoms, coords=coords, charge=-1, mult=1)

    @pytest.fixture
    def sn2_transition_state(self):
        """Transition state with C-Cl breaking and C-O forming."""
        # Transition state geometry with both bonds partially formed/broken
        atoms = ["C", "H", "H", "H", "Cl", "O", "H"]
        coords = np.array(
            [
                # Carbon center
                0.0,
                0.0,
                0.0,  # C
                1.09,
                0.0,
                0.0,  # H
                -0.36,
                1.03,
                0.0,  # H
                -0.36,
                -0.51,
                0.89,  # H
                # Cl partially dissociated
                -0.5,
                -0.7,
                -2.2,  # Cl
                # OH partially associated
                1.2,
                -0.2,
                -0.3,  # O
                1.9,
                -0.7,
                -0.5,  # H
            ]
        )
        return Geometry(atoms=atoms, coords=coords, charge=-1, mult=1)

    def test_sn2_reaction_creation(self, sn2_reactant, sn2_product):
        """Test creation of SN2 reaction object."""
        reaction = Reaction(
            reactant=sn2_reactant, product=sn2_product, name="SN2_CH3Cl_OH"
        )

        assert reaction.name == "SN2_CH3Cl_OH"
        assert reaction.reactant.natoms == 7
        assert reaction.product.natoms == 7
        assert not reaction.has_ts

    def test_sn2_with_transition_state(
        self, sn2_reactant, sn2_product, sn2_transition_state
    ):
        """Test SN2 reaction with transition state."""
        reaction = Reaction(
            reactant=sn2_reactant,
            product=sn2_product,
            ts=sn2_transition_state,
            name="SN2_CH3Cl_OH_with_TS",
        )

        assert reaction.has_ts
        assert reaction.ts.natoms == 7

    def test_sn2_interpolation(self, sn2_reactant, sn2_product):
        """Test linear interpolation along SN2 reaction coordinate."""
        reaction = Reaction(sn2_reactant, sn2_product, name="SN2_interpolation")

        # Create interpolated path
        path_geoms = reaction.interpolate(npoints=11)

        assert len(path_geoms) == 11
        assert all(geom.natoms == 7 for geom in path_geoms)
        assert all(geom.charge == -1 for geom in path_geoms)

        # Check endpoints match original geometries
        assert np.allclose(path_geoms[0].coords, sn2_reactant.coords, atol=1e-10)
        assert np.allclose(path_geoms[-1].coords, sn2_product.coords, atol=1e-10)

        # Check interpolation is monotonic
        c_cl_distances = []
        c_o_distances = []

        for geom in path_geoms:
            coords = geom.coords3d
            # C is atom 0, Cl is atom 4, O is atom 5
            c_cl_dist = np.linalg.norm(coords[0] - coords[4])
            c_o_dist = np.linalg.norm(coords[0] - coords[5])

            c_cl_distances.append(c_cl_dist)
            c_o_distances.append(c_o_dist)

        # C-Cl distance should increase along the path
        assert c_cl_distances[0] < c_cl_distances[-1], "C-Cl distance should increase"
        # C-O distance should decrease along the path
        assert c_o_distances[0] > c_o_distances[-1], "C-O distance should decrease"

    def test_sn2_energy_calculation(self, sn2_reactant, sn2_product):
        """Test energy calculation along SN2 pathway."""
        calculator = MLPCalculator(model_type="mock_ani")

        # Calculate energies for reactant and product
        calculator.calculate(sn2_reactant)
        calculator.calculate(sn2_product)

        assert sn2_reactant.energy is not None
        assert sn2_product.energy is not None
        assert sn2_reactant.forces is not None
        assert sn2_product.forces is not None

        # Create reaction and check energetics
        reaction = Reaction(sn2_reactant, sn2_product, name="SN2_energetics")
        assert reaction.reaction_energy is not None

    def test_sn2_forces_along_path(self, sn2_reactant, sn2_product):
        """Test force calculations along SN2 reaction path."""
        calculator = MLPCalculator(model_type="mock_forces")
        reaction = Reaction(sn2_reactant, sn2_product, name="SN2_forces")

        # Get interpolated path
        path_geoms = reaction.interpolate(npoints=5)

        # Calculate forces for each geometry
        for geom in path_geoms:
            calculator.calculate(geom)
            assert geom.forces is not None
            assert len(geom.forces) == 21  # 3 * 7 atoms
            assert not np.allclose(
                geom.forces, 0.0
            ), "Forces should not be zero along path"

    def test_sn2_rmsd_profile(self, sn2_reactant, sn2_product):
        """Test RMSD profile along SN2 reaction path."""
        reaction = Reaction(sn2_reactant, sn2_product, name="SN2_rmsd")
        path_geoms = reaction.interpolate(npoints=10)

        rmsd_from_reactant, rmsd_from_product = reaction.get_rmsd_profile(path_geoms)

        assert len(rmsd_from_reactant) == 10
        assert len(rmsd_from_product) == 10

        # RMSD from reactant should increase along path
        assert rmsd_from_reactant[0] == 0.0  # First point is reactant
        assert rmsd_from_reactant[-1] > rmsd_from_reactant[0]

        # RMSD from product should decrease along path
        assert rmsd_from_product[-1] == 0.0  # Last point is product
        assert rmsd_from_product[0] > rmsd_from_product[-1]

    def test_sn2_xyz_export(self, sn2_reactant, sn2_product, sn2_transition_state):
        """Test XYZ trajectory export for SN2 reaction."""
        reaction = Reaction(
            sn2_reactant, sn2_product, ts=sn2_transition_state, name="SN2_export"
        )

        # Test default export (reactant, TS, product)
        xyz_traj = reaction.to_xyz_trajectory()

        # Should contain 3 XYZ blocks
        xyz_blocks = xyz_traj.split("\n7\n")[
            1:
        ]  # Split on natoms line, skip first empty
        assert len([block for block in xyz_blocks if block.strip()]) >= 2

        # Test custom geometry list
        path_geoms = reaction.interpolate(npoints=3)
        xyz_custom = reaction.to_xyz_trajectory(path_geoms)
        assert "7" in xyz_custom  # Should contain natoms lines

    def test_sn2_mechanism_validation(self, sn2_reactant, sn2_product):
        """Test validation of SN2 mechanism characteristics."""
        reaction = Reaction(sn2_reactant, sn2_product, name="SN2_mechanism")
        path_geoms = reaction.interpolate(npoints=20)

        # Monitor key bond distances throughout the reaction
        c_cl_distances = []
        c_o_distances = []

        for geom in path_geoms:
            coords = geom.coords3d
            c_cl_dist = np.linalg.norm(coords[0] - coords[4])  # C-Cl
            c_o_dist = np.linalg.norm(coords[0] - coords[5])  # C-O

            c_cl_distances.append(c_cl_dist)
            c_o_distances.append(c_o_dist)

        c_cl_distances = np.array(c_cl_distances)
        c_o_distances = np.array(c_o_distances)

        # SN2 mechanism should show concerted bond breaking/forming
        # C-Cl should monotonically increase
        assert np.all(
            np.diff(c_cl_distances) >= -0.1
        ), "C-Cl distance should generally increase"

        # C-O should monotonically decrease
        assert np.all(
            np.diff(c_o_distances) <= 0.1
        ), "C-O distance should generally decrease"

        # The reaction should show characteristic SN2 inversion
        # (This is a simplified check - real validation would examine stereochemistry)
<<<<<<< HEAD
        assert c_cl_distances[-1] > c_cl_distances[0] + 1.0, "C-Cl should be significantly longer at end"
        assert c_o_distances[0] > c_o_distances[-1] + 1.0, "C-O should be significantly shorter at end"

    def test_sn2_geodesic_interpolation(self, sn2_reactant, sn2_product):
        """Test geodesic interpolation for SN2 reaction."""
        reaction = Reaction(sn2_reactant, sn2_product, name="SN2_geodesic")
        
        # Test basic functionality
        geodesic_path = reaction.interpolate(npoints=10, method="geodesic")
        linear_path = reaction.interpolate(npoints=10, method="linear")
        
        assert len(geodesic_path) == len(linear_path) == 10
        assert all(geom.natoms == 7 for geom in geodesic_path)
        
        # Check endpoints match for both methods
        assert np.allclose(geodesic_path[0].coords, linear_path[0].coords, atol=1e-10)
        assert np.allclose(geodesic_path[-1].coords, linear_path[-1].coords, atol=1e-10)
        
        # Analyze bond distances along geodesic path
        geo_c_cl_distances = []
        geo_c_o_distances = []
        
        for geom in geodesic_path:
            coords = geom.coords3d
            # C is atom 0, Cl is atom 4, O is atom 5
            c_cl_dist = np.linalg.norm(coords[0] - coords[4])
            c_o_dist = np.linalg.norm(coords[0] - coords[5])
            
            geo_c_cl_distances.append(c_cl_dist)
            geo_c_o_distances.append(c_o_dist)
        
        # Geodesic interpolation should also show reasonable reaction coordinate
        # C-Cl distance should increase along the path
        assert geo_c_cl_distances[0] < geo_c_cl_distances[-1], "Geodesic: C-Cl distance should increase"
        # C-O distance should decrease along the path  
        assert geo_c_o_distances[0] > geo_c_o_distances[-1], "Geodesic: C-O distance should decrease"
        
        # Compare with linear interpolation - geodesic should generally preserve
        # other bond lengths better (this is a simplified check)
        geo_bond_variance = self._calculate_bond_length_variance(geodesic_path)
        linear_bond_variance = self._calculate_bond_length_variance(linear_path)
        
        # Geodesic interpolation often (but not always) gives smoother paths
        # This is just a basic sanity check that it produces reasonable results
        assert geo_bond_variance > 0  # Some variance is expected in reaction paths

    def _calculate_bond_length_variance(self, geometries):
        """Helper method to calculate variance in non-reacting bond lengths."""
        # Calculate variance in C-H bond lengths (should remain relatively constant)
        ch_bond_lengths = []
        
        for geom in geometries:
            coords = geom.coords3d
            # C is atom 0, H atoms are 1, 2, 3
            for h_idx in [1, 2, 3]:
                ch_dist = np.linalg.norm(coords[0] - coords[h_idx])
                ch_bond_lengths.append(ch_dist)
        
        return np.var(ch_bond_lengths)
=======
        assert (
            c_cl_distances[-1] > c_cl_distances[0] + 1.0
        ), "C-Cl should be significantly longer at end"
        assert (
            c_o_distances[0] > c_o_distances[-1] + 1.0
        ), "C-O should be significantly shorter at end"
>>>>>>> main
