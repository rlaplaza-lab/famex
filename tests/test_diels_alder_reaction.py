"""Test Diels-Alder reaction: 1,3-butadiene + ethylene → cyclohexene

This is a classic [4+2] cycloaddition reaction that demonstrates
concerted pericyclic reaction mechanisms.

Inspired by pysisyphus Diels-Alder tests and organic reaction benchmarks.
"""

import numpy as np
import pytest

from qme import Geometry, MLPCalculator, Reaction


class TestDielsAlderReaction:
    """Test suite for Diels-Alder cycloaddition reaction."""

    @pytest.fixture
    def diels_alder_reactants(self):
        """1,3-butadiene + ethylene reactant complex."""
        # Butadiene (C4H6) + Ethylene (C2H4)
        atoms = [
            "C",
            "C",
            "C",
            "C",
            "H",
            "H",
            "H",
            "H",
            "H",
            "H",
            "C",
            "C",
            "H",
            "H",
            "H",
            "H",
        ]

        coords = np.array(
            [
                # 1,3-butadiene
                -2.0,
                0.0,
                0.0,  # C1
                -0.7,
                0.6,
                0.0,  # C2
                0.7,
                0.0,
                0.0,  # C3
                2.0,
                0.6,
                0.0,  # C4
                -2.8,
                0.7,
                0.0,  # H
                -2.2,
                -1.1,
                0.0,  # H
                -0.5,
                1.7,
                0.0,  # H
                0.5,
                -1.1,
                0.0,  # H
                2.2,
                1.7,
                0.0,  # H
                2.8,
                0.0,
                0.0,  # H
                # Ethylene (approaching from above)
                0.0,
                0.0,
                3.0,  # C5
                0.0,
                1.3,
                3.0,  # C6
                -0.9,
                -0.5,
                3.0,  # H
                0.9,
                -0.5,
                3.0,  # H
                -0.9,
                1.8,
                3.0,  # H
                0.9,
                1.8,
                3.0,  # H
            ]
        )

        return Geometry(atoms=atoms, coords=coords, charge=0, mult=1)

    @pytest.fixture
    def diels_alder_product(self):
        """Cyclohexene product."""
        # Six-membered ring structure
        atoms = [
            "C",
            "C",
            "C",
            "C",
            "H",
            "H",
            "H",
            "H",
            "H",
            "H",
            "C",
            "C",
            "H",
            "H",
            "H",
            "H",
        ]

        coords = np.array(
            [
                # Cyclohexene ring (chair-like conformation)
                -1.2,
                -0.7,
                0.0,  # C1
                -0.6,
                0.7,
                0.0,  # C2
                0.6,
                0.7,
                0.0,  # C3
                1.2,
                -0.7,
                0.0,  # C4
                -2.0,
                -0.8,
                0.9,  # H
                -1.5,
                -0.9,
                -0.9,  # H
                -1.0,
                1.5,
                0.9,  # H
                1.0,
                1.5,
                0.9,  # H
                1.5,
                -0.9,
                -0.9,  # H
                2.0,
                -0.8,
                0.9,  # H
                # Completing the ring
                0.0,
                -1.4,
                0.0,  # C5 (bridge)
                0.0,
                1.4,
                0.0,  # C6 (double bond)
                -0.9,
                -2.0,
                0.0,  # H
                0.9,
                -2.0,
                0.0,  # H
                -0.9,
                2.0,
                0.0,  # H
                0.9,
                2.0,
                0.0,  # H
            ]
        )

        return Geometry(atoms=atoms, coords=coords, charge=0, mult=1)

    @pytest.fixture
    def diels_alder_ts(self):
        """Transition state with partial bond formation."""
        atoms = [
            "C",
            "C",
            "C",
            "C",
            "H",
            "H",
            "H",
            "H",
            "H",
            "H",
            "C",
            "C",
            "H",
            "H",
            "H",
            "H",
        ]

        coords = np.array(
            [
                # Partially formed ring structure
                -1.5,
                -0.3,
                0.0,  # C1
                -0.8,
                0.8,
                0.0,  # C2
                0.8,
                0.8,
                0.0,  # C3
                1.5,
                -0.3,
                0.0,  # C4
                -2.2,
                -0.5,
                0.9,  # H
                -1.8,
                -0.6,
                -0.9,  # H
                -1.2,
                1.6,
                0.9,  # H
                1.2,
                1.6,
                0.9,  # H
                1.8,
                -0.6,
                -0.9,  # H
                2.2,
                -0.5,
                0.9,  # H
                # Ethylene partially associated
                -0.2,
                -1.0,
                1.5,  # C5 (forming bond to C1)
                0.2,
                -1.0,
                1.5,  # C6 (forming bond to C4)
                -0.7,
                -1.8,
                1.8,  # H
                0.7,
                -1.8,
                1.8,  # H
                -0.7,
                -0.2,
                2.2,  # H
                0.7,
                -0.2,
                2.2,  # H
            ]
        )

        return Geometry(atoms=atoms, coords=coords, charge=0, mult=1)

    def test_diels_alder_reaction_setup(
        self, diels_alder_reactants, diels_alder_product
    ):
        """Test basic Diels-Alder reaction setup."""
        reaction = Reaction(
            reactant=diels_alder_reactants,
            product=diels_alder_product,
            name="Diels_Alder_butadiene_ethylene",
        )

        assert reaction.name == "Diels_Alder_butadiene_ethylene"
        assert reaction.reactant.natoms == 16
        assert reaction.product.natoms == 16
        assert reaction.reactant.charge == 0
        assert reaction.product.charge == 0

    def test_diels_alder_with_ts(
        self, diels_alder_reactants, diels_alder_product, diels_alder_ts
    ):
        """Test Diels-Alder reaction with transition state."""
        reaction = Reaction(
            reactant=diels_alder_reactants,
            product=diels_alder_product,
            ts=diels_alder_ts,
            name="Diels_Alder_with_TS",
        )

        assert reaction.has_ts
        assert reaction.ts.natoms == 16

    def test_diels_alder_bond_formation(
        self, diels_alder_reactants, diels_alder_product
    ):
        """Test bond formation pattern in Diels-Alder reaction."""
        reaction = Reaction(diels_alder_reactants, diels_alder_product, name="DA_bonds")
        path_geoms = reaction.interpolate(npoints=15)

        # Monitor formation of new C-C bonds
        new_bond1_distances = []  # C1-C5 bond formation
        new_bond2_distances = []  # C4-C6 bond formation

        for geom in path_geoms:
            coords = geom.coords3d
            # C1 (index 0) to C5 (index 10)
            bond1_dist = np.linalg.norm(coords[0] - coords[10])
            # C4 (index 3) to C6 (index 11)
            bond2_dist = np.linalg.norm(coords[3] - coords[11])

            new_bond1_distances.append(bond1_dist)
            new_bond2_distances.append(bond2_dist)

        new_bond1_distances = np.array(new_bond1_distances)
        new_bond2_distances = np.array(new_bond2_distances)

        # Both bonds should form (distances decrease)
        assert (
            new_bond1_distances[0] > new_bond1_distances[-1] + 1.0
        ), "New C-C bond 1 should form"
        assert (
            new_bond2_distances[0] > new_bond2_distances[-1] + 1.0
        ), "New C-C bond 2 should form"

        # Bonds should form somewhat concertedly (characteristic of Diels-Alder)
        bond_ratio = new_bond1_distances / new_bond2_distances
        # Ratio shouldn't vary too wildly if concerted
        assert np.std(bond_ratio) < 1.0, "Bond formation should be reasonably concerted"

    def test_diels_alder_stereospecificity(
        self, diels_alder_reactants, diels_alder_product
    ):
        """Test stereochemical aspects of Diels-Alder reaction."""
        reaction = Reaction(
            diels_alder_reactants, diels_alder_product, name="DA_stereo"
        )
        path_geoms = reaction.interpolate(npoints=10)

        # Check that the reaction maintains proper orbital overlap
        # In real implementation, this would analyze molecular orbitals
        # Here we check geometric constraints for suprafacial addition

        for geom in path_geoms:
            coords = geom.coords3d

            # Check that ethylene approaches from consistent side
            # C5 and C6 should maintain similar z-coordinates during approach
            c5_z = coords[10, 2]  # C5 z-coordinate
            c6_z = coords[11, 2]  # C6 z-coordinate

            # Should approach from same side (suprafacial)
            z_diff = abs(c5_z - c6_z)
            assert z_diff < 2.0, "Ethylene should approach suprafacially"

    def test_diels_alder_regioselectivity(
        self, diels_alder_reactants, diels_alder_product
    ):
        """Test regioselectivity of Diels-Alder reaction."""
        # This test checks that the reaction follows expected regioselectivity
        # For symmetric reactants, there's only one major product

        reaction = Reaction(diels_alder_reactants, diels_alder_product, name="DA_regio")

        # Check that terminal carbons of butadiene form bonds with ethylene
        coords_r = diels_alder_reactants.coords3d
        coords_p = diels_alder_product.coords3d

        # Initial distances between terminal C of butadiene and ethylene C's
        c1_c5_initial = np.linalg.norm(coords_r[0] - coords_r[10])  # C1-C5
        c4_c6_initial = np.linalg.norm(coords_r[3] - coords_r[11])  # C4-C6

        # Final distances (should be bonding)
        c1_c5_final = np.linalg.norm(coords_p[0] - coords_p[10])
        c4_c6_final = np.linalg.norm(coords_p[3] - coords_p[11])

        # Both distances should decrease significantly (bond formation)
        assert c1_c5_initial > c1_c5_final + 1.0, "C1-C5 bond should form"
        assert c4_c6_initial > c4_c6_final + 1.0, "C4-C6 bond should form"

        # Final bonds should be typical C-C single bond length
        assert 1.3 < c1_c5_final < 1.7, "C1-C5 final distance should be bonding"
        assert 1.3 < c4_c6_final < 1.7, "C4-C6 final distance should be bonding"

    def test_diels_alder_energetics(
        self, diels_alder_reactants, diels_alder_product, diels_alder_ts
    ):
        """Test energy profile of Diels-Alder reaction."""
        calculator = MLPCalculator(model_type="mock_cycloaddition")

        # Calculate energies
        calculator.calculate(diels_alder_reactants)
        calculator.calculate(diels_alder_product)
        calculator.calculate(diels_alder_ts)

        reaction = Reaction(
            reactant=diels_alder_reactants,
            product=diels_alder_product,
            ts=diels_alder_ts,
            name="DA_energetics",
        )

        assert reaction.reaction_energy is not None
        assert reaction.activation_energy is not None

        # Diels-Alder is typically exothermic
        # (Though our mock calculator might not reflect this)
        print(f"Reaction energy: {reaction.reaction_energy:.6f} Hartree")
        print(f"Activation energy: {reaction.activation_energy:.6f} Hartree")

        # TS should be higher energy than reactants and products
        assert diels_alder_ts.energy > diels_alder_reactants.energy
        assert diels_alder_ts.energy > diels_alder_product.energy

    def test_diels_alder_concerted_mechanism(
        self, diels_alder_reactants, diels_alder_product
    ):
        """Test that the reaction follows a concerted mechanism."""
        reaction = Reaction(
            diels_alder_reactants, diels_alder_product, name="DA_concerted"
        )
        path_geoms = reaction.interpolate(npoints=20)

        # Calculate bond orders along the path to verify concerted mechanism
        c1_c5_distances = []
        c4_c6_distances = []
        c2_c3_distances = []  # Internal double bond in butadiene
        c5_c6_distances = []  # Double bond in ethylene

        for geom in path_geoms:
            coords = geom.coords3d

            c1_c5_distances.append(np.linalg.norm(coords[0] - coords[10]))
            c4_c6_distances.append(np.linalg.norm(coords[3] - coords[11]))
            c2_c3_distances.append(np.linalg.norm(coords[1] - coords[2]))
            c5_c6_distances.append(np.linalg.norm(coords[10] - coords[11]))

        c1_c5_distances = np.array(c1_c5_distances)
        c4_c6_distances = np.array(c4_c6_distances)
        c2_c3_distances = np.array(c2_c3_distances)
        c5_c6_distances = np.array(c5_c6_distances)

        # For concerted mechanism, bond changes should be coordinated
        # New bonds form while old pi bonds break simultaneously

        # New bonds should form (distances decrease)
        assert c1_c5_distances[0] > c1_c5_distances[-1]
        assert c4_c6_distances[0] > c4_c6_distances[-1]

        # Double bonds should lengthen (become single bonds)
        assert c2_c3_distances[-1] > c2_c3_distances[0]  # C2-C3 lengthens
        assert c5_c6_distances[-1] > c5_c6_distances[0]  # C5-C6 lengthens

        # Check that bond changes are correlated (concerted)
        # As new bonds form, old bonds should break
        new_bond_formation = (c1_c5_distances[0] - c1_c5_distances) / c1_c5_distances[0]
        old_bond_breaking = (c5_c6_distances - c5_c6_distances[0]) / c5_c6_distances[0]

        # Should be positively correlated for concerted mechanism
        correlation = np.corrcoef(new_bond_formation, old_bond_breaking)[0, 1]
        assert correlation > 0.5, "Bond formation and breaking should be correlated"

    def test_diels_alder_xyz_trajectory(
        self, diels_alder_reactants, diels_alder_product, diels_alder_ts
    ):
        """Test XYZ trajectory generation for Diels-Alder reaction."""
        reaction = Reaction(
            diels_alder_reactants,
            diels_alder_product,
            ts=diels_alder_ts,
            name="DA_trajectory",
        )

        # Generate trajectory
        xyz_traj = reaction.to_xyz_trajectory()

        # Should contain all atoms in each frame
        assert "16" in xyz_traj  # Number of atoms

        # Should contain carbon and hydrogen atoms
        assert "C " in xyz_traj
        assert "H " in xyz_traj

        # Test with interpolated path
        path_geoms = reaction.interpolate(npoints=5)
        calculator = MLPCalculator()

        for geom in path_geoms:
            calculator.calculate(geom)

        xyz_path = reaction.to_xyz_trajectory(path_geoms)

        # Should contain 5 geometry blocks
        frame_count = xyz_path.count("16\n")
        assert frame_count == 5
