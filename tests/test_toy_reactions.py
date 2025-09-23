"""
Toy reaction tests inspired by pysisyphus examples.

This module implements classic toy reactions used in computational chemistry
to demonstrate and test optimization algorithms. These include simple
model reactions that are easy to visualize and understand.
"""

import os
import tempfile

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from qme import QMEOptimizer


class TestToyReactions:
    """Test suite for toy reactions and model systems."""

    @pytest.fixture(params=["so3lr", "uma"])
    def qme_optimizer(self, request):
        """QME optimizer with different backends."""
        return QMEOptimizer(use_mock=True, backend=request.param)

    def test_h2_plus_h_reaction(self, qme_optimizer):
        """Test H2 + H -> H + H2 exchange reaction."""
        # Create H3 system in collinear arrangement
        # This is a classic test case for reaction dynamics

        positions = np.array(
            [
                [-1.5, 0.0, 0.0],  # H1 (reactant H2)
                [-0.8, 0.0, 0.0],  # H2 (reactant H2)
                [1.0, 0.0, 0.0],  # H3 (attacking H)
            ]
        )

        h3_system = Atoms("HHH", positions=positions)
        qme_optimizer.atoms = h3_system
        h3_system.set_calculator(qme_optimizer.calculator)

        # Test optimization of reactant-like configuration
        result = qme_optimizer.optimize_minimum(fmax=0.3, steps=10)

        assert result is not None
        assert "optimized_atoms" in result
        optimized = result["optimized_atoms"]
        assert len(optimized) == 3
        assert optimized.get_chemical_symbols() == ["H", "H", "H"]

    def test_hydrogen_exchange_product(self, qme_optimizer):
        """Test product configuration of H + H2 reaction."""
        # Product arrangement: H + H2 (switched from reactant)
        positions = np.array(
            [
                [-1.0, 0.0, 0.0],  # H1 (now isolated)
                [0.8, 0.0, 0.0],  # H2 (product H2)
                [1.5, 0.0, 0.0],  # H3 (product H2)
            ]
        )

        h3_system = Atoms("HHH", positions=positions)
        qme_optimizer.atoms = h3_system
        h3_system.set_calculator(qme_optimizer.calculator)

        # Optimize product configuration
        result = qme_optimizer.optimize_minimum(fmax=0.3, steps=10)

        assert result is not None
        optimized = result["optimized_atoms"]
        assert len(optimized) == 3

    def test_water_dimer_formation(self, qme_optimizer):
        """Test water dimer formation reaction."""
        # Create two water molecules for dimer formation
        water1 = molecule("H2O")
        water2 = molecule("H2O")

        # Position waters for hydrogen bonding
        pos1 = water1.get_positions()
        pos2 = water2.get_positions()
        pos2 += [2.5, 0.0, 0.0]  # Separate the molecules

        # Combine into single system
        positions = np.vstack([pos1, pos2])
        symbols = water1.get_chemical_symbols() + water2.get_chemical_symbols()

        water_dimer = Atoms(symbols, positions=positions)
        qme_optimizer.atoms = water_dimer
        water_dimer.set_calculator(qme_optimizer.calculator)

        # Optimize the dimer
        result = qme_optimizer.optimize_minimum(fmax=0.2, steps=15)

        assert result is not None
        optimized = result["optimized_atoms"]
        assert len(optimized) == 6  # 2 * 3 atoms
        expected_symbols = ["O", "H", "H", "O", "H", "H"]
        assert optimized.get_chemical_symbols() == expected_symbols

    def test_ammonia_water_complex(self, qme_optimizer):
        """Test NH3...H2O hydrogen-bonded complex."""
        # Create NH3 and H2O complex
        nh3 = molecule("NH3")
        h2o = molecule("H2O")

        # Position for hydrogen bonding interaction
        nh3_pos = nh3.get_positions()
        h2o_pos = h2o.get_positions()
        h2o_pos += [3.0, 0.0, 0.0]  # Separate initially

        # Combine systems
        positions = np.vstack([nh3_pos, h2o_pos])
        symbols = nh3.get_chemical_symbols() + h2o.get_chemical_symbols()

        complex_system = Atoms(symbols, positions=positions)
        qme_optimizer.atoms = complex_system
        complex_system.set_calculator(qme_optimizer.calculator)

        # Optimize complex
        result = qme_optimizer.optimize_minimum(fmax=0.2, steps=15)

        assert result is not None
        optimized = result["optimized_atoms"]
        assert len(optimized) == 7  # NH3 (4) + H2O (3)

    def test_methane_hydrogen_abstraction(self, qme_optimizer):
        """Test CH4 + H -> CH3 + H2 abstraction reaction setup."""
        # Create CH4 + H system
        ch4 = molecule("CH4")

        # Add an additional H atom for abstraction
        ch4_pos = ch4.get_positions()
        extra_h_pos = np.array([[2.0, 0.0, 0.0]])  # H atom approaching

        positions = np.vstack([ch4_pos, extra_h_pos])
        symbols = ch4.get_chemical_symbols() + ["H"]

        reaction_system = Atoms(symbols, positions=positions)
        qme_optimizer.atoms = reaction_system
        reaction_system.set_calculator(qme_optimizer.calculator)

        # Optimize reactant complex
        result = qme_optimizer.optimize_minimum(fmax=0.2, steps=15)

        assert result is not None
        optimized = result["optimized_atoms"]
        assert len(optimized) == 6  # CH4 (5) + H (1)
        expected_symbols = ["C", "H", "H", "H", "H", "H"]
        assert optimized.get_chemical_symbols() == expected_symbols

    def test_proton_transfer_chain(self, qme_optimizer):
        """Test proton transfer in H2O...H+...H2O system."""
        # Create water-hydronium-water system
        # Simplified as H-O-H...H...H-O-H

        positions = np.array(
            [
                [-2.0, 0.0, 0.0],  # H (water 1)
                [-1.0, 0.0, 0.0],  # O (water 1)
                [-1.5, 0.8, 0.0],  # H (water 1)
                [0.0, 0.0, 0.0],  # H+ (transferring proton)
                [1.0, 0.0, 0.0],  # O (water 2)
                [2.0, 0.0, 0.0],  # H (water 2)
                [1.5, 0.8, 0.0],  # H (water 2)
            ]
        )

        symbols = ["H", "O", "H", "H", "O", "H", "H"]
        pt_system = Atoms(symbols, positions=positions)

        qme_optimizer.atoms = pt_system
        pt_system.set_calculator(qme_optimizer.calculator)

        # Optimize proton transfer system
        result = qme_optimizer.optimize_minimum(fmax=0.3, steps=15)

        assert result is not None
        optimized = result["optimized_atoms"]
        assert len(optimized) == 7

    def test_diatomic_dissociation_series(self, qme_optimizer):
        """Test dissociation of different diatomic molecules."""
        # Test series: H2, N2 (using H2 as model), O2 (using H2 as model)
        # Since we're using mock calculators, we'll use H2 for all

        molecules = ["H2"]  # Could extend with others if available

        for mol_name in molecules:
            mol = molecule(mol_name)

            # Create series of stretched geometries
            base_positions = mol.get_positions()
            bond_lengths = [0.7, 1.0, 1.5, 2.0, 3.0]

            energies = []
            for length in bond_lengths:
                test_mol = mol.copy()
                positions = base_positions.copy()

                # Set bond length
                bond_vector = positions[1] - positions[0]
                bond_vector = bond_vector / np.linalg.norm(bond_vector)
                positions[1] = positions[0] + bond_vector * length

                test_mol.set_positions(positions)
                qme_optimizer.atoms = test_mol
                test_mol.set_calculator(qme_optimizer.calculator)

                # Short optimization
                result = qme_optimizer.optimize_minimum(fmax=0.3, steps=5)

                if result and "optimized_atoms" in result:
                    energy = result["optimized_atoms"].get_potential_energy()
                    energies.append((length, energy))

            # Should have computed several points
            assert len(energies) >= 3

            # All energies should be finite
            for length, energy in energies:
                assert isinstance(energy, float)
                assert np.isfinite(energy)

    def test_conformational_isomers(self, qme_optimizer):
        """Test optimization of different conformers."""
        # Use butane-like system (simplified as 4 connected atoms)
        # Create linear and bent configurations

        # Linear configuration
        linear_positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )

        # Bent configuration
        bent_positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 1.0, 0.0], [2.5, 1.0, 0.0]]
        )

        symbols = ["C", "C", "C", "C"]  # Model carbon chain

        for name, positions in [("linear", linear_positions), ("bent", bent_positions)]:
            conformer = Atoms(symbols, positions=positions)
            qme_optimizer.atoms = conformer
            conformer.set_calculator(qme_optimizer.calculator)

            result = qme_optimizer.optimize_minimum(fmax=0.2, steps=10)

            assert result is not None, f"Optimization failed for {name} conformer"
            optimized = result["optimized_atoms"]
            assert len(optimized) == 4
            assert optimized.get_chemical_symbols() == symbols

    def test_ion_molecule_reactions(self, qme_optimizer):
        """Test ion-molecule reaction setups."""
        # Model H3O+ + NH3 -> NH4+ + H2O reaction

        # Reactant side: H3O+ and NH3 separated
        h3o_plus = molecule("H2O")
        nh3 = molecule("NH3")

        # Add extra H to water to make H3O+
        h3o_positions = h3o_plus.get_positions()
        extra_h = np.array([[0.0, 0.0, 1.0]])  # H above the water
        h3o_positions = np.vstack([h3o_positions, extra_h])

        # Position NH3 nearby
        nh3_positions = nh3.get_positions()
        nh3_positions += [3.0, 0.0, 0.0]  # Separate from H3O+

        # Combine systems
        positions = np.vstack([h3o_positions, nh3_positions])
        symbols = ["O", "H", "H", "H"] + ["N", "H", "H", "H"]

        reaction_system = Atoms(symbols, positions=positions)
        qme_optimizer.atoms = reaction_system
        reaction_system.set_calculator(qme_optimizer.calculator)

        # Optimize reactant complex
        result = qme_optimizer.optimize_minimum(fmax=0.3, steps=15)

        assert result is not None
        optimized = result["optimized_atoms"]
        assert len(optimized) == 8

    def test_radical_reactions(self, qme_optimizer):
        """Test radical reaction models."""
        # Model H + CH4 -> H2 + CH3 radical reaction

        ch4 = molecule("CH4")
        ch4_positions = ch4.get_positions()

        # Add H radical
        h_radical_pos = np.array([[2.5, 0.0, 0.0]])

        positions = np.vstack([ch4_positions, h_radical_pos])
        symbols = ch4.get_chemical_symbols() + ["H"]

        radical_system = Atoms(symbols, positions=positions)
        qme_optimizer.atoms = radical_system
        radical_system.set_calculator(qme_optimizer.calculator)

        # Optimize radical system
        result = qme_optimizer.optimize_minimum(fmax=0.3, steps=10)

        assert result is not None
        optimized = result["optimized_atoms"]
        assert len(optimized) == 6

    def test_reaction_coordinate_scan(self, qme_optimizer):
        """Test scanning along a reaction coordinate."""
        # Scan H2 dissociation coordinate
        h2 = molecule("H2")
        base_positions = h2.get_positions()

        # Scan different bond lengths
        scan_distances = [0.7, 0.9, 1.1, 1.3, 1.5]
        scan_results = []

        for distance in scan_distances:
            h2_scan = h2.copy()
            positions = base_positions.copy()
            positions[1][0] = positions[0][0] + distance
            h2_scan.set_positions(positions)

            qme_optimizer.atoms = h2_scan
            h2_scan.set_calculator(qme_optimizer.calculator)

            # Constrained optimization (fix bond length, optimize other coordinates)
            # For simplicity, just do a regular optimization
            result = qme_optimizer.optimize_minimum(fmax=0.2, steps=3)

            if result and "optimized_atoms" in result:
                energy = result["optimized_atoms"].get_potential_energy()
                scan_results.append((distance, energy))

        # Should have completed most scans
        assert len(scan_results) >= len(scan_distances) // 2

        # Check energy data
        for distance, energy in scan_results:
            assert isinstance(energy, float)
            assert np.isfinite(energy)

    def test_bimolecular_collision_setup(self, qme_optimizer):
        """Test setup for bimolecular collision."""
        # H2 + H2 collision setup
        h2_1 = molecule("H2")
        h2_2 = molecule("H2")

        # Position molecules for collision
        pos1 = h2_1.get_positions()
        pos2 = h2_2.get_positions()

        # Separate and orient for collision
        pos2 += [4.0, 0.0, 0.0]  # Move second H2 away
        pos2 = pos2[::-1]  # Flip orientation

        positions = np.vstack([pos1, pos2])
        symbols = ["H", "H", "H", "H"]

        collision_system = Atoms(symbols, positions=positions)
        qme_optimizer.atoms = collision_system
        collision_system.set_calculator(qme_optimizer.calculator)

        # Optimize collision complex
        result = qme_optimizer.optimize_minimum(fmax=0.3, steps=10)

        assert result is not None
        optimized = result["optimized_atoms"]
        assert len(optimized) == 4
        assert optimized.get_chemical_symbols() == symbols
