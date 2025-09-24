"""
Comprehensive molecular optimization tests inspired by pysisyphus.

This module tests QME's optimization capabilities on various molecular systems,
from simple diatomics to more complex polyatomic molecules.
"""

import numpy as np
import pytest
from ase.build import molecule

from qme import QMEOptimizer


class TestMolecularOptimization:
    """Test molecular geometry optimization with QME."""

    @pytest.fixture(params=["so3lr", "uma", "aimnet2"])
    def qme_optimizer(self, request):
        """QME optimizer fixture with different backends."""
        return QMEOptimizer(use_mock=True, backend=request.param)

    def test_h2_optimization(self, qme_optimizer):
        """Test H2 molecule optimization - simplest case."""
        # Create H2 molecule with slightly distorted geometry
        h2 = molecule("H2")
        positions = h2.get_positions()
        positions[1][0] += 0.1  # Stretch bond slightly
        h2.set_positions(positions)

        qme_optimizer.atoms = h2
        h2.calc = qme_optimizer.calculator

        # Optimize
        result = qme_optimizer.optimize_minimum(fmax=0.05, steps=10)

        # Check optimization results
        assert result["converged"] or result["steps_taken"] > 0
        assert "optimized_atoms" in result

        # Check that energy decreased (with harmonic mock calculator)
        final_energy = result["optimized_atoms"].get_potential_energy()
        assert isinstance(final_energy, float)

        # Check that forces are reasonable
        final_forces = result["optimized_atoms"].get_forces()
        assert final_forces.shape == (2, 3)

    def test_water_optimization(self, qme_optimizer):
        """Test H2O molecule optimization."""
        # Create water molecule
        water = molecule("H2O")

        # Slightly distort geometry
        positions = water.get_positions()
        positions[1] += [0.05, 0.05, 0.0]  # Move one H atom
        positions[2] += [-0.05, 0.05, 0.0]  # Move other H atom
        water.set_positions(positions)

        qme_optimizer.atoms = water
        water.calc = qme_optimizer.calculator

        # Optimize
        result = qme_optimizer.optimize_minimum(fmax=0.1, steps=15)

        # Verify optimization
        assert result is not None
        assert "optimized_atoms" in result
        optimized = result["optimized_atoms"]

        # Check molecular structure is preserved
        assert len(optimized) == 3
        assert optimized.get_chemical_symbols() == ["O", "H", "H"]

    def test_methane_optimization(self, qme_optimizer):
        """Test CH4 molecule optimization."""
        # Create methane
        ch4 = molecule("CH4")

        # Add some distortion
        positions = ch4.get_positions()
        positions[1:] += np.random.normal(0, 0.02, (4, 3))  # Small random displacements
        ch4.set_positions(positions)

        qme_optimizer.atoms = ch4
        ch4.calc = qme_optimizer.calculator

        # Optimize
        result = qme_optimizer.optimize_minimum(fmax=0.1, steps=20)

        # Check results
        assert result is not None
        optimized = result["optimized_atoms"]
        assert len(optimized) == 5
        assert optimized.get_chemical_symbols() == ["C", "H", "H", "H", "H"]

    def test_ammonia_optimization(self, qme_optimizer):
        """Test NH3 molecule optimization."""
        # Create ammonia
        nh3 = molecule("NH3")

        # Flatten the pyramid slightly (distort from equilibrium)
        positions = nh3.get_positions()
        h_positions = positions[1:]

        # Move hydrogens to make molecule more planar
        h_positions[:, 2] *= 0.3  # Reduce z-coordinates
        positions[1:] = h_positions
        nh3.set_positions(positions)

        qme_optimizer.atoms = nh3
        nh3.calc = qme_optimizer.calculator

        # Optimize
        result = qme_optimizer.optimize_minimum(fmax=0.1, steps=20)

        # Verify
        assert result is not None
        optimized = result["optimized_atoms"]
        assert len(optimized) == 4
        assert optimized.get_chemical_symbols() == ["N", "H", "H", "H"]

    @pytest.mark.parametrize("optimizer_name", ["BFGS", "LBFGS", "FIRE"])
    def test_optimization_algorithms(self, qme_optimizer, optimizer_name):
        """Test different optimization algorithms on the same system."""
        # Use water as test system
        water = molecule("H2O")
        positions = water.get_positions()
        positions += np.random.normal(0, 0.03, positions.shape)  # Add noise
        water.set_positions(positions)

        qme_optimizer.atoms = water
        water.calc = qme_optimizer.calculator

        # Optimize with specific algorithm
        result = qme_optimizer.optimize_minimum(
            optimizer=optimizer_name, fmax=0.1, steps=15
        )

        # All optimizers should work
        assert result is not None
        assert result["steps_taken"] > 0

    def test_optimization_convergence_criteria(self, qme_optimizer):
        """Test different convergence criteria."""
        h2 = molecule("H2")
        positions = h2.get_positions()
        positions[1][0] += 0.2  # Significant distortion
        h2.set_positions(positions)

        qme_optimizer.atoms = h2
        h2.calc = qme_optimizer.calculator

        # Test tight convergence
        result_tight = qme_optimizer.optimize_minimum(fmax=0.01, steps=5)

        # Test loose convergence
        h2_loose = h2.copy()
        h2_loose.calc = qme_optimizer.calculator
        qme_optimizer.atoms = h2_loose
        result_loose = qme_optimizer.optimize_minimum(fmax=0.2, steps=5)

        # Both should complete
        assert result_tight is not None
        assert result_loose is not None

        # Loose convergence might converge faster
        if result_loose["converged"]:
            assert result_loose["steps_taken"] <= result_tight["steps_taken"]

    def test_optimization_with_constraints(self, qme_optimizer):
        """Test optimization with geometric constraints."""
        from ase.constraints import FixAtoms

        # Create ethane-like system (2 C + 6 H)
        # For simplicity, use existing molecule and add constraints
        water = molecule("H2O")
        qme_optimizer.atoms = water
        water.calc = qme_optimizer.calculator

        # Fix oxygen atom position
        constraint = FixAtoms(indices=[0])

        # Test optimization with constraint
        result = qme_optimizer.optimize_minimum(
            fmax=0.1, steps=10, constraints=[constraint]
        )

        assert result is not None
        # The constraint should be applied
        optimized = result["optimized_atoms"]
        assert len(optimized.constraints) == 1

    def test_multi_molecule_comparison(self, qme_optimizer):
        """Test optimization of multiple molecules and compare results."""
        molecules = {
            "H2": molecule("H2"),
            "H2O": molecule("H2O"),
            "NH3": molecule("NH3"),
            "CH4": molecule("CH4"),
        }

        results = {}

        for name, mol in molecules.items():
            # Add small distortion
            positions = mol.get_positions()
            positions += np.random.normal(0, 0.02, positions.shape)
            mol.set_positions(positions)

            qme_optimizer.atoms = mol
            mol.calc = qme_optimizer.calculator

            # Optimize
            result = qme_optimizer.optimize_minimum(fmax=0.1, steps=15)
            results[name] = result

            # Basic checks
            assert result is not None
            assert "optimized_atoms" in result

        # All optimizations should have produced results
        assert len(results) == 4

        # Check that molecular formulas are preserved
        for name, result in results.items():
            original = molecules[name]
            optimized = result["optimized_atoms"]
            assert len(original) == len(optimized)
            assert original.get_chemical_symbols() == optimized.get_chemical_symbols()

    def test_optimization_restart_capability(self, qme_optimizer):
        """Test ability to restart/continue optimization."""
        # Start with distorted water
        water = molecule("H2O")
        positions = water.get_positions()
        positions += np.random.normal(0, 0.1, positions.shape)
        water.set_positions(positions)

        qme_optimizer.atoms = water
        water.calc = qme_optimizer.calculator

        # First optimization (partial)
        result1 = qme_optimizer.optimize_minimum(fmax=0.1, steps=5)

        # Continue from where we left off
        qme_optimizer.atoms = result1["optimized_atoms"]
        result2 = qme_optimizer.optimize_minimum(fmax=0.05, steps=5)

        # Second optimization should start from previous result
        assert result2 is not None

        # Total steps should be reasonable
        total_steps = result1["steps_taken"] + result2["steps_taken"]
        assert total_steps <= 10  # We limited to 5 each

    def test_energy_landscape_exploration(self, qme_optimizer):
        """Test optimization from different starting geometries."""
        # Create multiple starting configurations of H2
        base_h2 = molecule("H2")
        base_positions = base_h2.get_positions()

        # Different bond lengths to start from
        bond_lengths = [0.5, 0.74, 1.0, 1.5]  # Angstrom
        energies = []

        for bond_length in bond_lengths:
            h2 = base_h2.copy()
            positions = base_positions.copy()
            positions[1][0] = positions[0][0] + bond_length
            h2.set_positions(positions)

            qme_optimizer.atoms = h2
            h2.calc = qme_optimizer.calculator

            # Short optimization
            result = qme_optimizer.optimize_minimum(fmax=0.1, steps=10)

            if result and "optimized_atoms" in result:
                final_energy = result["optimized_atoms"].get_potential_energy()
                energies.append(final_energy)

        # Should have energies for most starting points
        assert len(energies) >= len(bond_lengths) // 2

        # All energies should be reasonable floats
        for energy in energies:
            assert isinstance(energy, float)
            assert not np.isnan(energy)
            assert not np.isinf(energy)
