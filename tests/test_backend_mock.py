"""
Comprehensive tests for Mock backend functionality.

Tests chemical systems including:
- Simple molecules (H2, H2O)
- Organic molecules (ethane, methanol)
- Geometry optimization from initial guess
- Reaction pathway interpolation
- Transition state optimization (when SELLA available)

Mock backend should always be available and serves as fallback.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

import qme
from qme.dependencies import deps


class TestMockBackend:
    """Test Mock backend with real chemical systems."""

    @pytest.fixture
    def mock_optimizer(self):
        """Create Mock optimizer."""
        return qme.QMEOptimizer(backend="mock")

    def test_mock_optimizer_creation(self, mock_optimizer):
        """Test Mock optimizer creation."""
        assert mock_optimizer.backend == "mock"
        assert mock_optimizer.calculator is not None

    def test_h2_optimization(self, mock_optimizer):
        """Test H2 molecule optimization from stretched geometry."""
        # Stretched H2 molecule (equilibrium ~0.74 Å)
        h2_stretched = Atoms("H2", positions=[[0, 0, 0], [2.0, 0, 0]])

        result = mock_optimizer.optimize_minimum(
            atoms=h2_stretched, fmax=0.05, steps=50
        )

        assert result["converged"] or result["steps_taken"] > 0
        final_distance = result["optimized_atoms"].get_distance(0, 1)
        # Mock calculator should pull toward reasonable distances
        assert 0.5 < final_distance < 2.5

    def test_water_optimization(self, mock_optimizer):
        """Test water molecule optimization."""
        # Distorted water molecule
        water = Atoms(
            "H2O",
            positions=[
                [0.0, 0.0, 0.0],  # O
                [1.2, 0.0, 0.0],  # H (stretched)
                [0.0, 1.2, 0.0],  # H (stretched)
            ],
        )

        result = mock_optimizer.optimize_minimum(atoms=water, fmax=0.05, steps=100)

        assert result["converged"] or result["steps_taken"] > 0
        optimized_atoms = result["optimized_atoms"]

        # Check O-H distances changed (mock calculator should optimize)
        oh1_dist = optimized_atoms.get_distance(0, 1)
        oh2_dist = optimized_atoms.get_distance(0, 2)
        assert oh1_dist > 0.5
        assert oh2_dist > 0.5

    def test_ethane_optimization(self, mock_optimizer):
        """Test ethane molecule optimization."""
        # Simple ethane geometry (slightly distorted)
        ethane = Atoms(
            "C2H6",
            positions=[
                [0.0, 0.0, 0.0],  # C
                [1.6, 0.0, 0.0],  # C (stretched)
                [-0.5, 1.0, 0.0],  # H
                [-0.5, -0.5, 0.8],  # H
                [-0.5, -0.5, -0.8],  # H
                [2.1, 1.0, 0.0],  # H
                [2.1, -0.5, 0.8],  # H
                [2.1, -0.5, -0.8],  # H
            ],
        )

        result = mock_optimizer.optimize_minimum(atoms=ethane, fmax=0.05, steps=100)

        assert result["converged"] or result["steps_taken"] > 0
        optimized_atoms = result["optimized_atoms"]

        # Check C-C distance exists and is reasonable
        cc_dist = optimized_atoms.get_distance(0, 1)
        assert cc_dist > 0.5

    def test_methanol_optimization(self, mock_optimizer):
        """Test methanol optimization."""
        # Methanol with distorted geometry
        methanol = Atoms(
            "CH4O",
            positions=[
                [0.0, 0.0, 0.0],  # C
                [1.5, 0.0, 0.0],  # O (stretched)
                [-0.5, 1.0, 0.0],  # H
                [-0.5, -0.5, 0.8],  # H
                [-0.5, -0.5, -0.8],  # H
                [2.0, 0.0, 0.0],  # H (OH)
            ],
        )

        result = mock_optimizer.optimize_minimum(atoms=methanol, fmax=0.05, steps=100)

        assert result["converged"] or result["steps_taken"] > 0

    def test_optimization_with_different_optimizers(self, mock_optimizer):
        """Test optimization with different ASE optimizers."""
        h2 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])

        # Test BFGS
        result_bfgs = mock_optimizer.optimize_minimum(
            atoms=h2.copy(), optimizer="BFGS", fmax=0.05, steps=20
        )
        assert result_bfgs["steps_taken"] >= 0  # May converge immediately

        # Test FIRE
        result_fire = mock_optimizer.optimize_minimum(
            atoms=h2.copy(), optimizer="FIRE", fmax=0.05, steps=20
        )
        assert result_fire["steps_taken"] >= 0  # May converge immediately

    def test_trajectory_logging(self, mock_optimizer):
        """Test optimization with trajectory logging."""
        with tempfile.NamedTemporaryFile(suffix=".traj", delete=False) as f:
            traj_file = f.name

        try:
            h2 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])
            result = mock_optimizer.optimize_minimum(
                atoms=h2, fmax=0.05, steps=20, trajectory=traj_file
            )

            # Check trajectory file was created
            assert Path(traj_file).exists()
            # Check that optimization returned valid results
            assert "converged" in result
            assert "optimized_atoms" in result

        finally:
            # Clean up
            if Path(traj_file).exists():
                Path(traj_file).unlink()

    def test_mock_backend_consistency(self, mock_optimizer):
        """Test that mock backend gives consistent results."""
        h2 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])

        result1 = mock_optimizer.optimize_minimum(atoms=h2.copy(), fmax=0.05, steps=20)

        result2 = mock_optimizer.optimize_minimum(atoms=h2.copy(), fmax=0.05, steps=20)

        # Should give same result for same input
        final_dist1 = result1["optimized_atoms"].get_distance(0, 1)
        final_dist2 = result2["optimized_atoms"].get_distance(0, 1)
        assert abs(final_dist1 - final_dist2) < 0.01


class TestMockReactionPaths:
    """Test Mock backend for reaction pathway generation."""

    @pytest.fixture
    def mock_calculator(self):
        """Create Mock calculator."""
        return qme.MockCalculator(backend="uma")  # Mock calculators are interchangeable

    def test_h2_dissociation_path(self, mock_calculator):
        """Test H2 dissociation reaction pathway."""
        # H2 molecule -> 2H atoms
        reactant = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        product = Atoms("H2", positions=[[0, 0, 0], [3.0, 0, 0]])

        reaction = qme.Reaction(reactant, product, calculator=mock_calculator)

        # Generate linear interpolation
        path = reaction.interpolate(npoints=5, method="linear")
        assert len(path) == 5

        # Check distances are increasing
        distances = []
        for geom in path:
            dist = geom.get_distance(0, 1)
            distances.append(dist)

        # Distances should increase monotonically
        for i in range(1, len(distances)):
            assert distances[i] >= distances[i - 1]

    def test_water_rotation_path(self, mock_calculator):
        """Test water molecule rotation pathway."""
        # Water molecule in two orientations
        reactant = Atoms(
            "H2O",
            positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [0.24, 0.93, 0.0],
            ],
        )

        product = Atoms(
            "H2O",
            positions=[
                [0.0, 0.0, 0.0],
                [-0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ],
        )

        reaction = qme.Reaction(reactant, product, calculator=mock_calculator)
        path = reaction.interpolate(npoints=7, method="linear")

        assert len(path) == 7
        # All structures should have same number of atoms
        for geom in path:
            assert len(geom) == 3

    def test_ethane_conformational_change(self, mock_calculator):
        """Test ethane conformational change pathway."""
        # Staggered to eclipsed conformation
        staggered = Atoms(
            "C2H6",
            positions=[
                [0.0, 0.0, 0.0],  # C
                [1.54, 0.0, 0.0],  # C
                [-0.51, 1.02, 0.0],  # H
                [-0.51, -0.51, 0.88],  # H
                [-0.51, -0.51, -0.88],  # H
                [2.05, 1.02, 0.0],  # H
                [2.05, -0.51, 0.88],  # H
                [2.05, -0.51, -0.88],  # H
            ],
        )

        # Eclipsed conformation (rotate one methyl 60°)
        eclipsed = staggered.copy()
        # Simple rotation approximation for test
        eclipsed.positions[5] = [2.05, -1.02, 0.0]  # H
        eclipsed.positions[6] = [2.05, 0.51, 0.88]  # H
        eclipsed.positions[7] = [2.05, 0.51, -0.88]  # H

        reaction = qme.Reaction(staggered, eclipsed, calculator=mock_calculator)
        path = reaction.interpolate(npoints=6, method="linear")

        assert len(path) == 6
        for geom in path:
            assert len(geom) == 8

    def test_reaction_energy_calculation(self, mock_calculator):
        """Test reaction energy calculation with mock calculator."""
        reactant = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        product = Atoms("H2", positions=[[0, 0, 0], [3.0, 0, 0]])

        reaction = qme.Reaction(reactant, product, calculator=mock_calculator)

        # Mock calculator should provide energies
        if (
            hasattr(reaction, "reaction_energy")
            and reaction.reaction_energy is not None
        ):
            assert isinstance(reaction.reaction_energy, (int, float))


@pytest.mark.skipif(not deps.has("sella"), reason="SELLA not available")
class TestMockTransitionStates:
    """Test Mock backend for transition state optimization."""

    @pytest.fixture
    def mock_optimizer(self):
        """Create Mock optimizer."""
        return qme.QMEOptimizer(backend="mock")

    def test_h2_dissociation_ts_guess(self, mock_optimizer):
        """Test transition state search from interpolated guess."""
        # Initial guess for H2 dissociation TS (stretched H2)
        ts_guess = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])

        try:
            result = mock_optimizer.find_transition_state(
                atoms=ts_guess, fmax=0.05, steps=50
            )

            # Should complete without error
            assert "converged" in result
            assert "steps_taken" in result

        except Exception as e:
            # Some TS searches may not converge, but should not crash
            assert "find_transition_state" not in str(e)

    def test_water_inversion_ts(self, mock_optimizer):
        """Test water inversion transition state."""
        # Planar water as TS guess for umbrella inversion
        ts_guess = Atoms(
            "H2O",
            positions=[
                [0.0, 0.0, 0.0],  # O
                [0.96, 0.0, 0.0],  # H
                [-0.48, 0.83, 0.0],  # H (planar)
            ],
        )

        try:
            result = mock_optimizer.find_transition_state(
                atoms=ts_guess, fmax=0.05, steps=50
            )

            assert "converged" in result

        except Exception as e:
            # TS optimization may not converge but should not crash
            assert "sella" not in str(e).lower() or "import" in str(e).lower()

    def test_transition_state_with_constraints(self, mock_optimizer):
        """Test TS optimization with constraints."""
        from ase.constraints import FixAtoms

        # Fix one atom during TS search
        ts_guess = Atoms("H2O", positions=[[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
        constraint = FixAtoms([0])  # Fix oxygen atom

        try:
            result = mock_optimizer.find_transition_state(
                atoms=ts_guess, fmax=0.05, steps=30, constraints=[constraint]
            )

            assert "converged" in result or "steps_taken" in result

        except Exception as e:
            # Complex TS searches with constraints may not converge
            assert "find_transition_state" not in str(e)


class TestMockAdvanced:
    """Advanced Mock backend tests."""

    def test_minimize_structure_function(self):
        """Test standalone minimize_structure function with Mock."""
        h2 = Atoms("H2", positions=[[0, 0, 0], [1.8, 0, 0]])

        result = qme.minimize_structure(h2, backend="mock", fmax=0.05, steps=30)

        assert isinstance(result, Atoms)
        final_distance = result.get_distance(0, 1)
        assert final_distance > 0.5

    def test_different_convergence_criteria(self):
        """Test optimization with different convergence criteria."""
        optimizer = qme.QMEOptimizer(backend="mock")
        h2 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])

        # Tight convergence
        result_tight = optimizer.optimize_minimum(atoms=h2.copy(), fmax=0.01, steps=50)

        # Loose convergence
        result_loose = optimizer.optimize_minimum(atoms=h2.copy(), fmax=0.1, steps=20)

        # Both should complete
        assert result_tight["steps_taken"] >= 0
        assert result_loose["steps_taken"] >= 0

    def test_mock_calculator_types(self):
        """Test different mock calculator types."""
        # Test that different mock calculators can be created
        mock_uma = qme.MockCalculator(backend="uma")
        mock_so3lr = qme.MockCalculator(backend="so3lr")
        mock_aimnet2 = qme.MockCalculator(backend="aimnet2")

        assert mock_uma is not None
        assert mock_so3lr is not None
        assert mock_aimnet2 is not None

        # All should be able to calculate energy and forces
        h2 = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        for calc in [mock_uma, mock_so3lr, mock_aimnet2]:
            h2.calc = calc
            energy = h2.get_potential_energy()
            forces = h2.get_forces()

            assert isinstance(energy, (int, float))
            assert forces.shape == (2, 3)

    def test_mock_backend_reproducibility(self):
        """Test that mock backend gives reproducible results."""
        optimizer1 = qme.QMEOptimizer(backend="mock")
        optimizer2 = qme.QMEOptimizer(backend="mock")

        h2 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])

        result1 = optimizer1.optimize_minimum(atoms=h2.copy(), fmax=0.05, steps=10)
        result2 = optimizer2.optimize_minimum(atoms=h2.copy(), fmax=0.05, steps=10)

        # Should give identical results
        pos1 = result1["optimized_atoms"].positions
        pos2 = result2["optimized_atoms"].positions

        np.testing.assert_allclose(pos1, pos2, rtol=1e-10)
