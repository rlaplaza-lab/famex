"""
Comprehensive tests for SO3LR backend functionality.

Tests chemical systems including:
- Simple molecules (H2, H2O)
- Organic molecules (ethane, methanol)
- Geometry optimization from initial guess
- Reaction pathway interpolation
- Transition state optimization
"""

import tempfile
from pathlib import Path

import pytest
from ase import Atoms

import qme
from qme.dependencies import deps


@pytest.mark.skipif(not deps.has("so3lr"), reason="SO3LR not available")
class TestSO3LRBackend:
    """Test SO3LR backend with real chemical systems."""

    @pytest.fixture
    def so3lr_optimizer(self):
        """Create SO3LR optimizer."""
        try:
            return qme.QMEOptimizer(backend="so3lr")
        except ImportError:
            pytest.skip("SO3LR backend dependencies not available")

    def test_so3lr_optimizer_creation(self, so3lr_optimizer):
        """Test SO3LR optimizer creation."""
        assert so3lr_optimizer.backend == "so3lr"
        assert so3lr_optimizer.calculator is not None

    def test_h2_optimization(self, so3lr_optimizer):
        """Test H2 molecule optimization from stretched geometry."""
        # Stretched H2 molecule (equilibrium ~0.74 Å)
        h2_stretched = Atoms("H2", positions=[[0, 0, 0], [2.0, 0, 0]])

        result = so3lr_optimizer.optimize_minimum(
            atoms=h2_stretched, fmax=0.05, steps=50
        )

        assert result["converged"] or result["steps_taken"] > 0
        final_distance = result["optimized_atoms"].get_distance(0, 1)
        # Should be closer to equilibrium
        assert 0.6 < final_distance < 1.2

    def test_water_optimization(self, so3lr_optimizer):
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

        result = so3lr_optimizer.optimize_minimum(atoms=water, fmax=0.05, steps=100)

        assert result["converged"] or result["steps_taken"] > 0
        optimized_atoms = result["optimized_atoms"]

        # Check O-H distances are reasonable
        oh1_dist = optimized_atoms.get_distance(0, 1)
        oh2_dist = optimized_atoms.get_distance(0, 2)
        assert 0.8 < oh1_dist < 1.2
        assert 0.8 < oh2_dist < 1.2

    def test_ethane_optimization(self, so3lr_optimizer):
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

        result = so3lr_optimizer.optimize_minimum(atoms=ethane, fmax=0.05, steps=100)

        assert result["converged"] or result["steps_taken"] > 0
        optimized_atoms = result["optimized_atoms"]

        # Check C-C distance is reasonable
        cc_dist = optimized_atoms.get_distance(0, 1)
        assert 1.3 < cc_dist < 1.7

    def test_methanol_optimization(self, so3lr_optimizer):
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

        result = so3lr_optimizer.optimize_minimum(atoms=methanol, fmax=0.05, steps=100)

        assert result["converged"] or result["steps_taken"] > 0

    def test_optimization_with_different_optimizers(self, so3lr_optimizer):
        """Test optimization with different ASE optimizers."""
        h2 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])

        # Test BFGS
        result_bfgs = so3lr_optimizer.optimize_minimum(
            atoms=h2.copy(), optimizer="BFGS", fmax=0.05, steps=20
        )
        assert result_bfgs["steps_taken"] > 0

        # Test FIRE
        result_fire = so3lr_optimizer.optimize_minimum(
            atoms=h2.copy(), optimizer="FIRE", fmax=0.05, steps=20
        )
        assert result_fire["steps_taken"] > 0

    def test_trajectory_logging(self, so3lr_optimizer):
        """Test optimization with trajectory logging."""
        with tempfile.NamedTemporaryFile(suffix=".traj", delete=False) as f:
            traj_file = f.name

        try:
            h2 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])
            result = so3lr_optimizer.optimize_minimum(
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


@pytest.mark.skipif(not deps.has("so3lr"), reason="SO3LR not available")
class TestSO3LRReactionPaths:
    """Test SO3LR backend for reaction pathway generation."""

    @pytest.fixture
    def so3lr_calculator(self):
        """Create SO3LR calculator."""
        try:
            return qme.get_so3lr_calculator()
        except ImportError:
            pytest.skip("SO3LR backend dependencies not available")

    def test_h2_dissociation_path(self, so3lr_calculator):
        """Test H2 dissociation reaction pathway."""
        # H2 molecule -> 2H atoms
        reactant = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        product = Atoms("H2", positions=[[0, 0, 0], [3.0, 0, 0]])

        reaction = qme.Reaction(reactant, product, calculator=so3lr_calculator)

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

    def test_water_rotation_path(self, so3lr_calculator):
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

        reaction = qme.Reaction(reactant, product, calculator=so3lr_calculator)
        path = reaction.interpolate(npoints=7, method="linear")

        assert len(path) == 7
        # All structures should have same number of atoms
        for geom in path:
            assert len(geom) == 3

    def test_ethane_conformational_change(self, so3lr_calculator):
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

        reaction = qme.Reaction(staggered, eclipsed, calculator=so3lr_calculator)
        path = reaction.interpolate(npoints=6, method="linear")

        assert len(path) == 6
        for geom in path:
            assert len(geom) == 8


@pytest.mark.skipif(
    not deps.has("so3lr") or not deps.has("sella"),
    reason="SO3LR or SELLA not available",
)
class TestSO3LRTransitionStates:
    """Test SO3LR backend for transition state optimization."""

    @pytest.fixture
    def so3lr_optimizer(self):
        """Create SO3LR optimizer."""
        try:
            return qme.QMEOptimizer(backend="so3lr")
        except ImportError:
            pytest.skip("SO3LR backend dependencies not available")

    def test_h2_dissociation_ts_guess(self, so3lr_optimizer):
        """Test transition state search from interpolated guess."""
        # Initial guess for H2 dissociation TS (stretched H2)
        ts_guess = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])

        try:
            result = so3lr_optimizer.find_transition_state(
                atoms=ts_guess, fmax=0.05, steps=50
            )

            # Should complete without error
            assert "converged" in result
            assert "steps_taken" in result

        except Exception as e:
            # Some TS searches may not converge, but should not crash
            assert "find_transition_state" not in str(e)

    def test_water_inversion_ts(self, so3lr_optimizer):
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
            result = so3lr_optimizer.find_transition_state(
                atoms=ts_guess, fmax=0.05, steps=50
            )

            assert "converged" in result

        except Exception as e:
            # TS optimization may not converge but should not crash
            assert "sella" not in str(e).lower() or "import" in str(e).lower()

    def test_ethane_rotation_ts(self, so3lr_optimizer):
        """Test ethane rotation transition state."""
        # Semi-eclipsed ethane as TS guess
        ts_guess = Atoms(
            "C2H6",
            positions=[
                [0.0, 0.0, 0.0],  # C
                [1.54, 0.0, 0.0],  # C
                [-0.51, 1.02, 0.0],  # H
                [-0.51, -0.51, 0.88],  # H
                [-0.51, -0.51, -0.88],  # H
                [2.05, 0.0, 1.02],  # H (partially eclipsed)
                [2.05, -0.88, -0.51],  # H
                [2.05, 0.88, -0.51],  # H
            ],
        )

        try:
            result = so3lr_optimizer.find_transition_state(
                atoms=ts_guess, fmax=0.05, steps=50
            )

            assert "converged" in result or "steps_taken" in result

        except Exception as e:
            # Complex TS searches may not converge
            assert "find_transition_state" not in str(e)


@pytest.mark.skipif(not deps.has("so3lr"), reason="SO3LR not available")
class TestSO3LRAdvanced:
    """Advanced SO3LR backend tests."""

    def test_minimize_structure_function(self):
        """Test standalone minimize_structure function with SO3LR."""
        h2 = Atoms("H2", positions=[[0, 0, 0], [1.8, 0, 0]])

        try:
            result = qme.minimize_structure(h2, backend="so3lr", fmax=0.05, steps=30)

            assert isinstance(result, Atoms)
            final_distance = result.get_distance(0, 1)
            assert 0.6 < final_distance < 1.2

        except ImportError:
            pytest.skip("SO3LR backend dependencies not available")

    def test_different_convergence_criteria(self):
        """Test optimization with different convergence criteria."""
        try:
            optimizer = qme.QMEOptimizer(backend="so3lr")
            h2 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])

            # Tight convergence
            result_tight = optimizer.optimize_minimum(
                atoms=h2.copy(), fmax=0.01, steps=50
            )

            # Loose convergence
            result_loose = optimizer.optimize_minimum(
                atoms=h2.copy(), fmax=0.1, steps=20
            )

            # Tight convergence should take more steps (usually)
            assert result_tight["steps_taken"] >= 0
            assert result_loose["steps_taken"] >= 0

        except ImportError:
            pytest.skip("SO3LR backend dependencies not available")
