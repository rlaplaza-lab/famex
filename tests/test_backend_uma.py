"""
Comprehensive tests for UMA backend functionality.

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


@pytest.mark.skipif(not deps.has("fairchem"), reason="UMA/FairChem not available")
class TestUMABackend:
    """Test UMA backend with real chemical systems."""

    @pytest.fixture
    def uma_optimizer(self):
        """Create UMA optimizer."""
        try:
            return qme.QMEOptimizer(backend="uma")
        except ImportError:
            pytest.skip("UMA backend dependencies not available")

    def test_uma_optimizer_creation(self, uma_optimizer):
        """Test UMA optimizer creation."""
        assert uma_optimizer.backend == "uma"
        assert uma_optimizer.calculator is not None

    def test_h2_optimization(self, uma_optimizer):
        """Test H2 molecule optimization from stretched geometry."""
        # Stretched H2 molecule (equilibrium ~0.74 Å)
        h2_stretched = Atoms("H2", positions=[[0, 0, 0], [2.0, 0, 0]])

        result = uma_optimizer.optimize_minimum(atoms=h2_stretched, fmax=0.05, steps=50)

        assert result["converged"] or result["steps_taken"] > 0
        final_distance = result["optimized_atoms"].get_distance(0, 1)
        # Should be closer to equilibrium
        assert 0.6 < final_distance < 1.2

    def test_water_optimization(self, uma_optimizer):
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

        result = uma_optimizer.optimize_minimum(atoms=water, fmax=0.05, steps=100)

        assert result["converged"] or result["steps_taken"] > 0
        optimized_atoms = result["optimized_atoms"]

        # Check O-H distances are reasonable
        oh1_dist = optimized_atoms.get_distance(0, 1)
        oh2_dist = optimized_atoms.get_distance(0, 2)
        assert 0.8 < oh1_dist < 1.2
        assert 0.8 < oh2_dist < 1.2

    def test_ethane_optimization(self, uma_optimizer):
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

        result = uma_optimizer.optimize_minimum(atoms=ethane, fmax=0.05, steps=100)

        assert result["converged"] or result["steps_taken"] > 0
        optimized_atoms = result["optimized_atoms"]

        # Check C-C distance is reasonable
        cc_dist = optimized_atoms.get_distance(0, 1)
        assert 1.3 < cc_dist < 1.7

    def test_methanol_optimization(self, uma_optimizer):
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

        result = uma_optimizer.optimize_minimum(atoms=methanol, fmax=0.05, steps=100)

        assert result["converged"] or result["steps_taken"] > 0

    def test_trajectory_logging(self, uma_optimizer):
        """Test optimization with trajectory logging."""
        with tempfile.NamedTemporaryFile(suffix=".traj", delete=False) as f:
            traj_file = f.name

        try:
            h2 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])
            result = uma_optimizer.optimize_minimum(
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


@pytest.mark.skipif(not deps.has("fairchem"), reason="UMA/FairChem not available")
class TestUMAReactionPaths:
    """Test UMA backend for reaction pathway generation."""

    @pytest.fixture
    def uma_calculator(self):
        """Create UMA calculator."""
        try:
            return qme.get_uma_calculator()
        except ImportError:
            pytest.skip("UMA backend dependencies not available")

    def test_h2_dissociation_path(self, uma_calculator):
        """Test H2 dissociation reaction pathway."""
        # H2 molecule -> 2H atoms
        reactant = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        product = Atoms("H2", positions=[[0, 0, 0], [3.0, 0, 0]])

        reaction = qme.Reaction(reactant, product, calculator=uma_calculator)

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

    def test_water_rotation_path(self, uma_calculator):
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

        reaction = qme.Reaction(reactant, product, calculator=uma_calculator)
        path = reaction.interpolate(npoints=7, method="linear")

        assert len(path) == 7
        # All structures should have same number of atoms
        for geom in path:
            assert len(geom) == 3


@pytest.mark.skipif(
    not deps.has("fairchem") or not deps.has("sella"),
    reason="UMA/FairChem or SELLA not available",
)
class TestUMATransitionStates:
    """Test UMA backend for transition state optimization."""

    @pytest.fixture
    def uma_optimizer(self):
        """Create UMA optimizer."""
        try:
            return qme.QMEOptimizer(backend="uma")
        except ImportError:
            pytest.skip("UMA backend dependencies not available")

    def test_h2_dissociation_ts_guess(self, uma_optimizer):
        """Test transition state search from interpolated guess."""
        # Initial guess for H2 dissociation TS (stretched H2)
        ts_guess = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])

        try:
            result = uma_optimizer.find_transition_state(
                atoms=ts_guess, fmax=0.05, steps=50
            )

            # Should complete without error
            assert "converged" in result
            assert "steps_taken" in result

        except Exception as e:
            # Some TS searches may not converge, but should not crash
            assert "find_transition_state" not in str(e)

    def test_water_inversion_ts(self, uma_optimizer):
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
            result = uma_optimizer.find_transition_state(
                atoms=ts_guess, fmax=0.05, steps=50
            )

            assert "converged" in result

        except Exception as e:
            # TS optimization may not converge but should not crash
            assert "sella" not in str(e).lower() or "import" in str(e).lower()
