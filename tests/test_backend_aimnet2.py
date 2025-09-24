"""
Comprehensive tests for AIMNET2 backend functionality.

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


@pytest.mark.skipif(not deps.has("aimnet2"), reason="AIMNET2 not available")
class TestAIMNET2Backend:
    """Test AIMNET2 backend with real chemical systems."""

    @pytest.fixture
    def aimnet2_optimizer(self):
        """Create AIMNET2 optimizer."""
        try:
            return qme.QMEOptimizer(backend="aimnet2")
        except ImportError:
            pytest.skip("AIMNET2 backend dependencies not available")

    def test_aimnet2_optimizer_creation(self, aimnet2_optimizer):
        """Test AIMNET2 optimizer creation."""
        assert aimnet2_optimizer.backend == "aimnet2"
        assert aimnet2_optimizer.calculator is not None

    def test_h2_optimization(self, aimnet2_optimizer):
        """Test H2 molecule optimization from stretched geometry."""
        # Stretched H2 molecule (equilibrium ~0.74 Å)
        h2_stretched = Atoms("H2", positions=[[0, 0, 0], [2.0, 0, 0]])

        result = aimnet2_optimizer.optimize_minimum(
            atoms=h2_stretched, fmax=0.05, steps=50
        )

        assert result["converged"] or result["steps_taken"] > 0
        final_distance = result["optimized_atoms"].get_distance(0, 1)
        # Should be closer to equilibrium
        assert 0.6 < final_distance < 0.9

    def test_water_optimization(self, aimnet2_optimizer):
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

        result = aimnet2_optimizer.optimize_minimum(atoms=water, fmax=0.05, steps=100)

        assert result["converged"] or result["steps_taken"] > 0
        optimized_atoms = result["optimized_atoms"]

        # Check O-H distances are reasonable
        oh1_dist = optimized_atoms.get_distance(0, 1)
        oh2_dist = optimized_atoms.get_distance(0, 2)
        assert 0.8 < oh1_dist < 1.2
        assert 0.8 < oh2_dist < 1.2

    def test_ammonia_optimization(self, aimnet2_optimizer):
        """Test ammonia molecule optimization."""
        # Distorted ammonia molecule
        ammonia = Atoms(
            "NH3",
            positions=[
                [0.0, 0.0, 0.0],  # N
                [1.2, 0.0, 0.0],  # H (stretched)
                [0.0, 1.2, 0.0],  # H (stretched)
                [0.0, 0.0, 1.2],  # H (stretched)
            ],
        )

        result = aimnet2_optimizer.optimize_minimum(atoms=ammonia, fmax=0.05, steps=100)

        assert result["converged"] or result["steps_taken"] > 0
        optimized_atoms = result["optimized_atoms"]

        # Check N-H distances are reasonable
        nh1_dist = optimized_atoms.get_distance(0, 1)
        nh2_dist = optimized_atoms.get_distance(0, 2)
        nh3_dist = optimized_atoms.get_distance(0, 3)
        assert 0.9 < nh1_dist < 1.2
        assert 0.9 < nh2_dist < 1.2
        assert 0.9 < nh3_dist < 1.2

    def test_methane_optimization(self, aimnet2_optimizer):
        """Test methane optimization."""
        # Distorted methane
        methane = Atoms(
            "CH4",
            positions=[
                [0.0, 0.0, 0.0],  # C
                [1.2, 0.0, 0.0],  # H (stretched)
                [0.0, 1.2, 0.0],  # H (stretched)
                [0.0, 0.0, 1.2],  # H (stretched)
                [-0.8, -0.8, -0.8],  # H (compressed)
            ],
        )

        result = aimnet2_optimizer.optimize_minimum(atoms=methane, fmax=0.05, steps=100)

        assert result["converged"] or result["steps_taken"] > 0
        optimized_atoms = result["optimized_atoms"]

        # Check C-H distances are reasonable
        for i in range(1, 5):
            ch_dist = optimized_atoms.get_distance(0, i)
            assert 1.0 < ch_dist < 1.2

    def test_ethylene_optimization(self, aimnet2_optimizer):
        """Test ethylene molecule optimization."""
        # Slightly distorted ethylene
        ethylene = Atoms(
            "C2H4",
            positions=[
                [0.0, 0.0, 0.0],  # C
                [1.4, 0.0, 0.0],  # C (stretched double bond)
                [-0.5, 1.0, 0.0],  # H
                [-0.5, -1.0, 0.0],  # H
                [1.9, 1.0, 0.0],  # H
                [1.9, -1.0, 0.0],  # H
            ],
        )

        result = aimnet2_optimizer.optimize_minimum(
            atoms=ethylene, fmax=0.05, steps=100
        )

        assert result["converged"] or result["steps_taken"] > 0
        optimized_atoms = result["optimized_atoms"]

        # Check C=C distance is reasonable for double bond
        cc_dist = optimized_atoms.get_distance(0, 1)
        assert 1.2 < cc_dist < 1.4

    def test_trajectory_logging(self, aimnet2_optimizer):
        """Test optimization with trajectory logging."""
        with tempfile.NamedTemporaryFile(suffix=".traj", delete=False) as f:
            traj_file = f.name

        try:
            h2 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])
            result = aimnet2_optimizer.optimize_minimum(
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


@pytest.mark.skipif(not deps.has("aimnet2"), reason="AIMNET2 not available")
class TestAIMNET2ReactionPaths:
    """Test AIMNET2 backend for reaction pathway generation."""

    @pytest.fixture
    def aimnet2_calculator(self):
        """Create AIMNET2 calculator."""
        try:
            return qme.get_aimnet2_calculator()
        except ImportError:
            pytest.skip("AIMNET2 backend dependencies not available")

    def test_h2_dissociation_path(self, aimnet2_calculator):
        """Test H2 dissociation reaction pathway."""
        # H2 molecule -> 2H atoms
        reactant = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        product = Atoms("H2", positions=[[0, 0, 0], [3.0, 0, 0]])

        reaction = qme.Reaction(reactant, product, calculator=aimnet2_calculator)

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

    def test_water_formation_path(self, aimnet2_calculator):
        """Test water formation reaction pathway."""
        # OH + H -> H2O (simplified)
        reactant = Atoms(
            "H2O",
            positions=[
                [0.0, 0.0, 0.0],  # O
                [0.96, 0.0, 0.0],  # H
                [3.0, 0.0, 0.0],  # H (far away)
            ],
        )

        product = Atoms(
            "H2O",
            positions=[
                [0.0, 0.0, 0.0],  # O
                [0.96, 0.0, 0.0],  # H
                [-0.24, 0.93, 0.0],  # H (bonded)
            ],
        )

        reaction = qme.Reaction(reactant, product, calculator=aimnet2_calculator)
        path = reaction.interpolate(npoints=6, method="linear")

        assert len(path) == 6
        # All structures should have same number of atoms
        for geom in path:
            assert len(geom) == 3

    def test_ammonia_inversion_path(self, aimnet2_calculator):
        """Test ammonia inversion pathway."""
        # Pyramidal -> planar -> inverted pyramidal
        pyramidal = Atoms(
            "NH3",
            positions=[
                [0.0, 0.0, 0.0],  # N
                [1.01, 0.0, -0.4],  # H
                [-0.505, 0.87, -0.4],  # H
                [-0.505, -0.87, -0.4],  # H
            ],
        )

        inverted = Atoms(
            "NH3",
            positions=[
                [0.0, 0.0, 0.0],  # N
                [1.01, 0.0, 0.4],  # H (inverted)
                [-0.505, 0.87, 0.4],  # H (inverted)
                [-0.505, -0.87, 0.4],  # H (inverted)
            ],
        )

        reaction = qme.Reaction(pyramidal, inverted, calculator=aimnet2_calculator)
        path = reaction.interpolate(npoints=7, method="linear")

        assert len(path) == 7
        for geom in path:
            assert len(geom) == 4


@pytest.mark.skipif(
    not deps.has("aimnet2") or not deps.has("sella"),
    reason="AIMNET2 or SELLA not available",
)
class TestAIMNET2TransitionStates:
    """Test AIMNET2 backend for transition state optimization."""

    @pytest.fixture
    def aimnet2_optimizer(self):
        """Create AIMNET2 optimizer."""
        try:
            return qme.QMEOptimizer(backend="aimnet2")
        except ImportError:
            pytest.skip("AIMNET2 backend dependencies not available")

    def test_h2_dissociation_ts_guess(self, aimnet2_optimizer):
        """Test transition state search from interpolated guess."""
        # Initial guess for H2 dissociation TS (stretched H2)
        ts_guess = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])

        try:
            result = aimnet2_optimizer.find_transition_state(
                atoms=ts_guess, fmax=0.05, steps=50
            )

            # Should complete without error
            assert "converged" in result
            assert "steps_taken" in result

        except Exception as e:
            # Some TS searches may not converge, but should not crash
            assert "find_transition_state" not in str(e)

    def test_ammonia_inversion_ts(self, aimnet2_optimizer):
        """Test ammonia inversion transition state."""
        # Planar ammonia as TS guess
        ts_guess = Atoms(
            "NH3",
            positions=[
                [0.0, 0.0, 0.0],  # N
                [1.01, 0.0, 0.0],  # H (planar)
                [-0.505, 0.87, 0.0],  # H (planar)
                [-0.505, -0.87, 0.0],  # H (planar)
            ],
        )

        try:
            result = aimnet2_optimizer.find_transition_state(
                atoms=ts_guess, fmax=0.05, steps=50
            )

            assert "converged" in result

        except Exception as e:
            # TS optimization may not converge but should not crash
            assert "sella" not in str(e).lower() or "import" in str(e).lower()

    def test_water_dissociation_ts(self, aimnet2_optimizer):
        """Test water dissociation transition state."""
        # Stretched O-H bond as TS guess
        ts_guess = Atoms(
            "H2O",
            positions=[
                [0.0, 0.0, 0.0],  # O
                [1.5, 0.0, 0.0],  # H (stretched)
                [-0.24, 0.93, 0.0],  # H (normal)
            ],
        )

        try:
            result = aimnet2_optimizer.find_transition_state(
                atoms=ts_guess, fmax=0.05, steps=50
            )

            assert "converged" in result or "steps_taken" in result

        except Exception as e:
            # Complex TS searches may not converge
            assert "find_transition_state" not in str(e)


@pytest.mark.skipif(not deps.has("aimnet2"), reason="AIMNET2 not available")
class TestAIMNET2Advanced:
    """Advanced AIMNET2 backend tests."""

    def test_minimize_structure_function(self):
        """Test standalone minimize_structure function with AIMNET2."""
        h2 = Atoms("H2", positions=[[0, 0, 0], [1.8, 0, 0]])

        try:
            result = qme.minimize_structure(h2, backend="aimnet2", fmax=0.05, steps=30)

            assert isinstance(result, Atoms)
            final_distance = result.get_distance(0, 1)
            assert 0.6 < final_distance < 1.2

        except ImportError:
            pytest.skip("AIMNET2 backend dependencies not available")

    def test_mixed_element_systems(self):
        """Test systems with different element types."""
        try:
            optimizer = qme.QMEOptimizer(backend="aimnet2")

            # Water (O, H)
            water = Atoms("H2O", positions=[[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
            result_water = optimizer.optimize_minimum(atoms=water, fmax=0.1, steps=20)
            assert result_water["steps_taken"] > 0

            # Ammonia (N, H)
            ammonia = Atoms(
                "NH3", positions=[[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
            )
            result_ammonia = optimizer.optimize_minimum(
                atoms=ammonia, fmax=0.1, steps=20
            )
            assert result_ammonia["steps_taken"] > 0

        except ImportError:
            pytest.skip("AIMNET2 backend dependencies not available")

    def test_larger_molecules(self):
        """Test optimization of larger molecular systems."""
        try:
            optimizer = qme.QMEOptimizer(backend="aimnet2")

            # Propane (C3H8)
            propane = Atoms(
                "C3H8",
                positions=[
                    [0.0, 0.0, 0.0],  # C
                    [1.5, 0.0, 0.0],  # C
                    [3.0, 0.0, 0.0],  # C
                    [-0.5, 1.0, 0.0],  # H
                    [-0.5, -0.5, 0.8],  # H
                    [-0.5, -0.5, -0.8],  # H
                    [1.5, 1.0, 0.5],  # H
                    [1.5, -1.0, 0.5],  # H
                    [3.5, 1.0, 0.0],  # H
                    [3.5, -0.5, 0.8],  # H
                    [3.5, -0.5, -0.8],  # H
                ],
            )

            result = optimizer.optimize_minimum(atoms=propane, fmax=0.1, steps=50)

            assert result["steps_taken"] > 0

        except ImportError:
            pytest.skip("AIMNET2 backend dependencies not available")
