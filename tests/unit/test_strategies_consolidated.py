from __future__ import annotations

import warnings
from unittest.mock import patch

import numpy as np
import pytest

import qme
from qme.core.explorer import Explorer
from qme.strategies.cineb import MultiStructureCINEBStrategy
from qme.strategies.irc import LocalIRCStrategy
from qme.strategies.minima import LocalMinimaStrategy
from qme.strategies.minima_interpolate import MultiStructureMinimaInterpolateStrategy
from qme.strategies.neb import MultiStructureNEBStrategy
from qme.strategies.path_interpolate import PathInterpolateStrategy
from qme.strategies.ts import LocalTSStrategy
from qme.strategies.ts_interpolate import MultiStructureTSGuessStrategy
from tests.test_utils import TestMoleculeFactory

# ============================================================================
# Shared Fixtures
# ============================================================================
# Note: water_molecule, h2_molecule, and reactant_product_pair are available
# from conftest.py. Only test-specific fixtures are defined here.


@pytest.fixture
def water_dissociation_ts_guess():
    return TestMoleculeFactory.get_water_dissociation_ts_guess()


# ============================================================================
# Strategy Initialization Tests
# ============================================================================


class TestStrategyInitialization:
    @pytest.mark.parametrize(
        ("strategy_class", "atoms_fixture", "expected_attrs"),
        [
            (LocalMinimaStrategy, "water_molecule", {"profiler": None}),
            (LocalTSStrategy, "water_dissociation_ts_guess", {}),
            (LocalIRCStrategy, "water_dissociation_ts_guess", {}),
            (MultiStructureNEBStrategy, "reactant_product_pair", {}),
            (MultiStructureCINEBStrategy, "reactant_product_pair", {}),
            (PathInterpolateStrategy, "reactant_product_pair", {}),
        ],
        ids=["minima", "ts", "irc", "neb", "cineb", "path_interpolate"],
    )
    def test_strategy_initialization(self, request, strategy_class, atoms_fixture, expected_attrs):
        atoms = request.getfixturevalue(atoms_fixture)
        if isinstance(atoms, tuple):
            explorer = Explorer(list(atoms), backend="mock")
        else:
            explorer = Explorer(atoms, backend="mock")

        strategy = strategy_class(explorer)

        assert strategy.explorer == explorer
        for attr, expected_value in expected_attrs.items():
            assert getattr(strategy, attr) == expected_value


# ============================================================================
# Single-Structure Strategy Tests
# ============================================================================


class TestSingleStructureStrategies:
    @pytest.mark.parametrize(
        ("strategy_class", "atoms_fixture", "expected_strategy_name", "skip_if_no_sella"),
        [
            (
                LocalMinimaStrategy,
                "h2_molecule",
                "minima:local",
                False,
            ),
            (
                LocalTSStrategy,
                "water_dissociation_ts_guess",
                "ts:local",
                True,
            ),
            (
                LocalIRCStrategy,
                "water_dissociation_ts_guess",
                "path:irc",
                False,
            ),
        ],
        ids=["minima", "ts", "irc"],
    )
    def test_strategy_run_basic(
        self,
        request,
        strategy_class,
        atoms_fixture,
        expected_strategy_name,
        skip_if_no_sella,
    ):
        if skip_if_no_sella and not qme.deps.has("sella"):
            pytest.skip("Sella required for TS optimization")

        atoms = request.getfixturevalue(atoms_fixture)
        explorer = Explorer(atoms, backend="mock")
        strategy = strategy_class(explorer)

        try:
            result = strategy.run([atoms], steps=5, fmax=0.5)
            assert "optimized_atoms" in result
            assert "strategy" in result
            assert result["strategy"] == expected_strategy_name
        except ValueError as e:
            if "not suitable" in str(e).lower():
                pytest.skip(f"Mock backend doesn't support {strategy_class.__name__}")
            raise

    @pytest.mark.parametrize(
        ("strategy_class", "atoms_fixture", "skip_if_no_sella"),
        [
            (LocalMinimaStrategy, "water_molecule", False),
            (
                LocalTSStrategy,
                "water_dissociation_ts_guess",
                True,
            ),
        ],
        ids=["minima", "ts"],
    )
    def test_strategy_run_with_frequencies(
        self, request, strategy_class, atoms_fixture, skip_if_no_sella
    ):
        if skip_if_no_sella and not qme.deps.has("sella"):
            pytest.skip("Sella required for TS optimization")

        atoms = request.getfixturevalue(atoms_fixture)
        explorer = Explorer(atoms, backend="mock")
        strategy = strategy_class(explorer)

        try:
            result = strategy.run([atoms], steps=3, fmax=0.5, calculate_frequencies=True)
            assert "frequency_analysis" in result
            if strategy_class == LocalMinimaStrategy:
                assert "is_minimum" in result
                assert "free_energy_correction" in result
            else:
                assert "is_ts" in result
        except ValueError as e:
            if "not suitable" in str(e).lower():
                pytest.skip(f"Mock backend doesn't support {strategy_class.__name__}")
            raise


# ============================================================================
# Multi-Structure Strategy Tests
# ============================================================================


class TestMultiStructureStrategies:
    @pytest.mark.parametrize(
        ("strategy_class", "expected_strategy_name"),
        [
            (MultiStructureNEBStrategy, "path:neb"),
            (MultiStructureCINEBStrategy, "path:cineb"),
            (MultiStructureMinimaInterpolateStrategy, "minima:interpolate"),
            (PathInterpolateStrategy, "path:interpolate"),
        ],
        ids=["neb", "cineb", "minima_interpolate", "path_interpolate"],
    )
    def test_multi_structure_strategy_run_basic(
        self, reactant_product_pair, strategy_class, expected_strategy_name
    ):
        reactant, product = reactant_product_pair
        explorer = Explorer([reactant, product], backend="mock")
        strategy = strategy_class(explorer)

        # PathInterpolateStrategy doesn't need steps/fmax, others do
        if strategy_class == PathInterpolateStrategy:
            result = strategy.run([reactant, product], npoints=5)
        else:
            result = strategy.run([reactant, product], npoints=5, steps=5, fmax=0.5)

        assert "optimized_atoms" in result
        assert "strategy" in result
        assert result["strategy"] == expected_strategy_name
        assert isinstance(result["optimized_atoms"], list)

    @pytest.mark.parametrize(
        ("strategy_class", "method"),
        [
            (MultiStructureNEBStrategy, "linear"),
            (MultiStructureNEBStrategy, "geodesic"),
            (MultiStructureNEBStrategy, "idpp"),
            (MultiStructureCINEBStrategy, "linear"),
            (MultiStructureCINEBStrategy, "geodesic"),
            (MultiStructureCINEBStrategy, "idpp"),
            (MultiStructureMinimaInterpolateStrategy, "linear"),
            (MultiStructureMinimaInterpolateStrategy, "geodesic"),
            (MultiStructureMinimaInterpolateStrategy, "idpp"),
            (PathInterpolateStrategy, "linear"),
            (PathInterpolateStrategy, "geodesic"),
            (PathInterpolateStrategy, "idpp"),
        ],
        ids=[
            "neb_linear",
            "neb_geodesic",
            "neb_idpp",
            "cineb_linear",
            "cineb_geodesic",
            "cineb_idpp",
            "minima_interpolate_linear",
            "minima_interpolate_geodesic",
            "minima_interpolate_idpp",
            "path_interpolate_linear",
            "path_interpolate_geodesic",
            "path_interpolate_idpp",
        ],
    )
    def test_strategy_run_with_interpolation_methods(
        self, reactant_product_pair, strategy_class, method
    ):
        reactant, product = reactant_product_pair
        explorer = Explorer([reactant, product], backend="mock")
        strategy = strategy_class(explorer)

        if strategy_class == MultiStructureTSGuessStrategy:
            with pytest.raises(ValueError, match="not suitable for transition state"):
                strategy.run([reactant, product], npoints=5, method=method, steps=2, fmax=0.5)
        elif strategy_class == PathInterpolateStrategy:
            # PathInterpolateStrategy doesn't need steps/fmax
            result = strategy.run([reactant, product], npoints=5, method=method)
            assert result["optimized_atoms"] is not None
            assert isinstance(result["optimized_atoms"], list)
        else:
            result = strategy.run([reactant, product], npoints=5, method=method, steps=3, fmax=0.5)
            assert result["optimized_atoms"] is not None
            assert isinstance(result["optimized_atoms"], list)

    @pytest.mark.parametrize(
        "strategy_class",
        [MultiStructureNEBStrategy, MultiStructureCINEBStrategy],
        ids=["neb", "cineb"],
    )
    def test_path_strategy_validates_minimum_images(self, reactant_product_pair, strategy_class):
        reactant, product = reactant_product_pair
        explorer = Explorer([reactant, product], backend="mock")
        strategy = strategy_class(explorer)

        with pytest.raises(ValueError, match="at least 3 images"):
            strategy.run([reactant, product], npoints=2, steps=3, fmax=0.5)

    @pytest.mark.parametrize(
        "strategy_class",
        [MultiStructureNEBStrategy, MultiStructureCINEBStrategy],
        ids=["neb", "cineb"],
    )
    def test_path_strategy_filters_redundant_structures(
        self, reactant_product_pair, strategy_class
    ):
        reactant, product = reactant_product_pair
        explorer = Explorer([reactant, product], backend="mock")
        strategy = strategy_class(explorer)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = strategy.run(
                [reactant, product],
                npoints=5,
                steps=3,
                fmax=0.5,
                rmsd_threshold=0.1,
                energy_threshold=0.001,
            )
            assert result["optimized_atoms"] is not None

    @pytest.mark.parametrize(
        ("strategy_class", "patch_module"),
        [
            (MultiStructureNEBStrategy, "qme.strategies.neb.PathManager.attach_calculators"),
            (MultiStructureCINEBStrategy, "qme.strategies.cineb.PathManager.attach_calculators"),
        ],
        ids=["neb", "cineb"],
    )
    def test_path_strategy_handles_calculator_attachment_failure(
        self, reactant_product_pair, strategy_class, patch_module
    ):
        reactant, product = reactant_product_pair
        explorer = Explorer([reactant, product], backend="mock")
        strategy = strategy_class(explorer)

        with patch(patch_module) as mock_attach:
            mock_attach.side_effect = lambda explorer, path: None

            with pytest.raises(RuntimeError, match="Failed to attach calculators"):
                strategy.run([reactant, product], npoints=5, steps=3, fmax=0.5)


# ============================================================================
# Strategy-Specific Tests
# ============================================================================


class TestLocalMinimaStrategy:
    def test_run_with_different_optimizers(self, h2_molecule):
        atoms = h2_molecule
        explorer = Explorer(atoms, backend="mock")
        strategy = LocalMinimaStrategy(explorer)

        for optimizer in ["LBFGS", "BFGS"]:
            result = strategy.run([atoms], steps=3, fmax=0.5, local_optimizer_name=optimizer)
            assert result["optimized_atoms"] is not None

    def test_run_multiple_structures(self, water_molecule):
        atoms1 = water_molecule
        atoms2 = water_molecule.copy()
        explorer = Explorer([atoms1, atoms2], backend="mock")
        strategy = LocalMinimaStrategy(explorer)

        result = strategy.run([atoms1, atoms2], steps=3, fmax=0.5)

        assert isinstance(result["optimized_atoms"], list)
        assert len(result["optimized_atoms"]) == 2
        assert isinstance(result["converged"], list)
        assert len(result["converged"]) == 2

    def test_run_single_structure_list(self, water_molecule):
        atoms = water_molecule
        explorer = Explorer(atoms, backend="mock")
        strategy = LocalMinimaStrategy(explorer)

        result = strategy.run([atoms], steps=3, fmax=0.5)

        assert isinstance(result["optimized_atoms"], type(atoms))
        assert isinstance(result["converged"], bool)

    def test_run_with_initial_hessian(self, water_molecule):
        atoms = water_molecule
        n = len(atoms) * 3
        explorer = Explorer(atoms, backend="mock", initial_hessian=np.eye(n))
        strategy = LocalMinimaStrategy(explorer)

        result = strategy.run([atoms], steps=3, fmax=0.5, local_optimizer_name="LBFGS")
        assert result["optimized_atoms"] is not None

    def test_run_with_profiler(self, water_molecule):
        atoms = water_molecule
        explorer = Explorer(atoms, backend="mock", profile=True)
        strategy = LocalMinimaStrategy(explorer, profiler=explorer.profiler)

        result = strategy.run([atoms], steps=3, fmax=0.5)

        assert "performance" in result
        assert result["optimized_atoms"] is not None

    def test_with_constraints(self, water_molecule):
        atoms = water_molecule
        explorer = Explorer(atoms, backend="mock", constraints="fix 0")
        strategy = LocalMinimaStrategy(explorer)

        result = strategy.run([atoms], steps=3, fmax=0.5)
        assert result["optimized_atoms"] is not None

    def test_handles_optimizer_failure(self, water_molecule):
        atoms = water_molecule
        explorer = Explorer(atoms, backend="mock")
        strategy = LocalMinimaStrategy(explorer)

        result = strategy.run([atoms], steps=2, fmax=1e-10)

        assert "optimized_atoms" in result
        assert "converged" in result


class TestLocalTSStrategy:
    @pytest.mark.skipif(not qme.deps.has("sella"), reason="Sella required for TS optimization")
    def test_run_with_validation(self):
        atoms = TestMoleculeFactory.get_water_dissociation_ts_guess()
        explorer = Explorer(atoms, backend="mock")
        strategy = LocalTSStrategy(explorer)

        try:
            result = strategy.run([atoms], steps=3, fmax=0.5, validate_ts=True)
            assert "ts_validation" in result
            assert "is_ts" in result
        except ValueError as e:
            if "not suitable" in str(e).lower():
                pytest.skip("Mock backend doesn't support TS optimization")
            raise

    def test_validates_backend(self):
        atoms = TestMoleculeFactory.get_water_dissociation_ts_guess()
        explorer = Explorer(atoms, backend="mock")
        strategy = LocalTSStrategy(explorer)

        with pytest.raises(ValueError, match="not suitable for transition state"):
            strategy.run([atoms], steps=3, fmax=0.5)

    def test_run_multiple_structures(self):
        atoms1 = TestMoleculeFactory.get_water_dissociation_ts_guess()
        atoms2 = TestMoleculeFactory.get_water_dissociation_ts_guess()
        explorer = Explorer([atoms1, atoms2], backend="mock")
        strategy = LocalTSStrategy(explorer)

        with pytest.raises(ValueError, match="not suitable"):
            strategy.run([atoms1, atoms2], steps=3, fmax=0.5)


class TestMultiStructureInterpolateStrategies:
    @pytest.mark.parametrize(
        "strategy_class",
        [MultiStructureMinimaInterpolateStrategy, MultiStructureTSGuessStrategy],
        ids=["minima", "ts"],
    )
    def test_run_with_frequencies(self, reactant_product_pair, strategy_class):
        reactant, product = reactant_product_pair
        explorer = Explorer([reactant, product], backend="mock")
        strategy = strategy_class(explorer)

        if strategy_class == MultiStructureTSGuessStrategy:
            with pytest.raises(ValueError, match="not suitable for transition state"):
                strategy.run(
                    [reactant, product],
                    npoints=5,
                    steps=2,
                    fmax=0.5,
                    calculate_frequencies=True,
                )
        else:
            result = strategy.run(
                [reactant, product],
                npoints=5,
                steps=2,
                fmax=0.5,
                calculate_frequencies=True,
            )
            assert "frequency_analysis" in result
            assert "is_minimum" in result

    def test_minima_interpolate_handles_optimization_failure(self, reactant_product_pair):
        reactant, product = reactant_product_pair
        explorer = Explorer([reactant, product], backend="mock")
        strategy = MultiStructureMinimaInterpolateStrategy(explorer)

        with patch.object(LocalMinimaStrategy, "run") as mock_run:
            mock_run.side_effect = [
                {"optimized_atoms": reactant, "converged": True},
                ValueError("Error"),
            ]

            result = strategy.run([reactant, product], npoints=3, steps=2, fmax=0.5)

            assert "optimized_atoms" in result
            assert isinstance(result["optimized_atoms"], list)
            assert len(result["optimized_atoms"]) >= 2

    def test_minima_interpolate_requires_multiple_structures(self, water_molecule):
        atoms = water_molecule
        explorer = Explorer(atoms, backend="mock")
        strategy = MultiStructureMinimaInterpolateStrategy(explorer)

        with pytest.raises(ValueError, match="at least 2"):
            strategy.run([atoms], npoints=5, steps=2, fmax=0.5)

    def test_cineb_run_with_climb_false(self, reactant_product_pair):
        reactant, product = reactant_product_pair
        explorer = Explorer([reactant, product], backend="mock")
        strategy = MultiStructureCINEBStrategy(explorer)

        result = strategy.run([reactant, product], npoints=5, climb=False, steps=3, fmax=0.5)
        assert result["optimized_atoms"] is not None

    def test_neb_validates_inputs(self, water_molecule):
        atoms = water_molecule
        explorer = Explorer(atoms, backend="mock")
        strategy = MultiStructureNEBStrategy(explorer)

        with pytest.raises(ValueError, match="No atoms provided"):
            strategy.run([])

    def test_neb_run_with_spring_constant(self, reactant_product_pair):
        reactant, product = reactant_product_pair
        explorer = Explorer([reactant, product], backend="mock")
        strategy = MultiStructureNEBStrategy(explorer)

        result = strategy.run(
            [reactant, product],
            npoints=5,
            spring_constant=10.0,
            steps=3,
            fmax=0.5,
        )
        assert result["optimized_atoms"] is not None

    def test_neb_path_length_exactly_3(self, reactant_product_pair):
        """Test NEB works with exactly 3 images (minimum required)."""
        reactant, product = reactant_product_pair
        explorer = Explorer([reactant, product], backend="mock")
        strategy = MultiStructureNEBStrategy(explorer)

        # Should work with npoints=3 (minimum required)
        result = strategy.run([reactant, product], npoints=3, steps=3, fmax=0.5)
        assert result["optimized_atoms"] is not None

    def test_neb_nested_path_flattening(self, reactant_product_pair):
        """Test NEB handles nested path structure and flattens it."""
        from unittest.mock import patch

        reactant, product = reactant_product_pair
        explorer = Explorer([reactant, product], backend="mock")
        strategy = MultiStructureNEBStrategy(explorer)

        # Create a path that could appear nested
        flat_path = [reactant, product, reactant]  # 3+ atoms for valid path

        with patch("qme.strategies.neb.PathManager.interpolate") as mock_interpolate:
            mock_interpolate.return_value = flat_path

            # Should handle the path structure
            result = strategy.run([reactant, product], npoints=5, steps=3, fmax=0.5)
            assert result is not None

    def test_neb_calculator_attachment_failure_detailed(self, reactant_product_pair):
        """Test NEB raises RuntimeError when calculator attachment fails."""
        from unittest.mock import patch

        reactant, product = reactant_product_pair
        explorer = Explorer([reactant, product], backend="mock")
        strategy = MultiStructureNEBStrategy(explorer)

        with patch("qme.strategies.neb.PathManager.attach_calculators") as mock_attach:
            # Simulate attachment that doesn't attach calculators
            def mock_attach_func(explorer, path):
                # Don't attach calculators - leave calc as None
                for img in path:
                    img.calc = None

            mock_attach.side_effect = mock_attach_func

            with pytest.raises(RuntimeError, match="Failed to attach calculators"):
                strategy.run([reactant, product], npoints=5, steps=3, fmax=0.5)

    def test_cineb_run_with_spring_constant(self, reactant_product_pair):
        reactant, product = reactant_product_pair
        explorer = Explorer([reactant, product], backend="mock")
        strategy = MultiStructureCINEBStrategy(explorer)

        result = strategy.run(
            [reactant, product],
            npoints=5,
            spring_constant=10.0,
            steps=3,
            fmax=0.5,
        )
        assert result["optimized_atoms"] is not None


class TestLocalIRCStrategy:
    def test_irc_basic_functionality(self):
        atoms = TestMoleculeFactory.get_water_dissociation_ts_guess()
        explorer = Explorer(atoms, backend="mock")
        strategy = LocalIRCStrategy(explorer)

        result = strategy.run([atoms], steps=5, step_size=0.1, fmax=0.5, direction="both")

        assert "trajectory" in result
        assert "strategy" in result
        assert result["strategy"] == "path:irc"
        assert isinstance(result["trajectory"], list)
        assert len(result["trajectory"]) > 0

    @pytest.mark.parametrize(
        "direction", ["forward", "backward", "both"], ids=["forward", "backward", "both"]
    )
    def test_irc_directions(self, direction):
        atoms = TestMoleculeFactory.get_water_dissociation_ts_guess()
        explorer = Explorer(atoms, backend="mock")
        strategy = LocalIRCStrategy(explorer)

        result = strategy.run([atoms], steps=5, step_size=0.1, fmax=0.5, direction=direction)

        assert "trajectory" in result
        if direction == "forward":
            assert "forward_path" in result
        elif direction == "backward":
            assert "backward_path" in result

    def test_irc_requires_single_structure(self):
        atoms1 = TestMoleculeFactory.get_water_dissociation_ts_guess()
        atoms2 = TestMoleculeFactory.get_water_dissociation_ts_guess()
        explorer = Explorer([atoms1, atoms2], backend="mock")
        strategy = LocalIRCStrategy(explorer)

        with pytest.raises(ValueError, match="single structure"):
            strategy.run([atoms1, atoms2], steps=5, step_size=0.1, fmax=0.5)


class TestPathInterpolateStrategy:
    def test_path_interpolate_basic(self, reactant_product_pair):
        reactant, product = reactant_product_pair
        explorer = Explorer([reactant, product], backend="mock")
        strategy = PathInterpolateStrategy(explorer)

        result = strategy.run([reactant, product], npoints=5, method="linear")

        assert "optimized_atoms" in result
        assert "strategy" in result
        assert result["strategy"] == "path:interpolate"
        assert isinstance(result["optimized_atoms"], list)
        assert len(result["optimized_atoms"]) == 5
        assert result["converged"] is True

    @pytest.mark.parametrize(
        "method", ["linear", "geodesic", "idpp"], ids=["linear", "geodesic", "idpp"]
    )
    def test_path_interpolate_methods(self, reactant_product_pair, method):
        reactant, product = reactant_product_pair
        explorer = Explorer([reactant, product], backend="mock")
        strategy = PathInterpolateStrategy(explorer)

        result = strategy.run([reactant, product], npoints=5, method=method)

        assert isinstance(result["optimized_atoms"], list)
        assert len(result["optimized_atoms"]) == 5

    def test_path_interpolate_requires_multiple_structures(self, water_molecule):
        atoms = water_molecule
        explorer = Explorer(atoms, backend="mock")
        strategy = PathInterpolateStrategy(explorer)

        with pytest.raises(ValueError, match="at least 2"):
            strategy.run([atoms], npoints=5)
