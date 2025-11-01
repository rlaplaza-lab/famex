"""Consolidated unit tests for optimization strategies with shared parametrized patterns."""

from __future__ import annotations

import warnings
from unittest.mock import patch

import numpy as np
import pytest

import qme
from qme.core.explorer import Explorer
from qme.strategies.cineb import MultiStructureCINEBStrategy
from qme.strategies.minima import LocalMinimaStrategy
from qme.strategies.minima_interpolate import MultiStructureMinimaInterpolateStrategy
from qme.strategies.neb import MultiStructureNEBStrategy
from qme.strategies.ts import LocalTSStrategy
from qme.strategies.ts_interpolate import MultiStructureTSGuessStrategy
from tests.test_utils import TestMoleculeFactory


# Shared fixtures for common test molecules
@pytest.fixture
def water_distorted():
    """Water molecule with distorted geometry."""
    return TestMoleculeFactory.get_water_distorted()


@pytest.fixture
def h2_stretched():
    """H2 molecule with stretched bond."""
    return TestMoleculeFactory.get_h2_stretched()


@pytest.fixture
def reactant_product_pair():
    """Pair of reactant and product structures for path/TS tests."""
    reactant = TestMoleculeFactory.get_water_distorted()
    product = TestMoleculeFactory.get_water_distorted()
    pos = product.get_positions()
    pos[1, 0] += 0.2
    product.set_positions(pos)
    return reactant, product


@pytest.fixture
def water_dissociation_ts_guess():
    """Water dissociation TS guess for TS strategy tests."""
    return TestMoleculeFactory.get_water_dissociation_ts_guess()


# Parametrized strategy initialization tests
@pytest.mark.parametrize(
    ("strategy_class", "atoms_fixture", "expected_attrs"),
    [
        (LocalMinimaStrategy, "water_distorted", {"profiler": None}),
        (LocalTSStrategy, "water_dissociation_ts_guess", {}),
        (
            MultiStructureNEBStrategy,
            "reactant_product_pair",
            {},
        ),
        (
            MultiStructureCINEBStrategy,
            "reactant_product_pair",
            {},
        ),
    ],
    ids=["minima", "ts", "neb", "cineb"],
)
def test_strategy_initialization(request, strategy_class, atoms_fixture, expected_attrs):
    """Test strategy initialization across all strategy types."""
    atoms = request.getfixturevalue(atoms_fixture)
    if isinstance(atoms, tuple):
        # For NEB/CINEB, atoms is a tuple of (reactant, product)
        explorer = Explorer(list(atoms), backend="mock")
    else:
        explorer = Explorer(atoms, backend="mock")

    strategy = strategy_class(explorer)

    assert strategy.explorer == explorer
    for attr, expected_value in expected_attrs.items():
        assert getattr(strategy, attr) == expected_value


# Parametrized basic run tests for single-structure strategies
@pytest.mark.parametrize(
    ("strategy_class", "atoms_getter", "expected_strategy_name", "skip_if_no_sella"),
    [
        (
            LocalMinimaStrategy,
            lambda: TestMoleculeFactory.get_h2_stretched(),
            "minima:local",
            False,
        ),
        (
            LocalTSStrategy,
            lambda: TestMoleculeFactory.get_water_dissociation_ts_guess(),
            "ts:local",
            True,
        ),
    ],
    ids=["minima", "ts"],
)
def test_strategy_run_basic(
    strategy_class,
    atoms_getter,
    expected_strategy_name,
    skip_if_no_sella,
):
    """Test basic run functionality for single-structure strategies."""
    if skip_if_no_sella and not qme.deps.has("sella"):
        pytest.skip("Sella required for TS optimization")

    atoms = atoms_getter()
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


# Parametrized frequency tests
@pytest.mark.parametrize(
    ("strategy_class", "atoms_getter", "skip_if_no_sella"),
    [
        (LocalMinimaStrategy, lambda: TestMoleculeFactory.get_water_distorted(), False),
        (
            LocalTSStrategy,
            lambda: TestMoleculeFactory.get_water_dissociation_ts_guess(),
            True,
        ),
    ],
    ids=["minima", "ts"],
)
def test_strategy_run_with_frequencies(strategy_class, atoms_getter, skip_if_no_sella):
    """Test run with frequency calculation for strategies that support it."""
    if skip_if_no_sella and not qme.deps.has("sella"):
        pytest.skip("Sella required for TS optimization")

    atoms = atoms_getter()
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


# Parametrized multi-structure interpolation tests
@pytest.mark.parametrize(
    ("strategy_class", "expected_strategy_name"),
    [
        (MultiStructureNEBStrategy, "path:neb"),
        (MultiStructureCINEBStrategy, "path:cineb"),
        (MultiStructureMinimaInterpolateStrategy, "minima:interpolate"),
    ],
    ids=["neb", "cineb", "minima_interpolate"],
)
def test_multi_structure_strategy_run_basic(
    reactant_product_pair, strategy_class, expected_strategy_name
):
    """Test basic run for multi-structure interpolation strategies."""
    reactant, product = reactant_product_pair
    explorer = Explorer([reactant, product], backend="mock")
    strategy = strategy_class(explorer)

    result = strategy.run([reactant, product], npoints=5, steps=5, fmax=0.5)

    assert "optimized_atoms" in result
    assert "strategy" in result
    assert result["strategy"] == expected_strategy_name
    assert isinstance(result["optimized_atoms"], list)


@pytest.mark.parametrize(
    ("strategy_class", "method", "skip_ts"),
    [
        (MultiStructureNEBStrategy, "linear", False),
        (MultiStructureNEBStrategy, "geodesic", False),
        (MultiStructureNEBStrategy, "idpp", False),
        (MultiStructureCINEBStrategy, "linear", False),
        (MultiStructureCINEBStrategy, "geodesic", False),
        (MultiStructureCINEBStrategy, "idpp", False),
        (MultiStructureMinimaInterpolateStrategy, "linear", False),
        (MultiStructureMinimaInterpolateStrategy, "geodesic", False),
        (MultiStructureMinimaInterpolateStrategy, "idpp", False),
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
    ],
)
def test_strategy_run_with_interpolation_methods(
    reactant_product_pair,
    strategy_class,
    method,
    skip_ts,
):
    """Test strategies with different interpolation methods."""
    reactant, product = reactant_product_pair
    explorer = Explorer([reactant, product], backend="mock")
    strategy = strategy_class(explorer)

    # TS interpolation strategies fail with mock backend
    if strategy_class == MultiStructureTSGuessStrategy:
        with pytest.raises(ValueError, match="not suitable for transition state"):
            strategy.run([reactant, product], npoints=5, method=method, steps=2, fmax=0.5)
    else:
        result = strategy.run([reactant, product], npoints=5, method=method, steps=3, fmax=0.5)
        assert result["optimized_atoms"] is not None
        assert isinstance(result["optimized_atoms"], list)


@pytest.mark.parametrize(
    "strategy_class",
    [MultiStructureNEBStrategy, MultiStructureCINEBStrategy],
    ids=["neb", "cineb"],
)
def test_path_strategy_validates_minimum_images(reactant_product_pair, strategy_class):
    """Test that path strategies validate minimum number of images."""
    reactant, product = reactant_product_pair
    explorer = Explorer([reactant, product], backend="mock")
    strategy = strategy_class(explorer)

    # npoints=2 gives only 2 images after interpolation, need at least 3
    with pytest.raises(ValueError, match="at least 3 images"):
        strategy.run([reactant, product], npoints=2, steps=3, fmax=0.5)


@pytest.mark.parametrize(
    "strategy_class",
    [MultiStructureNEBStrategy, MultiStructureCINEBStrategy],
    ids=["neb", "cineb"],
)
def test_path_strategy_filters_redundant_structures(reactant_product_pair, strategy_class):
    """Test that path strategies filter redundant structures."""
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
    reactant_product_pair,
    strategy_class,
    patch_module,
):
    """Test path strategies handle calculator attachment failures."""
    reactant, product = reactant_product_pair
    explorer = Explorer([reactant, product], backend="mock")
    strategy = strategy_class(explorer)

    with patch(patch_module) as mock_attach:
        mock_attach.side_effect = lambda explorer, path: None

        with pytest.raises(RuntimeError, match="Failed to attach calculators"):
            strategy.run([reactant, product], npoints=5, steps=3, fmax=0.5)


# Strategy-specific tests that don't fit the parametrized patterns
class TestLocalMinimaStrategySpecific:
    """Strategy-specific tests for LocalMinimaStrategy."""

    def test_run_with_different_optimizers(self):
        """Test run with different local optimizers."""
        atoms = TestMoleculeFactory.get_h2_stretched()
        explorer = Explorer(atoms, backend="mock")
        strategy = LocalMinimaStrategy(explorer)

        for optimizer in ["LBFGS", "BFGS"]:
            result = strategy.run([atoms], steps=3, fmax=0.5, local_optimizer_name=optimizer)
            assert result["optimized_atoms"] is not None

    def test_run_multiple_structures(self):
        """Test run with multiple structures."""
        atoms1 = TestMoleculeFactory.get_water_distorted()
        atoms2 = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer([atoms1, atoms2], backend="mock")
        strategy = LocalMinimaStrategy(explorer)

        result = strategy.run([atoms1, atoms2], steps=3, fmax=0.5)

        assert isinstance(result["optimized_atoms"], list)
        assert len(result["optimized_atoms"]) == 2
        assert isinstance(result["converged"], list)
        assert len(result["converged"]) == 2

    def test_run_single_structure_list(self):
        """Test run with single-element list."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")
        strategy = LocalMinimaStrategy(explorer)

        result = strategy.run([atoms], steps=3, fmax=0.5)

        assert isinstance(result["optimized_atoms"], type(atoms))
        assert isinstance(result["converged"], bool)

    def test_run_with_initial_hessian(self):
        """Test run with initial Hessian."""
        atoms = TestMoleculeFactory.get_water_distorted()
        n = len(atoms) * 3
        explorer = Explorer(atoms, backend="mock", initial_hessian=np.eye(n))
        strategy = LocalMinimaStrategy(explorer)

        result = strategy.run([atoms], steps=3, fmax=0.5, local_optimizer_name="LBFGS")
        assert result["optimized_atoms"] is not None

    def test_run_with_profiler(self):
        """Test run with profiler enabled."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock", profile=True)
        strategy = LocalMinimaStrategy(explorer, profiler=explorer.profiler)

        result = strategy.run([atoms], steps=3, fmax=0.5)

        assert "performance" in result
        assert result["optimized_atoms"] is not None

    def test_with_constraints(self):
        """Test strategy respects constraints."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock", constraints="fix 0")
        strategy = LocalMinimaStrategy(explorer)

        result = strategy.run([atoms], steps=3, fmax=0.5)
        assert result["optimized_atoms"] is not None

    def test_handles_optimizer_failure(self):
        """Test strategy handles optimizer failures gracefully."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")
        strategy = LocalMinimaStrategy(explorer)

        result = strategy.run([atoms], steps=2, fmax=1e-10)

        assert "optimized_atoms" in result
        assert "converged" in result


class TestLocalTSStrategySpecific:
    """Strategy-specific tests for LocalTSStrategy."""

    @pytest.mark.skipif(not qme.deps.has("sella"), reason="Sella required for TS optimization")
    def test_run_with_validation(self):
        """Test run with TS validation."""
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
        """Test that strategy validates backend supports TS optimization."""
        atoms = TestMoleculeFactory.get_water_dissociation_ts_guess()
        explorer = Explorer(atoms, backend="mock")
        strategy = LocalTSStrategy(explorer)

        with pytest.raises(ValueError, match="not suitable for transition state"):
            strategy.run([atoms], steps=3, fmax=0.5)

    def test_run_multiple_structures(self):
        """Test run with multiple structures."""
        atoms1 = TestMoleculeFactory.get_water_dissociation_ts_guess()
        atoms2 = TestMoleculeFactory.get_water_dissociation_ts_guess()
        explorer = Explorer([atoms1, atoms2], backend="mock")
        strategy = LocalTSStrategy(explorer)

        with pytest.raises(ValueError, match="not suitable"):
            strategy.run([atoms1, atoms2], steps=3, fmax=0.5)


class TestMultiStructureInterpolateStrategies:
    """Tests for multi-structure interpolation strategies."""

    @pytest.mark.parametrize(
        "strategy_class",
        [MultiStructureMinimaInterpolateStrategy, MultiStructureTSGuessStrategy],
        ids=["minima", "ts"],
    )
    def test_run_with_frequencies(self, reactant_product_pair, strategy_class):
        """Test run with frequency calculation."""
        reactant, product = reactant_product_pair
        explorer = Explorer([reactant, product], backend="mock")
        strategy = strategy_class(explorer)

        if strategy_class == MultiStructureTSGuessStrategy:
            # TS interpolation fails with mock backend
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
        """Test strategy handles individual structure optimization failures."""
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

    def test_minima_interpolate_requires_multiple_structures(self):
        """Test that minima interpolation requires multiple structures."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")
        strategy = MultiStructureMinimaInterpolateStrategy(explorer)

        with pytest.raises(ValueError, match="at least 2"):
            strategy.run([atoms], npoints=5, steps=2, fmax=0.5)

    def test_cineb_run_with_climb_false(self, reactant_product_pair):
        """Test CI-NEB run with climb=False."""
        reactant, product = reactant_product_pair
        explorer = Explorer([reactant, product], backend="mock")
        strategy = MultiStructureCINEBStrategy(explorer)

        result = strategy.run([reactant, product], npoints=5, climb=False, steps=3, fmax=0.5)
        assert result["optimized_atoms"] is not None

    def test_neb_validates_inputs(self):
        """Test that NEB strategy validates inputs."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")
        strategy = MultiStructureNEBStrategy(explorer)

        with pytest.raises(ValueError, match="No atoms provided"):
            strategy.run([])

    def test_neb_run_with_spring_constant(self, reactant_product_pair):
        """Test NEB run with custom spring constant."""
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

    def test_cineb_run_with_spring_constant(self, reactant_product_pair):
        """Test CI-NEB run with custom spring constant."""
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
