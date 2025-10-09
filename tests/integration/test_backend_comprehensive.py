"""
Comprehensive backend testing for QME.

This module consolidates all backend testing functionality including:
- Minima optimization across all backends
- Transition state optimization across all backends
- NEB optimization across all backends
- CLI functionality testing
- Geometric optimizer integration
- Performance comparisons

Test systems include:
- H2 molecule (simple diatomic)
- H2O molecule (bent triatomic)
- CH4 molecule (tetrahedral)
- Water dissociation pathway
- SN2-like reaction coordinate
"""

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from click.testing import CliRunner

from qme import Explorer
from qme.backend_availability import get_available_backends
from qme.cli import main
from qme.dependencies import deps
from tests.test_utils import (
    BackendTestMixin,
    StandardTestAssertions,
    TestMoleculeFactory,
    TestResultHandler,
)


class TestBackendMinimaOptimization:
    """Test minima optimization across all available backends."""

    @pytest.fixture(params=get_available_backends(include_mock=False))
    def backend(self, request):
        """Parametrized fixture for available backends."""
        return request.param

    def test_h2_optimization(self, backend):
        """Test H2 optimization across all backends."""
        h2 = TestMoleculeFactory.get_h2_stretched()
        optimizer = Explorer(atoms=h2, backend=backend)

        start_time = time.time()
        result = optimizer.run(mode="minima", local_optimizer_name="BFGS", fmax=0.05, steps=20)
        optimization_time = time.time() - start_time

        # Use standardized result handling
        strategy_result = TestResultHandler.normalize_result(result)
        final_atoms = TestResultHandler.extract_atoms(result)

        # Use standardized assertions
        StandardTestAssertions.assert_optimization_result(strategy_result)
        StandardTestAssertions.assert_reasonable_geometry(final_atoms, backend)

        # Check H-H distance
        final_distance = final_atoms.get_distance(0, 1)
        assert 0.6 < final_distance < 1.2

        print(f"Backend {backend}: H2 optimization took {optimization_time:.3f}s")

    def test_water_optimization(self, backend):
        """Test H2O optimization across all backends."""
        water = TestMoleculeFactory.get_water_distorted()
        optimizer = Explorer(atoms=water, backend=backend)

        start_time = time.time()
        result = optimizer.run(mode="minima", local_optimizer_name="BFGS", fmax=0.05, steps=50)
        optimization_time = time.time() - start_time

        # Use standardized result handling
        strategy_result = TestResultHandler.normalize_result(result)
        final_atoms = TestResultHandler.extract_atoms(result)

        # Use standardized assertions
        StandardTestAssertions.assert_optimization_result(strategy_result)
        StandardTestAssertions.assert_reasonable_geometry(final_atoms, backend)

        # Check O-H distances
        oh1_dist = final_atoms.get_distance(0, 1)
        oh2_dist = final_atoms.get_distance(0, 2)

        assert 0.85 < oh1_dist < 1.1
        assert 0.85 < oh2_dist < 1.1

        print(
            f"Backend {backend}: H2O optimization took {optimization_time:.3f}s, "
            f"{strategy_result['steps_taken']} steps, O-H distances: {oh1_dist:.3f}, "
            f"{oh2_dist:.3f} Å"
        )

    def test_methane_optimization(self, backend):
        """Test CH4 optimization across all backends."""
        methane = TestMoleculeFactory.get_methane_distorted()
        optimizer = Explorer(atoms=methane, backend=backend)

        start_time = time.time()
        result = optimizer.run(mode="minima", local_optimizer_name="BFGS", fmax=0.05, steps=50)
        optimization_time = time.time() - start_time

        # Use standardized result handling
        strategy_result = TestResultHandler.normalize_result(result)
        final_atoms = TestResultHandler.extract_atoms(result)

        # Use standardized assertions
        StandardTestAssertions.assert_optimization_result(strategy_result)
        StandardTestAssertions.assert_reasonable_geometry(final_atoms, backend)

        # Check C-H distances
        ch_distances = [final_atoms.get_distance(0, i) for i in range(1, 5)]

        for dist in ch_distances:
            assert 0.95 < dist < 1.25

        print(
            f"Backend {backend}: CH4 optimization took {optimization_time:.3f}s, "
            f"{strategy_result['steps_taken']} steps, avg C-H distance: "
            f"{np.mean(ch_distances):.3f} Å"
        )


class TestBackendTransitionStateOptimization:
    """Test transition state optimization across all available backends."""

    @pytest.fixture(params=get_available_backends(include_mock=False))
    def backend(self, request):
        """Parametrized fixture for available ML backends (excluding mock for TS)."""
        # Ensure SELLA is available for transition state optimization
        if not deps.has("sella"):
            pytest.skip("SELLA not available for transition state optimization")
        return request.param

    def test_water_dissociation_ts(self, backend):
        """Test water dissociation transition state across all backends."""
        water_ts_guess = TestMoleculeFactory.get_water_dissociation_ts_guess()
        optimizer = Explorer(atoms=water_ts_guess, backend=backend)

        start_time = time.time()
        result = optimizer.run(mode="ts", fmax=0.1, steps=50)
        optimization_time = time.time() - start_time

        # Use standardized result handling
        strategy_result = TestResultHandler.normalize_result(result)
        final_atoms = TestResultHandler.extract_atoms(result)

        # Use standardized assertions
        StandardTestAssertions.assert_optimization_result(strategy_result)
        StandardTestAssertions.assert_reasonable_geometry(final_atoms, backend)

        # Check O-H distances
        oh1_dist = final_atoms.get_distance(0, 1)  # O-H (dissociating)
        oh2_dist = final_atoms.get_distance(0, 2)  # O-H (staying)

        # Dissociating H should be farther
        if strategy_result.get("converged", False) and backend != "mock":
            assert oh1_dist > oh2_dist  # Dissociating H should be farther
            assert oh1_dist > 1.3
            assert 0.8 < oh2_dist < 1.5  # Remaining OH should be reasonable

        print(
            f"Backend {backend}: H2O dissociation TS took {optimization_time:.3f}s, "
            f"{strategy_result['steps_taken']} steps, O-H distances: "
            f"{oh1_dist:.3f}, {oh2_dist:.3f} Å"
        )

    def test_sn2_like_ts(self, backend):
        """Test SN2-like transition state across all backends."""
        sn2_ts_guess = TestMoleculeFactory.get_sn2_like_ts_guess()
        optimizer = Explorer(atoms=sn2_ts_guess, backend=backend)

        start_time = time.time()
        result = optimizer.run(mode="ts", fmax=0.01, steps=50)
        optimization_time = time.time() - start_time

        # Use standardized result handling
        strategy_result = TestResultHandler.normalize_result(result)
        final_atoms = TestResultHandler.extract_atoms(result)

        # Use standardized assertions
        StandardTestAssertions.assert_optimization_result(strategy_result)
        StandardTestAssertions.assert_reasonable_geometry(final_atoms, backend)

        # Check C-F and C-Cl distances
        cf_dist = final_atoms.get_distance(0, 1)  # C-F
        ccl_dist = final_atoms.get_distance(0, 2)  # C-Cl

        if strategy_result.get("converged", False) and backend != "mock":
            # Both should be reasonable bond distances
            assert 1.1 < cf_dist < 1.7
            assert 1.8 < ccl_dist < 2.5

        print(
            f"Backend {backend}: SN2-like TS took {optimization_time:.3f}s, "
            f"{strategy_result['steps_taken']} steps, C-F: {cf_dist:.3f}, "
            f"C-Cl: {ccl_dist:.3f} Å"
        )


class TestBackendNEBOptimization:
    """Test NEB optimization across all available backends."""

    @pytest.fixture(params=get_available_backends(include_mock=False))
    def backend(self, request):
        """Parametrized fixture for available ML backends."""
        if not deps.has("sella"):
            pytest.skip("SELLA not available for NEB optimization")
        return request.param

    def test_water_dissociation_neb(self, backend):
        """Test water dissociation NEB across all backends."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()

        # Modify product to represent dissociated state
        pos = product.get_positions()
        pos[1, 0] += 1.0  # Move H away from O
        product.set_positions(pos)

        optimizer = Explorer(atoms=reactant, backend=backend)

        start_time = time.time()
        result = optimizer.run(mode="neb", product=product, npoints=5, fmax=0.1, steps=20)
        optimization_time = time.time() - start_time

        # Use standardized result handling
        strategy_result = TestResultHandler.normalize_result(result)
        final_atoms = TestResultHandler.extract_atoms(result)

        # Use standardized assertions
        StandardTestAssertions.assert_optimization_result(strategy_result)
        StandardTestAssertions.assert_reasonable_geometry(final_atoms, backend)

        print(
            f"Backend {backend}: H2O dissociation NEB took {optimization_time:.3f}s, "
            f"{strategy_result['steps_taken']} steps"
        )


class TestBackendCLI:
    """Test CLI functionality across all available backends."""

    @pytest.mark.parametrize("backend", get_available_backends())
    def test_minima_optimization_cli(self, backend: str):
        """Test minima optimization via CLI across all backends."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            # Create test molecule
            atoms = TestMoleculeFactory.get_water_distorted()
            xyz_path = os.path.join(tmp, "test.xyz")
            atoms.write(xyz_path)

            # Run optimization
            result = runner.invoke(
                main,
                [
                    "opt",
                    xyz_path,
                    "--backend",
                    backend,
                    "--optimizer",
                    "lbfgs",
                    "--steps",
                    "5",
                ],
            )

            # Verify success
            assert result.exit_code == 0, f"CLI failed: {result.output}"
            out_path = os.path.splitext(xyz_path)[0] + ".opt.xyz"
            assert os.path.exists(out_path), f"Output file not created: {out_path}"

    @pytest.mark.parametrize("backend", get_available_backends(include_mock=False))
    def test_transition_state_optimization_cli(self, backend: str):
        """Test transition state optimization via CLI across all backends."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            # Create test molecule
            atoms = TestMoleculeFactory.get_water_distorted()
            xyz_path = os.path.join(tmp, "test.xyz")
            atoms.write(xyz_path)

            # Run TS optimization
            result = runner.invoke(
                main,
                [
                    "tsopt",
                    xyz_path,
                    "--backend",
                    backend,
                    "--optimizer",
                    "sella",
                    "--steps",
                    "3",
                ],
            )

            # Verify success
            assert result.exit_code == 0, f"CLI failed: {result.output}"
            out_path = os.path.splitext(os.path.basename(xyz_path))[0] + ".ts.xyz"
            assert os.path.exists(
                os.path.join(tmp, out_path)
            ), f"Output file not created: {out_path}"

    @pytest.mark.parametrize("backend", get_available_backends())
    def test_neb_optimization_cli(self, backend: str):
        """Test NEB optimization via CLI across all backends."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            # Create reactant and product structures
            reactant_atoms = TestMoleculeFactory.get_water_distorted()
            product_atoms = TestMoleculeFactory.get_water_distorted()
            # Slightly modify product
            pos = product_atoms.get_positions()
            pos[1, 0] += 0.1  # Move H atom slightly
            product_atoms.set_positions(pos)

            reactant_path = os.path.join(tmp, "reactant.xyz")
            product_path = os.path.join(tmp, "product.xyz")
            reactant_atoms.write(reactant_path)
            product_atoms.write(product_path)

            # Run NEB optimization
            result = runner.invoke(
                main,
                [
                    "tsopt",
                    reactant_path,
                    "--product",
                    product_path,
                    "--mode",
                    "neb",
                    "--backend",
                    backend,
                    "--npoints",
                    "5",
                    "--steps",
                    "5",
                    "--fmax",
                    "0.1",
                    "--spring-constant",
                    "1.0",
                ],
            )

            # Verify success
            assert result.exit_code == 0, f"CLI failed: {result.output}"
            out_path = os.path.splitext(os.path.basename(reactant_path))[0] + ".neb.xyz"
            assert os.path.exists(
                os.path.join(tmp, out_path)
            ), f"Output file not created: {out_path}"


class TestGeometricOptimizerIntegration:
    """Test geomeTRIC optimizer integration with backends."""

    def test_geometric_availability(self):
        """Test that geomeTRIC is available."""
        if not deps.has("geometric"):
            pytest.skip("geomeTRIC not available")

    @pytest.fixture(params=get_available_backends(include_mock=False))
    def backend(self, request):
        """Parametrized fixture for available backends."""
        if not deps.has("geometric"):
            pytest.skip("geomeTRIC not available")
        return request.param

    def test_geometric_minima_optimization(self, backend):
        """Test geomeTRIC minima optimization with different backends."""
        water = TestMoleculeFactory.get_water_distorted()
        optimizer = Explorer(atoms=water, backend=backend, local_optimizer="geometric")

        result = optimizer.optimize_minima(fmax=0.1, steps=10)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        result_dict = result[0]
        assert isinstance(result_dict, dict)
        assert "optimized_atoms" in result_dict
        final_atoms = result_dict["optimized_atoms"]
        assert hasattr(final_atoms, "get_distance")
        assert len(final_atoms) == 3

    def test_geometric_ts_optimization(self, backend):
        """Test geomeTRIC transition state optimization with different backends."""
        # Skip if no ML backends available for TS optimization
        ml_backends = get_available_backends(include_mock=False)
        if not ml_backends:
            pytest.skip("No ML backends available for TS optimization")

        # Create a TS guess (stretched H2O)
        water = TestMoleculeFactory.get_water_distorted()
        ts_guess = water.copy()
        pos = ts_guess.get_positions()
        pos[1] += [0.5, 0.0, 0.0]  # Move H away from O
        ts_guess.set_positions(pos)

        optimizer = Explorer(
            atoms=ts_guess,
            backend=backend,
            local_optimizer="geometric",
        )

        result = optimizer.optimize_ts(fmax=0.1, steps=10)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        result_dict = result[0]
        assert isinstance(result_dict, dict)
        assert "optimized_atoms" in result_dict
        final_atoms = result_dict["optimized_atoms"]
        assert hasattr(final_atoms, "get_distance")
        assert len(final_atoms) == 3

    def test_geometric_vs_sella_comparison(self):
        """Compare geomeTRIC and Sella optimizers on the same system."""
        if not deps.has("geometric") or not deps.has("sella"):
            pytest.skip("Both geomeTRIC and Sella must be available")

        # Use first available ML backend
        ml_backends = get_available_backends(include_mock=False)
        if not ml_backends:
            pytest.skip("No ML backends available for comparison")

        backend = ml_backends[0]
        water = TestMoleculeFactory.get_water_distorted()

        # Test minima optimization
        geometric_opt = Explorer(atoms=water.copy(), backend=backend, local_optimizer="geometric")
        sella_opt = Explorer(atoms=water.copy(), backend=backend, local_optimizer="sella")

        # Run both optimizations
        geometric_result = geometric_opt.optimize_minima(fmax=0.1, steps=10)
        sella_result = sella_opt.optimize_minima(fmax=0.1, steps=10)

        # Both should produce valid results
        assert geometric_result is not None
        assert sella_result is not None
        assert isinstance(geometric_result, list)
        assert isinstance(sella_result, list)
        assert len(geometric_result) == 1
        assert len(sella_result) == 1

        # Extract Atoms objects from result dictionaries
        geometric_atoms = geometric_result[0]["optimized_atoms"]
        sella_atoms = sella_result[0]["optimized_atoms"]
        assert hasattr(geometric_atoms, "get_distance")
        assert hasattr(sella_atoms, "get_distance")
        assert len(geometric_atoms) == len(sella_atoms)


class TestBackendPerformanceComparison:
    """Test performance comparison between different backends."""

    def test_backend_performance_benchmark(self):
        """Benchmark different backends on the same system."""
        backends = get_available_backends(include_mock=False)
        if len(backends) < 2:
            pytest.skip("Need at least 2 backends for performance comparison")

        # Test on benzene for more realistic performance testing
        benzene = TestMoleculeFactory.get_benzene()

        results = {}
        for backend in backends[:3]:  # Test only first 3 backends to keep test time reasonable
            optimizer = Explorer(atoms=benzene.copy(), backend=backend)

            start_time = time.time()
            result = optimizer.run(mode="minima", fmax=0.05, steps=20)
            optimization_time = time.time() - start_time

            strategy_result = TestResultHandler.normalize_result(result)
            results[backend] = {
                "time": optimization_time,
                "steps": strategy_result.get("steps_taken", 0),
                "converged": strategy_result.get("converged", False),
            }

            print(f"{backend}: {optimization_time:.3f}s, {results[backend]['steps']} steps")

        # All backends should complete successfully
        for backend, result in results.items():
            assert result["time"] > 0, f"{backend} should take some time"
            assert result["steps"] > 0, f"{backend} should take some steps"
