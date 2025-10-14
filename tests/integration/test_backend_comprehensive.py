"""
Comprehensive backend testing for QME.

This module consolidates all backend testing functionality including:
- Minima optimization across all backends
- Transition state optimization across all backends
- NEB optimization across all backends
- CI-NEB optimization across all backends
- CLI functionality testing
- Optimizer integration
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

import numpy as np
import pytest
from click.testing import CliRunner

from qme import Explorer
from qme.cli import main
from qme.dependencies import deps
from tests.test_utils import (
    BackendTestRunner,
    StandardTestAssertions,
    TestMoleculeFactory,
    TestResultHandler,
)


class TestBackendMinimaOptimization:
    """Test minima optimization across all available backends with warning-based error handling."""

    def test_h2_optimization_with_warnings(self):
        """Test H2 optimization across all backends with warning-based error handling."""

        def _test_h2_optimization(backend):
            h2 = TestMoleculeFactory.get_h2_stretched()
            optimizer = Explorer(atoms=h2, backend=backend, target="minima", strategy="local")

            start_time = time.time()
            result = optimizer.run(mode="minima", local_optimizer_name="BFGS", fmax=0.05, steps=20)
            optimization_time = time.time() - start_time

            # Use standardized result handling
            strategy_result = TestResultHandler.process_result(result, backend)
            final_atoms = strategy_result["optimized_atoms"]

            # Use standardized assertions
            StandardTestAssertions.assert_optimization_result(strategy_result)
            StandardTestAssertions.assert_reasonable_geometry(final_atoms, backend)

            # Check H-H distance - strict bounds for H2 equilibrium
            final_distance = final_atoms.get_distance(0, 1)
            assert (
                0.70 < final_distance < 0.80
            ), f"H2 bond length {final_distance:.3f} Å is unreasonable (expected ~0.74 Å)"

            return {
                "optimization_time": optimization_time,
                "final_distance": final_distance,
                "steps_taken": strategy_result.get("steps_taken", 0),
            }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(_test_h2_optimization, include_mock=False)

        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)

        # Print summary
        print("\nH2 optimization test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")

        # Verify successful results
        for backend in successful:
            result = results[backend]["result"]
            print(
                f"  {backend}: H2 optimization took {result['optimization_time']:.3f}s, "
                f"final distance: {result['final_distance']:.3f} Å"
            )

    def test_water_optimization_with_warnings(self):
        """Test H2O optimization across all backends with warning-based error handling."""

        def _test_water_optimization(backend):
            water = TestMoleculeFactory.get_water_distorted()
            optimizer = Explorer(atoms=water, backend=backend, target="minima", strategy="local")

            start_time = time.time()
            result = optimizer.run(mode="minima", local_optimizer_name="BFGS", fmax=0.05, steps=50)
            optimization_time = time.time() - start_time

            # Use standardized result handling
            strategy_result = TestResultHandler.process_result(result, backend)
            final_atoms = strategy_result["optimized_atoms"]

            # Use standardized assertions
            StandardTestAssertions.assert_optimization_result(strategy_result)
            StandardTestAssertions.assert_reasonable_geometry(final_atoms, backend)

            # Check O-H distances
            oh1_dist = final_atoms.get_distance(0, 1)
            oh2_dist = final_atoms.get_distance(0, 2)

            # Strict bounds for water O-H bonds (should be ~0.96 Å)
            assert (
                0.90 < oh1_dist < 1.05
            ), f"Water O-H bond length {oh1_dist:.3f} Å is unreasonable (expected ~0.96 Å)"
            assert (
                0.90 < oh2_dist < 1.05
            ), f"Water O-H bond length {oh2_dist:.3f} Å is unreasonable (expected ~0.96 Å)"

            return {
                "optimization_time": optimization_time,
                "oh1_distance": oh1_dist,
                "oh2_distance": oh2_dist,
                "steps_taken": strategy_result.get("steps_taken", 0),
            }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(_test_water_optimization, include_mock=False)

        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)

        # Print summary
        print("\nH2O optimization test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")

        # Verify successful results
        for backend in successful:
            result = results[backend]["result"]
            print(
                f"  {backend}: H2O optimization took {result['optimization_time']:.3f}s, "
                f"O-H distances: {result['oh1_distance']:.3f}, {result['oh2_distance']:.3f} Å"
            )

    def test_methane_optimization_with_warnings(self):
        """Test CH4 optimization across all backends with warning-based error handling."""

        def _test_methane_optimization(backend):
            methane = TestMoleculeFactory.get_methane_distorted()
            optimizer = Explorer(atoms=methane, backend=backend, target="minima", strategy="local")

            start_time = time.time()
            result = optimizer.run(mode="minima", local_optimizer_name="BFGS", fmax=0.05, steps=50)
            optimization_time = time.time() - start_time

            # Use standardized result handling
            strategy_result = TestResultHandler.process_result(result, backend)
            final_atoms = strategy_result["optimized_atoms"]

            # Use standardized assertions
            StandardTestAssertions.assert_optimization_result(strategy_result)
            StandardTestAssertions.assert_reasonable_geometry(final_atoms, backend)

            # Check C-H distances
            ch_distances = [final_atoms.get_distance(0, i) for i in range(1, 5)]

            # Strict bounds for methane C-H bonds (should be ~1.09 Å)
            for dist in ch_distances:
                assert (
                    1.05 < dist < 1.15
                ), f"Methane C-H bond length {dist:.3f} Å is unreasonable (expected ~1.09 Å)"

            return {
                "optimization_time": optimization_time,
                "ch_distances": ch_distances,
                "avg_ch_distance": np.mean(ch_distances),
                "steps_taken": strategy_result.get("steps_taken", 0),
            }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_methane_optimization, include_mock=False
        )

        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)

        # Print summary
        print("\nCH4 optimization test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")

        # Verify successful results
        for backend in successful:
            result = results[backend]["result"]
            print(
                f"  {backend}: CH4 optimization took {result['optimization_time']:.3f}s, "
                f"avg C-H distance: {result['avg_ch_distance']:.3f} Å"
            )


class TestBackendTransitionStateOptimization:
    """Test transition state optimization across all available backends
    with warning-based error handling."""

    def test_water_dissociation_ts_with_warnings(self):
        """Test water dissociation transition state across all backends
        with warning-based error handling."""
        # Ensure SELLA is available for transition state optimization
        if not deps.has("sella"):
            pytest.skip("SELLA not available for transition state optimization")

        def _test_water_dissociation_ts(backend):
            water_ts_guess = TestMoleculeFactory.get_water_dissociation_ts_guess()
            optimizer = Explorer(
                atoms=water_ts_guess, backend=backend, target="ts", strategy="local"
            )

            start_time = time.time()
            result = optimizer.run(mode="ts", fmax=0.1, steps=50)
            optimization_time = time.time() - start_time

            # Use standardized result handling
            strategy_result = TestResultHandler.process_result(result, backend)
            final_atoms = strategy_result["optimized_atoms"]

            # Use standardized assertions
            StandardTestAssertions.assert_optimization_result(strategy_result)
            StandardTestAssertions.assert_reasonable_geometry(final_atoms, backend)

            # Check O-H distances
            oh1_dist = final_atoms.get_distance(0, 1)  # O-H (dissociating)
            oh2_dist = final_atoms.get_distance(0, 2)  # O-H (staying)

            # Dissociating H should be farther - strict bounds for water dissociation TS
            if strategy_result.get("converged", False) and backend != "mock":
                assert (
                    oh1_dist > oh2_dist
                ), f"Dissociating H should be farther: {oh1_dist:.3f} vs {oh2_dist:.3f} Å"
                assert oh1_dist > 1.5, f"Dissociating H distance {oh1_dist:.3f} Å too short for TS"
                assert (
                    0.9 < oh2_dist < 1.1
                ), f"Remaining OH bond length {oh2_dist:.3f} Å unreasonable for TS"

            return {
                "optimization_time": optimization_time,
                "oh1_distance": oh1_dist,
                "oh2_distance": oh2_dist,
                "converged": strategy_result.get("converged", False),
                "steps_taken": strategy_result.get("steps_taken", 0),
            }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_water_dissociation_ts, include_mock=False
        )

        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)

        # Print summary
        print("\nWater dissociation TS test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")

        # Verify successful results
        for backend in successful:
            result = results[backend]["result"]
            print(
                f"  {backend}: H2O dissociation TS took {result['optimization_time']:.3f}s, "
                f"O-H distances: {result['oh1_distance']:.3f}, {result['oh2_distance']:.3f} Å"
            )

    def test_sn2_like_ts_with_warnings(self):
        """Test SN2-like transition state across all backends with warning-based error handling."""
        # Ensure SELLA is available for transition state optimization
        if not deps.has("sella"):
            pytest.skip("SELLA not available for transition state optimization")

        def _test_sn2_like_ts(backend):
            sn2_ts_guess = TestMoleculeFactory.get_sn2_like_ts_guess()
            optimizer = Explorer(atoms=sn2_ts_guess, backend=backend, target="ts", strategy="local")

            start_time = time.time()
            result = optimizer.run(mode="ts", fmax=0.01, steps=50)
            optimization_time = time.time() - start_time

            # Use standardized result handling
            strategy_result = TestResultHandler.process_result(result, backend)
            final_atoms = strategy_result["optimized_atoms"]

            # Use standardized assertions
            StandardTestAssertions.assert_optimization_result(strategy_result)
            StandardTestAssertions.assert_reasonable_geometry(final_atoms, backend)

            # Check C-F and C-Cl distances
            cf_dist = final_atoms.get_distance(0, 1)  # C-F
            ccl_dist = final_atoms.get_distance(0, 2)  # C-Cl

            if strategy_result.get("converged", False) and backend != "mock":
                # Strict bounds for SN2 transition state - both bonds should be partially formed
                assert (
                    1.5 < cf_dist < 2.5
                ), f"SN2 C-F bond length {cf_dist:.3f} Å unreasonable for TS"
                assert (
                    1.5 < ccl_dist < 2.5
                ), f"SN2 C-Cl bond length {ccl_dist:.3f} Å unreasonable for TS"

            return {
                "optimization_time": optimization_time,
                "cf_distance": cf_dist,
                "ccl_distance": ccl_dist,
                "converged": strategy_result.get("converged", False),
                "steps_taken": strategy_result.get("steps_taken", 0),
            }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(_test_sn2_like_ts, include_mock=False)

        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)

        # Print summary
        print("\nSN2-like TS test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")

        # Verify successful results
        for backend in successful:
            result = results[backend]["result"]
            print(
                f"  {backend}: SN2-like TS took {result['optimization_time']:.3f}s, "
                f"C-F: {result['cf_distance']:.3f}, C-Cl: {result['ccl_distance']:.3f} Å"
            )


class TestBackendNEBOptimization:
    """Test NEB optimization across all available backends with warning-based error handling."""

    def test_water_dissociation_neb_with_warnings(self):
        """Test water dissociation NEB across all backends with warning-based error handling."""
        if not deps.has("sella"):
            pytest.skip("SELLA not available for NEB optimization")

        def _test_water_dissociation_neb(backend):
            reactant = TestMoleculeFactory.get_water_distorted()
            product = TestMoleculeFactory.get_water_distorted()

            # Create a simple dissociation by moving H away
            product.positions[2] = product.positions[2] + np.array([2.0, 0.0, 0.0])

            optimizer = Explorer(atoms=reactant, backend=backend, target="path", strategy="neb")

            start_time = time.time()
            result = optimizer.run(mode="neb", product=product, npoints=5, fmax=0.1, steps=20)
            optimization_time = time.time() - start_time

            # Use standardized result handling
            strategy_result = TestResultHandler.process_result(result, backend)
            final_atoms = strategy_result["optimized_atoms"]

            # Standardized assertions
            StandardTestAssertions.assert_reasonable_geometry(final_atoms, backend)

            return {
                "optimization_time": optimization_time,
                "steps_taken": strategy_result.get("steps_taken", 0),
                "npoints": len(final_atoms) if isinstance(final_atoms, list) else 1,
            }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_water_dissociation_neb, include_mock=False
        )

        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)

        # Print summary
        print("\nWater dissociation NEB test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")

        # Verify successful results
        for backend in successful:
            result = results[backend]["result"]
            print(
                f"  {backend}: H2O dissociation NEB took {result['optimization_time']:.3f}s, "
                f"{result['steps_taken']} steps"
            )


class TestBackendCINEBOptimization:
    """Test CI-NEB optimization across all available backends with warning-based error handling."""

    def test_water_dissociation_cineb_with_warnings(self):
        """Test water dissociation CI-NEB across all backends with warning-based error handling."""
        if not deps.has("sella"):
            pytest.skip("SELLA not available for CI-NEB optimization")

        def _test_water_dissociation_cineb(backend):
            reactant = TestMoleculeFactory.get_water_distorted()
            product = TestMoleculeFactory.get_water_distorted()

            # Create a simple dissociation by moving H away
            product.positions[2] = product.positions[2] + np.array([2.0, 0.0, 0.0])

            optimizer = Explorer(atoms=reactant, backend=backend, target="path", strategy="cineb")

            start_time = time.time()
            result = optimizer.run(
                mode="cineb", product=product, npoints=5, fmax=0.1, steps=20, climb=True
            )
            optimization_time = time.time() - start_time

            # Use standardized result handling
            strategy_result = TestResultHandler.process_result(result, backend)
            final_atoms = strategy_result["optimized_atoms"]

            # Standardized assertions
            StandardTestAssertions.assert_reasonable_geometry(final_atoms, backend)

            return {
                "optimization_time": optimization_time,
                "steps_taken": strategy_result.get("steps_taken", 0),
                "npoints": len(final_atoms) if isinstance(final_atoms, list) else 1,
            }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_water_dissociation_cineb, include_mock=False
        )

        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)

        # Print summary
        print("\nWater dissociation CI-NEB test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")

        # Verify successful results
        for backend in successful:
            result = results[backend]["result"]
            print(
                f"  {backend}: H2O dissociation CI-NEB took {result['optimization_time']:.3f}s, "
                f"{result['steps_taken']} steps"
            )


class TestBackendCLI:
    """Test CLI functionality across all available backends with warning-based error handling."""

    def test_minima_optimization_cli_with_warnings(self):
        """Test minima optimization via CLI across all backends
        with warning-based error handling."""

        def _test_minima_optimization_cli(backend):
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

                return {
                    "exit_code": result.exit_code,
                    "output_file_exists": os.path.exists(out_path),
                }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_minima_optimization_cli, include_mock=True  # Include mock for CLI tests
        )

        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)

        # Print summary
        print("\nCLI minima optimization test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")

        # Verify successful results
        for backend in successful:
            result = results[backend]["result"]
            assert result["exit_code"] == 0
            assert result["output_file_exists"] is True

    def test_transition_state_optimization_cli_with_warnings(self):
        """Test transition state optimization via CLI across all backends
        with warning-based error handling."""
        if not deps.has("sella"):
            pytest.skip("SELLA not available for transition state optimization")

        def _test_transition_state_optimization_cli(backend):
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

                return {
                    "exit_code": result.exit_code,
                    "output_file_exists": os.path.exists(os.path.join(tmp, out_path)),
                }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_transition_state_optimization_cli, include_mock=False
        )

        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)

        # Print summary
        print("\nCLI TS optimization test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")

        # Verify successful results
        for backend in successful:
            result = results[backend]["result"]
            assert result["exit_code"] == 0
            assert result["output_file_exists"] is True

    def test_neb_optimization_cli_with_warnings(self):
        """Test NEB optimization via CLI across all backends with warning-based error handling."""
        if not deps.has("sella"):
            pytest.skip("SELLA not available for NEB optimization")

        def _test_neb_optimization_cli(backend):
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
                        "path",
                        "neb",
                        reactant_path,
                        product_path,
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

                return {
                    "exit_code": result.exit_code,
                    "output_file_exists": os.path.exists(os.path.join(tmp, out_path)),
                }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_neb_optimization_cli, include_mock=True  # Include mock for CLI tests
        )

        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)

        # Print summary
        print("\nCLI NEB optimization test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")

        # Verify successful results
        for backend in successful:
            result = results[backend]["result"]
            assert result["exit_code"] == 0
            assert result["output_file_exists"] is True

    def test_cineb_optimization_cli_with_warnings(self):
        """Test CI-NEB optimization via CLI across all backends
        with warning-based error handling."""
        if not deps.has("sella"):
            pytest.skip("SELLA not available for CI-NEB optimization")

        def _test_cineb_optimization_cli(backend):
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

                # Run CI-NEB optimization
                result = runner.invoke(
                    main,
                    [
                        "path",
                        "cineb",
                        reactant_path,
                        product_path,
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
                out_path = os.path.splitext(os.path.basename(reactant_path))[0] + ".cineb.xyz"
                assert os.path.exists(
                    os.path.join(tmp, out_path)
                ), f"Output file not created: {out_path}"

                return {
                    "exit_code": result.exit_code,
                    "output_file_exists": os.path.exists(os.path.join(tmp, out_path)),
                }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_cineb_optimization_cli, include_mock=False
        )

        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)

        # Print summary
        print("\nCLI CI-NEB optimization test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")

        # Verify successful results
        for backend in successful:
            result = results[backend]["result"]
            assert result["exit_code"] == 0
            assert result["output_file_exists"] is True


class TestBackendPerformanceComparison:
    """Test performance comparison between different backends with warning-based error handling."""

    def test_backend_performance_benchmark_with_warnings(self):
        """Benchmark different backends on the same system with warning-based error handling."""

        def _test_backend_performance(backend):
            # Test on benzene for more realistic performance testing
            benzene = TestMoleculeFactory.get_benzene()

            optimizer = Explorer(
                atoms=benzene.copy(), backend=backend, target="minima", strategy="local"
            )

            start_time = time.time()
            result = optimizer.run(mode="minima", fmax=0.05, steps=20)
            optimization_time = time.time() - start_time

            # Handle both dict and list results
            if isinstance(result, dict):
                strategy_result = result
            else:
                strategy_result = TestResultHandler.process_result(result, backend)

            # Handle steps_taken which can be int or list
            steps_taken = strategy_result.get("steps_taken", 0)
            if isinstance(steps_taken, list):
                steps_taken = steps_taken[0] if steps_taken else 0

            # Strict tric checks for benzene
            final_atoms = strategy_result["optimized_atoms"]
            StandardTestAssertions.assert_optimization_result(strategy_result)
            StandardTestAssertions.assert_reasonable_geometry(final_atoms, backend)

            # Check benzene C-C bond lengths (should be ~1.4 Å for aromatic bonds)
            c_c_distances = []
            for i in range(6):  # 6 carbon atoms
                for j in range(i + 1, 6):
                    dist = final_atoms.get_distance(i, j)
                    if dist < 2.0:  # Only consider nearby carbons
                        c_c_distances.append(dist)

            # Strict checks for benzene geometry
            if c_c_distances:
                avg_c_c_dist = np.mean(c_c_distances)
                # Benzene C-C bonds should be around 1.4 Å
                assert (
                    1.35 < avg_c_c_dist < 1.45
                ), f"Benzene C-C bond length {avg_c_c_dist:.3f} Å is unreasonable"

            return {
                "time": optimization_time,
                "steps": steps_taken,
                "converged": strategy_result.get("converged", False),
                "avg_c_c_distance": np.mean(c_c_distances) if c_c_distances else None,
            }

        # Run test across available backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(_test_backend_performance, include_mock=False)

        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)

        # Print summary
        print("\nBackend performance benchmark results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")

        # Verify successful results and print performance data
        for backend in successful:
            result = results[backend]["result"]
            assert result["time"] > 0, f"{backend} should take some time"
            assert result["steps"] > 0, f"{backend} should take some steps"
            print(
                f"  {backend}: {result['time']:.3f}s, {result['steps']} steps, "
                f"avg C-C distance: {result['avg_c_c_distance']:.3f} Å"
            )
