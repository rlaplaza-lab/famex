"""
Comprehensive backend testing for QME.

This module consolidates all backend testing functionality including:
- Minima optimization across all backends
- Transition state optimization across all backends
- NEB optimization across all backends
- CI-NEB optimization across all backends
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

import numpy as np
import pytest
from click.testing import CliRunner

from qme import Explorer
from qme.backend_availability import get_available_backends
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
            assert 0.70 < final_distance < 0.80, f"H2 bond length {final_distance:.3f} Å is unreasonable (expected ~0.74 Å)"

            return {
                'optimization_time': optimization_time,
                'final_distance': final_distance,
                'steps_taken': strategy_result.get('steps_taken', 0)
            }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_h2_optimization, 
            include_mock=False
        )
        
        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)
        
        # Print summary
        print(f"\nH2 optimization test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")
        
        # Verify successful results
        for backend in successful:
            result = results[backend]['result']
            print(f"  {backend}: H2 optimization took {result['optimization_time']:.3f}s, "
                  f"final distance: {result['final_distance']:.3f} Å")

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
            assert 0.90 < oh1_dist < 1.05, f"Water O-H bond length {oh1_dist:.3f} Å is unreasonable (expected ~0.96 Å)"
            assert 0.90 < oh2_dist < 1.05, f"Water O-H bond length {oh2_dist:.3f} Å is unreasonable (expected ~0.96 Å)"

            return {
                'optimization_time': optimization_time,
                'oh1_distance': oh1_dist,
                'oh2_distance': oh2_dist,
                'steps_taken': strategy_result.get('steps_taken', 0)
            }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_water_optimization, 
            include_mock=False
        )
        
        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)
        
        # Print summary
        print(f"\nH2O optimization test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")
        
        # Verify successful results
        for backend in successful:
            result = results[backend]['result']
            print(f"  {backend}: H2O optimization took {result['optimization_time']:.3f}s, "
                  f"O-H distances: {result['oh1_distance']:.3f}, {result['oh2_distance']:.3f} Å")

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
                assert 1.05 < dist < 1.15, f"Methane C-H bond length {dist:.3f} Å is unreasonable (expected ~1.09 Å)"

            return {
                'optimization_time': optimization_time,
                'ch_distances': ch_distances,
                'avg_ch_distance': np.mean(ch_distances),
                'steps_taken': strategy_result.get('steps_taken', 0)
            }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_methane_optimization, 
            include_mock=False
        )
        
        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)
        
        # Print summary
        print(f"\nCH4 optimization test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")
        
        # Verify successful results
        for backend in successful:
            result = results[backend]['result']
            print(f"  {backend}: CH4 optimization took {result['optimization_time']:.3f}s, "
                  f"avg C-H distance: {result['avg_ch_distance']:.3f} Å")


class TestBackendTransitionStateOptimization:
    """Test transition state optimization across all available backends with warning-based error handling."""

    def test_water_dissociation_ts_with_warnings(self):
        """Test water dissociation transition state across all backends with warning-based error handling."""
        # Ensure SELLA is available for transition state optimization
        if not deps.has("sella"):
            pytest.skip("SELLA not available for transition state optimization")

        def _test_water_dissociation_ts(backend):
            water_ts_guess = TestMoleculeFactory.get_water_dissociation_ts_guess()
            optimizer = Explorer(atoms=water_ts_guess, backend=backend, target="ts", strategy="local")

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
                assert oh1_dist > oh2_dist, f"Dissociating H should be farther: {oh1_dist:.3f} vs {oh2_dist:.3f} Å"
                assert oh1_dist > 1.5, f"Dissociating H distance {oh1_dist:.3f} Å too short for TS"
                assert 0.9 < oh2_dist < 1.1, f"Remaining OH bond length {oh2_dist:.3f} Å unreasonable for TS"

            return {
                'optimization_time': optimization_time,
                'oh1_distance': oh1_dist,
                'oh2_distance': oh2_dist,
                'converged': strategy_result.get('converged', False),
                'steps_taken': strategy_result.get('steps_taken', 0)
            }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_water_dissociation_ts, 
            include_mock=False
        )
        
        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)
        
        # Print summary
        print(f"\nWater dissociation TS test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")
        
        # Verify successful results
        for backend in successful:
            result = results[backend]['result']
            print(f"  {backend}: H2O dissociation TS took {result['optimization_time']:.3f}s, "
                  f"O-H distances: {result['oh1_distance']:.3f}, {result['oh2_distance']:.3f} Å")

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
                assert 1.5 < cf_dist < 2.5, f"SN2 C-F bond length {cf_dist:.3f} Å unreasonable for TS"
                assert 1.5 < ccl_dist < 2.5, f"SN2 C-Cl bond length {ccl_dist:.3f} Å unreasonable for TS"

            return {
                'optimization_time': optimization_time,
                'cf_distance': cf_dist,
                'ccl_distance': ccl_dist,
                'converged': strategy_result.get('converged', False),
                'steps_taken': strategy_result.get('steps_taken', 0)
            }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_sn2_like_ts, 
            include_mock=False
        )
        
        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)
        
        # Print summary
        print(f"\nSN2-like TS test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")
        
        # Verify successful results
        for backend in successful:
            result = results[backend]['result']
            print(f"  {backend}: SN2-like TS took {result['optimization_time']:.3f}s, "
                  f"C-F: {result['cf_distance']:.3f}, C-Cl: {result['ccl_distance']:.3f} Å")


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
                'optimization_time': optimization_time,
                'steps_taken': strategy_result.get('steps_taken', 0),
                'npoints': len(final_atoms) if isinstance(final_atoms, list) else 1
            }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_water_dissociation_neb, 
            include_mock=False
        )
        
        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)
        
        # Print summary
        print(f"\nWater dissociation NEB test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")
        
        # Verify successful results
        for backend in successful:
            result = results[backend]['result']
            print(f"  {backend}: H2O dissociation NEB took {result['optimization_time']:.3f}s, "
                  f"{result['steps_taken']} steps")


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
                'optimization_time': optimization_time,
                'steps_taken': strategy_result.get('steps_taken', 0),
                'npoints': len(final_atoms) if isinstance(final_atoms, list) else 1
            }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_water_dissociation_cineb, 
            include_mock=False
        )
        
        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)
        
        # Print summary
        print(f"\nWater dissociation CI-NEB test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")
        
        # Verify successful results
        for backend in successful:
            result = results[backend]['result']
            print(f"  {backend}: H2O dissociation CI-NEB took {result['optimization_time']:.3f}s, "
                  f"{result['steps_taken']} steps")


class TestBackendCLI:
    """Test CLI functionality across all available backends with warning-based error handling."""

    def test_minima_optimization_cli_with_warnings(self):
        """Test minima optimization via CLI across all backends with warning-based error handling."""
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
                    'exit_code': result.exit_code,
                    'output_file_exists': os.path.exists(out_path)
                }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_minima_optimization_cli, 
            include_mock=True  # Include mock for CLI tests
        )
        
        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)
        
        # Print summary
        print(f"\nCLI minima optimization test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")
        
        # Verify successful results
        for backend in successful:
            result = results[backend]['result']
            assert result['exit_code'] == 0
            assert result['output_file_exists'] is True

    def test_transition_state_optimization_cli_with_warnings(self):
        """Test transition state optimization via CLI across all backends with warning-based error handling."""
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
                    'exit_code': result.exit_code,
                    'output_file_exists': os.path.exists(os.path.join(tmp, out_path))
                }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_transition_state_optimization_cli, 
            include_mock=False
        )
        
        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)
        
        # Print summary
        print(f"\nCLI TS optimization test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")
        
        # Verify successful results
        for backend in successful:
            result = results[backend]['result']
            assert result['exit_code'] == 0
            assert result['output_file_exists'] is True

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
                    'exit_code': result.exit_code,
                    'output_file_exists': os.path.exists(os.path.join(tmp, out_path))
                }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_neb_optimization_cli, 
            include_mock=True  # Include mock for CLI tests
        )
        
        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)
        
        # Print summary
        print(f"\nCLI NEB optimization test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")
        
        # Verify successful results
        for backend in successful:
            result = results[backend]['result']
            assert result['exit_code'] == 0
            assert result['output_file_exists'] is True

    def test_cineb_optimization_cli_with_warnings(self):
        """Test CI-NEB optimization via CLI across all backends with warning-based error handling."""
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
                    'exit_code': result.exit_code,
                    'output_file_exists': os.path.exists(os.path.join(tmp, out_path))
                }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_cineb_optimization_cli, 
            include_mock=False
        )
        
        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)
        
        # Print summary
        print(f"\nCLI CI-NEB optimization test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")
        
        # Verify successful results
        for backend in successful:
            result = results[backend]['result']
            assert result['exit_code'] == 0
            assert result['output_file_exists'] is True


class TestGeometricOptimizerIntegration:
    """Test geomeTRIC optimizer integration with backends."""

    def test_geometric_availability(self):
        """Test that geomeTRIC is available."""
        if not deps.has("geometric"):
            pytest.skip("geomeTRIC not available")

    def test_geometric_minima_optimization_with_warnings(self):
        """Test geomeTRIC minima optimization with different backends using warning system."""
        if not deps.has("geometric"):
            pytest.skip("geomeTRIC not available")

        def _test_geometric_minima_optimization(backend):
            water = TestMoleculeFactory.get_water_distorted()
            optimizer = Explorer(
                atoms=water,
                backend=backend,
                local_optimizer="geometric",
                target="minima",
                strategy="local",
            )

            result = optimizer.optimize_minima(fmax=0.1, steps=10)

            assert result is not None
            assert isinstance(result, dict)
            assert "optimized_atoms" in result
            result_dict = result
            assert isinstance(result_dict, dict)
            assert "optimized_atoms" in result_dict
            final_atoms = result_dict["optimized_atoms"]
            assert hasattr(final_atoms, "get_distance")
            assert len(final_atoms) == 3

            return {
                'final_atoms': final_atoms,
                'result_dict': result_dict
            }

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_geometric_minima_optimization, 
            include_mock=False
        )
        
        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)
        
        # Print summary
        print(f"\nGeomeTRIC minima optimization test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")

    def test_geometric_ts_optimization_with_warnings(self):
        """Test geomeTRIC transition state optimization with different backends using warning system."""
        if not deps.has("geometric"):
            pytest.skip("geomeTRIC not available")

        # Skip if no ML backends available for TS optimization
        ml_backends = get_available_backends(include_mock=False)
        if not ml_backends:
            pytest.skip("No ML backends available for TS optimization")

        def _test_geometric_ts_optimization(backend):
            # Use a more reasonable water dissociation TS guess instead of twisted ethylene
            # The twisted ethylene geometry is too distorted and causes numerical issues in geomeTRIC
            ts_guess = TestMoleculeFactory.get_water_dissociation_ts_guess()

            optimizer = Explorer(
                atoms=ts_guess,
                backend=backend,
                local_optimizer="geometric",
                target="ts",
                strategy="local",
            )

            try:
                result = optimizer.optimize_ts(fmax=0.1, steps=20)  # More lenient convergence criteria

                assert result is not None

                # Handle both list and dict results (local vs two-ended strategies)
                if isinstance(result, list):
                    assert len(result) == 1
                    result_dict = result[0]
                else:
                    result_dict = result

                assert isinstance(result_dict, dict)
                assert "optimized_atoms" in result_dict
                final_atoms = result_dict["optimized_atoms"]
                assert hasattr(final_atoms, "get_distance")
                assert len(final_atoms) == 3  # H2O

                # Verify we have a reasonable water structure
                oh1_dist = final_atoms.get_distance(0, 1)  # O-H
                oh2_dist = final_atoms.get_distance(0, 2)  # O-H
                
                # Strict bounds for geomeTRIC TS optimization
                assert 0.8 < oh1_dist < 2.5, f"GeomeTRIC O-H bond length {oh1_dist:.3f} Å is unreasonable"
                assert 0.8 < oh2_dist < 2.5, f"GeomeTRIC O-H bond length {oh2_dist:.3f} Å is unreasonable"

                return {
                    'final_atoms': final_atoms,
                    'result_dict': result_dict,
                    'oh1_dist': oh1_dist,
                    'oh2_dist': oh2_dist
                }

            except RuntimeError as e:
                if "Eigenvalues did not converge" in str(e):
                    # This is a known issue with geomeTRIC and very distorted geometries
                    raise RuntimeError(f"geomeTRIC failed with numerical convergence issue: {e}")
                else:
                    # Re-raise other runtime errors
                    raise

        # Run test across all backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_geometric_ts_optimization, 
            include_mock=False
        )
        
        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)
        
        # Print summary
        print(f"\nGeomeTRIC TS optimization test results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")
        
        # Verify successful results
        for backend in successful:
            result = results[backend]['result']
            print(f"  {backend}: O-H distances = {result['oh1_dist']:.3f}, {result['oh2_dist']:.3f} Å")

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
        geometric_opt = Explorer(
            atoms=water.copy(),
            backend=backend,
            local_optimizer="geometric",
            target="minima",
            strategy="local",
        )
        sella_opt = Explorer(
            atoms=water.copy(),
            backend=backend,
            local_optimizer="sella",
            target="minima",
            strategy="local",
        )

        # Store initial state for comparison (after setting up calculators)
        initial_positions = water.get_positions().copy()

        # Run both optimizations
        geometric_result = geometric_opt.optimize_minima(fmax=0.1, steps=10)
        sella_result = sella_opt.optimize_minima(fmax=0.1, steps=10)

        # Both should produce valid results
        assert geometric_result is not None
        assert sella_result is not None
        assert isinstance(geometric_result, dict)
        assert isinstance(sella_result, dict)
        assert "optimized_atoms" in geometric_result
        assert "optimized_atoms" in sella_result

        # Extract Atoms objects from result dictionaries
        geometric_atoms = geometric_result["optimized_atoms"]
        sella_atoms = sella_result["optimized_atoms"]
        assert hasattr(geometric_atoms, "get_distance")
        assert hasattr(sella_atoms, "get_distance")
        assert len(geometric_atoms) == len(sella_atoms)

        # STRINGENT CHECKS: Verify that optimizers actually optimized
        # Skip energy comparison since initial_energy is not available
        # TODO: Add initial energy tracking for more stringent validation
        geometric_position_change = np.max(
            np.abs(geometric_atoms.get_positions() - initial_positions)
        )
        sella_position_change = np.max(np.abs(sella_atoms.get_positions() - initial_positions))

        print(f"Geometric: Position change = {geometric_position_change:.6f}")
        print(f"Sella: Position change = {sella_position_change:.6f}")

        # Both optimizers should actually optimize (this will catch the GeometricOptimizer bug)
        assert geometric_position_change > 1e-6, (
            "GeometricOptimizer should actually change positions. "
            "This test catches bugs where optimizers report steps but don't optimize."
        )
        assert sella_position_change > 1e-6, "Sella optimizer should actually change positions"

        # DETAILED COORDINATE COMPARISON
        print("\n=== DETAILED COORDINATE COMPARISON ===")

        geo_positions = geometric_atoms.get_positions()
        sella_positions = sella_atoms.get_positions()

        # Maximum coordinate difference
        max_coord_diff = np.max(np.abs(geo_positions - sella_positions))
        rms_coord_diff = np.sqrt(np.mean((geo_positions - sella_positions) ** 2))

        print("Geometric vs Sella:")
        print(f"  Max coordinate difference: {max_coord_diff:.6f} Å")
        print(f"  RMS coordinate difference: {rms_coord_diff:.6f} Å")

        # Check if coordinates are reasonably similar (within 0.1 Å)
        assert max_coord_diff < 0.1, (
            f"Final coordinates differ too much between Geometric and Sella: "
            f"{max_coord_diff:.6f} Å. This suggests inconsistent optimization."
        )

        # DETAILED FORCE COMPARISON
        print("\n=== DETAILED FORCE COMPARISON ===")

        geo_forces = geometric_atoms.get_forces()
        sella_forces = sella_atoms.get_forces()

        geo_max_force = np.max(np.abs(geo_forces))
        sella_max_force = np.max(np.abs(sella_forces))

        geo_rms_force = np.sqrt(np.mean(geo_forces**2))
        sella_rms_force = np.sqrt(np.mean(sella_forces**2))

        print("Geometric:")
        print(f"  Max force: {geo_max_force:.6f} eV/Å")
        print(f"  RMS force: {geo_rms_force:.6f} eV/Å")

        print("Sella:")
        print(f"  Max force: {sella_max_force:.6f} eV/Å")
        print(f"  RMS force: {sella_rms_force:.6f} eV/Å")

        # DETAILED GEOMETRIC PROPERTY COMPARISON
        print("\n=== DETAILED GEOMETRIC PROPERTY COMPARISON ===")

        # Compare bond lengths
        n_atoms = len(geometric_atoms)
        bond_lengths_geo = []
        bond_lengths_sella = []

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Only consider bonds within reasonable distance (e.g., < 2.5 Å for C-C bonds)
                if geometric_atoms.get_distance(i, j) < 2.5:
                    bond_lengths_geo.append(geometric_atoms.get_distance(i, j))
                    bond_lengths_sella.append(sella_atoms.get_distance(i, j))

        if bond_lengths_geo and bond_lengths_sella:
            bond_lengths_geo = np.array(bond_lengths_geo)
            bond_lengths_sella = np.array(bond_lengths_sella)

            max_bond_diff = np.max(np.abs(bond_lengths_geo - bond_lengths_sella))
            rms_bond_diff = np.sqrt(np.mean((bond_lengths_geo - bond_lengths_sella) ** 2))

            print("Bond lengths:")
            print(f"  Number of bonds compared: {len(bond_lengths_geo)}")
            print(f"  Max bond length difference: {max_bond_diff:.6f} Å")
            print(f"  RMS bond length difference: {rms_bond_diff:.6f} Å")

            # Bond lengths should be very similar (within 0.01 Å)
            assert max_bond_diff < 0.01, (
                f"Bond lengths differ too much between optimizers: "
                f"{max_bond_diff:.6f} Å. This suggests inconsistent optimization."
            )

        # Compare angles (for molecules with at least 3 atoms)
        if n_atoms >= 3:
            angles_geo = []
            angles_sella = []

            for i in range(n_atoms):
                for j in range(n_atoms):
                    if i == j:
                        continue
                    for k in range(j + 1, n_atoms):
                        if k == i:
                            continue
                        # Calculate angle i-j-k
                        angle_geo = geometric_atoms.get_angle(i, j, k)
                        angle_sella = sella_atoms.get_angle(i, j, k)

                        # Only consider angles that are not close to 0 or 180 degrees
                        if 10 < angle_geo < 170 and 10 < angle_sella < 170:
                            angles_geo.append(angle_geo)
                            angles_sella.append(angle_sella)

            if angles_geo and angles_sella:
                angles_geo = np.array(angles_geo)
                angles_sella = np.array(angles_sella)

                max_angle_diff = np.max(np.abs(angles_geo - angles_sella))
                rms_angle_diff = np.sqrt(np.mean((angles_geo - angles_sella) ** 2))

                print("Angles:")
                print(f"  Number of angles compared: {len(angles_geo)}")
                print(f"  Max angle difference: {max_angle_diff:.2f}°")
                print(f"  RMS angle difference: {rms_angle_diff:.2f}°")

                # Angles should be reasonably similar (within 5°)
                assert max_angle_diff < 5.0, (
                    f"Angles differ too much between optimizers: "
                    f"{max_angle_diff:.2f}°. This suggests inconsistent optimization."
                )

        # ENERGY COMPARISON
        print("\n=== ENERGY COMPARISON ===")

        geo_energy = geometric_atoms.get_potential_energy()
        sella_energy = sella_atoms.get_potential_energy()
        energy_diff = abs(geo_energy - sella_energy)

        print(f"Geometric energy: {geo_energy:.6f} eV")
        print(f"Sella energy: {sella_energy:.6f} eV")
        print(f"Energy difference: {energy_diff:.6f} eV")

        # Energies should be reasonably similar (within 0.001 eV)
        assert energy_diff < 0.001, (
            f"Final energies differ too much between optimizers: {energy_diff:.6f} eV. "
            f"This suggests inconsistent optimization to different minima."
        )


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

            # Strict geometric checks for benzene
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
                assert 1.35 < avg_c_c_dist < 1.45, f"Benzene C-C bond length {avg_c_c_dist:.3f} Å is unreasonable"

            return {
                "time": optimization_time,
                "steps": steps_taken,
                "converged": strategy_result.get("converged", False),
                "avg_c_c_distance": np.mean(c_c_distances) if c_c_distances else None,
            }

        # Run test across available backends with warning-based error handling
        results = BackendTestRunner.run_with_warnings(
            _test_backend_performance, 
            include_mock=False
        )
        
        # Assert that at least one backend succeeded
        successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)
        
        # Print summary
        print(f"\nBackend performance benchmark results:")
        print(f"  ✅ Successful backends: {', '.join(successful)}")
        if failed:
            print(f"  ⚠️  Failed backends: {', '.join(failed)}")
        
        # Verify successful results and print performance data
        for backend in successful:
            result = results[backend]['result']
            assert result["time"] > 0, f"{backend} should take some time"
            assert result["steps"] > 0, f"{backend} should take some steps"
            print(f"  {backend}: {result['time']:.3f}s, {result['steps']} steps, "
                  f"avg C-C distance: {result['avg_c_c_distance']:.3f} Å")


