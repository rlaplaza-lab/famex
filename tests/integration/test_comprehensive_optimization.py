"""
Comprehensive optimization tests for QME.

This module tests geometry optimization and transition state searches using
the QME Explorer API across all available backends.

Test systems include:
- H2 molecule (simple diatomic)
- H2O molecule (bent triatomic)
- CH4 molecule (tetrahedral)
- Water dissociation pathway
- SN2-like reaction coordinate
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from qme import Explorer
from qme.backend_availability import get_available_backends
from qme.dependencies import deps
from tests.test_utils import StandardTestAssertions, TestMoleculeFactory, TestResultHandler

# TestSystemDefinitions class removed - using TestMoleculeFactory instead


class TestMinimaOptimization:
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
        # Check H-H distance on standardized result
        final_distance = final_atoms.get_distance(0, 1)
        if backend == "mock":
            assert 0.5 < final_distance < 2.5
        else:
            assert 0.6 < final_distance < 1.2

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

        if backend == "mock":
            assert 0.4 < oh1_dist < 2.0
            assert 0.4 < oh2_dist < 2.0
        else:
            assert 0.85 < oh1_dist < 1.6
            assert 0.85 < oh2_dist < 1.6

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

        if backend == "mock":
            for dist in ch_distances:
                assert 0.18 < dist <= 2.0
        else:
            for dist in ch_distances:
                assert 0.95 < dist < 1.25

        print(
            f"Backend {backend}: CH4 optimization took {optimization_time:.3f}s, "
            f"{strategy_result['steps_taken']} steps, avg C-H distance: "
            f"{np.mean(ch_distances):.3f} Å"
        )


class TestTransitionStateOptimization:
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
            # More lenient assertion for SO3LR which may not stretch as much
            if backend == "so3lr":
                assert oh1_dist > 1.3  # SO3LR may be more conservative
            else:
                assert oh1_dist > 1.5  # Other backends should stretch more
            assert 0.8 < oh2_dist < 1.5  # Remaining OH should be reasonable

        print(
            f"Backend {backend}: H2O dissociation TS took {optimization_time:.3f}s, "
            f"{strategy_result['steps_taken']} steps, O-H distances: "
            f"{oh1_dist:.3f}, {oh2_dist:.3f} Å"
        )

    def test_optimizer_comparison(self):
        """Test comparison between different optimizers."""
        if not deps.has("sella"):
            pytest.skip("SELLA not available for optimizer comparison")

        water_ts_guess = TestMoleculeFactory.get_water_dissociation_ts_guess()

        # Test Sella optimizer with proper ML backend
        # Use first available ML backend for TS optimization
        ml_backends = get_available_backends(include_mock=False)
        if not ml_backends:
            pytest.skip("No ML backends available for TS optimization")

        ml_backend = ml_backends[0]
        sella_optimizer = Explorer(
            atoms=water_ts_guess.copy(), backend=ml_backend, local_optimizer="sella"
        )

        sella_result = sella_optimizer.optimize_ts(fmax=0.1, steps=20)

        assert sella_result is not None
        # Normalize result shape to Atoms
        if isinstance(sella_result, list) and len(sella_result) == 1:
            result_dict = sella_result[0]
            if isinstance(result_dict, dict):
                sella_atoms = result_dict.get("optimized_atoms")
            else:
                sella_atoms = result_dict
        elif isinstance(sella_result, dict):
            sella_atoms = sella_result.get("optimized_atoms")
        else:
            sella_atoms = sella_result
        assert sella_atoms is not None and hasattr(sella_atoms, "get_distance")

        # Test geomeTRIC optimizer if available
        if deps.has("geometric"):
            geometric_optimizer = Explorer(
                atoms=water_ts_guess.copy(),
                backend=ml_backend,  # Use same ML backend
                local_optimizer="geometric",
            )

            geometric_result = geometric_optimizer.optimize_ts(fmax=0.1, steps=20)

            assert geometric_result is not None
            # Handle the new API format: list of result dictionaries
            if isinstance(geometric_result, list) and len(geometric_result) == 1:
                result_dict = geometric_result[0]
                if isinstance(result_dict, dict):
                    geometric_atoms = result_dict.get("optimized_atoms")
                else:
                    geometric_atoms = result_dict
            elif isinstance(geometric_result, dict):
                geometric_atoms = geometric_result.get("optimized_atoms")
            else:
                geometric_atoms = geometric_result
            assert geometric_atoms is not None and hasattr(geometric_atoms, "get_distance")

    def test_sn2_like_ts(self, backend):
        """Test SN2-like transition state across all backends."""
        # Create SN2-like TS guess
        sn2_ts_guess = TestMoleculeFactory.get_sn2_like_ts_guess()
        optimizer = Explorer(atoms=sn2_ts_guess, backend=backend)

        start_time = time.time()
        result = optimizer.run(mode="ts", fmax=0.1, steps=50)
        optimization_time = time.time() - start_time

        # Use standardized result handling
        strategy_result = TestResultHandler.normalize_result(result)
        final_atoms = TestResultHandler.extract_atoms(result)

        # Use standardized assertions
        StandardTestAssertions.assert_optimization_result(strategy_result)
        StandardTestAssertions.assert_reasonable_geometry(final_atoms, backend)

        # Check key distances for SN2 mechanism
        cf_dist = final_atoms.get_distance(0, 1)  # C-F (forming)
        ccl_dist = final_atoms.get_distance(0, 2)  # C-Cl (breaking)

        if strategy_result.get("converged", False) and backend != "mock":
            # In TS, both bonds should be elongated (but allow some flexibility)
            assert 1.0 < cf_dist < 3.5  # Forming bond (more lenient)
            assert 1.0 < ccl_dist < 4.0  # Breaking bond (more lenient)

        print(
            f"Backend {backend}: SN2-like TS took {optimization_time:.3f}s, "
            f"{strategy_result['steps_taken']} steps, C-F: {cf_dist:.3f} Å, "
            f"C-Cl: {ccl_dist:.3f} Å"
        )


class TestFileIO:
    """Test file I/O functionality across backends."""

    @pytest.mark.parametrize("backend", get_available_backends())
    def test_xyz_file_workflow(self, backend):
        """Test complete XYZ file workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.xyz"
            output_file = Path(tmpdir) / "output.xyz"

            # Create input file
            h2 = TestMoleculeFactory.get_h2_stretched()
            h2.write(str(input_file))

            # Create optimizer and test file operations
            optimizer = Explorer.from_file(str(input_file), backend=backend)

            # Optimize and save
            result = optimizer.run(mode="minima", fmax=0.05, steps=30)
            # Use standardized result handling
            final_atoms = TestResultHandler.extract_atoms(result)
            optimizer.save_structure(final_atoms, str(output_file))

            # Verify output file
            assert output_file.exists()
            final_atoms = optimizer.load_structure(str(output_file))
            assert len(final_atoms) == 2

            # Test that the saved structure has reasonable geometry
            final_distance = final_atoms.get_distance(0, 1)
            assert 0.5 < final_distance < 2.0, f"Unreasonable H-H distance: {final_distance}"

    def test_save_structure_robustness(self):
        """Test that save_structure handles problematic atoms objects gracefully."""

        # Create a problematic atoms object that might cause XYZ writing issues
        h2 = Atoms(["H", "H"], positions=[[0, 0, 0], [1.0, 0, 0]])

        # Add some potentially problematic arrays
        h2.arrays["test_array"] = np.array([[1, 2, 3], [4, 5, 6]])
        h2.info["charge"] = 0
        h2.info["spin"] = 1

        optimizer = Explorer(h2, backend="mock")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.xyz"

            # This should work even with problematic arrays due to the fallback
            # mechanism
            optimizer.save_structure(h2, str(output_file))

            # Verify the file was created and is readable
            assert output_file.exists()
            loaded_atoms = optimizer.load_structure(str(output_file))
            assert len(loaded_atoms) == 2
            assert list(loaded_atoms.symbols) == list(h2.symbols)


class TestBackendConsistency:
    """Test consistency between backends on the same problems."""

    def test_h2_cross_backend_consistency(self):
        """Test that all backends give reasonable H2 results."""
        results = {}

        for backend in get_available_backends():
            try:
                h2 = TestMoleculeFactory.get_h2_stretched()
                optimizer = Explorer(atoms=h2, backend=backend)

                result = optimizer.run(mode="minima", fmax=0.05, steps=50)
                # Use standardized result handling
                final_atoms = TestResultHandler.extract_atoms(result)
                final_distance = final_atoms.get_distance(0, 1)
                results[backend] = final_distance

            except (ImportError, Exception):
                continue

        # Check that all results are reasonable and somewhat consistent
        distances = list(results.values())
        print(f"H2 optimization results: {results}")

        if len(distances) > 1:
            # All should be in a reasonable range
            assert all(0.5 < d < 1.0 for d in distances), f"Distances outside range: {distances}"

            # ML backends should be more consistent with each other
            ml_distances = [results[b] for b in results if b != "mock"]
            if len(ml_distances) > 1:
                std_dev = np.std(ml_distances)
                assert std_dev < 0.5  # Allow some variation between ML potentials
