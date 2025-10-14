"""
Standardized test utilities for QME test suite.

This module provides common utilities, fixtures, and patterns used across
all test modules to ensure consistency and reduce duplication.
"""

import os
import tempfile
import warnings

import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import write


class TestMoleculeFactory:
    """Factory for creating standardized test molecules."""

    @staticmethod
    def get_h2_stretched() -> Atoms:
        """H2 molecule with stretched bond (equilibrium ~0.74 Å)."""
        return Atoms(["H", "H"], positions=[[0, 0, 0], [2.0, 0, 0]])

    @staticmethod
    def get_water_distorted() -> Atoms:
        """Water molecule with distorted geometry."""
        return Atoms(
            ["O", "H", "H"],
            positions=[
                [0.0, 0.0, 0.0],  # O
                [1.5, 0.0, 0.0],  # H (stretched)
                [-0.3, 1.3, 0.0],  # H (stretched, bent)
            ],
        )

    @staticmethod
    def get_methane_distorted() -> Atoms:
        """Methane molecule with distorted tetrahedral geometry."""
        return Atoms(
            ["C", "H", "H", "H", "H"],
            positions=[
                [0.0, 0.0, 0.0],  # C
                [1.5, 0.0, 0.0],  # H (stretched)
                [0.0, 1.5, 0.0],  # H (stretched)
                [0.0, 0.0, 1.5],  # H (stretched)
                [-1.0, -1.0, -1.0],  # H (displaced)
            ],
        )

    @staticmethod
    def get_benzene() -> Atoms:
        """Benzene molecule for testing."""
        return molecule("C6H6")

    @staticmethod
    def get_ethylene() -> Atoms:
        """Ethylene molecule (C2H4) with planar geometry."""
        return Atoms(
            ["C", "C", "H", "H", "H", "H"],
            positions=[
                [0.0, 0.0, 0.0],  # C
                [1.34, 0.0, 0.0],  # C (C=C bond)
                [-0.7, 1.0, 0.0],  # H
                [-0.7, -1.0, 0.0],  # H
                [2.04, 1.0, 0.0],  # H
                [2.04, -1.0, 0.0],  # H
            ],
        )

    @staticmethod
    def get_ethylene_twisted_ts_guess() -> Atoms:
        """Ethylene twisted TS guess (90-degree rotation around C=C bond)."""
        return Atoms(
            ["C", "C", "H", "H", "H", "H"],
            positions=[
                [0.0, 0.0, 0.0],  # C
                [1.34, 0.0, 0.0],  # C (C=C bond)
                [-0.7, 0.0, 1.0],  # H (twisted up)
                [-0.7, 0.0, -1.0],  # H (twisted down)
                [2.04, 0.0, 1.0],  # H (twisted up)
                [2.04, 0.0, -1.0],  # H (twisted down)
            ],
        )

    @staticmethod
    def get_water_dissociation_ts_guess() -> Atoms:
        """Water dissociation TS guess (H2O -> H + OH)."""
        return Atoms(
            "H2O",
            positions=[
                [0.0, 0.0, 0.0],  # O
                [2.5, 0.0, 0.0],  # H (dissociating, far)
                [-0.5, 0.8, 0.0],  # H (staying)
            ],
        )

    @staticmethod
    def get_sn2_like_ts_guess() -> Atoms:
        """Simple SN2-like transition state guess (F- + CH3Cl -> FCH3 + Cl-)."""
        return Atoms(
            "CH3FCl",
            positions=[
                [0.0, 0.0, 0.0],  # C (center)
                [-2.0, 0.0, 0.0],  # F (approaching nucleophile)
                [2.0, 0.0, 0.0],  # Cl (leaving group)
                [0.0, 1.1, 0.0],  # H
                [0.0, -0.5, 1.0],  # H
                [0.0, -0.5, -1.0],  # H
            ],
        )

    @staticmethod
    def get_ethane_distorted() -> Atoms:
        """Ethane molecule with distorted geometry."""
        return Atoms(
            ["C", "C", "H", "H", "H", "H", "H", "H"],
            positions=[
                [0.0, 0.0, 0.0],  # C
                [2.0, 0.0, 0.0],  # C (stretched C-C)
                [-0.7, 1.2, 0.0],  # H
                [-0.7, -0.6, 1.0],  # H
                [-0.7, -0.6, -1.0],  # H
                [2.7, 1.2, 0.0],  # H
                [2.7, -0.6, 1.0],  # H
                [2.7, -0.6, -1.0],  # H
            ],
        )

    @staticmethod
    def get_methanol_distorted() -> Atoms:
        """Methanol molecule with distorted geometry."""
        return Atoms(
            ["C", "O", "H", "H", "H", "H"],
            positions=[
                [0.0, 0.0, 0.0],  # C
                [1.7, 0.0, 0.0],  # O (stretched C-O)
                [2.5, 0.0, 0.0],  # H (O-H)
                [-0.7, 1.2, 0.0],  # H
                [-0.7, -0.6, 1.0],  # H
                [-0.7, -0.6, -1.0],  # H
            ],
        )


class TestFileManager:
    """Utility for managing test files and temporary directories."""

    @staticmethod
    def create_temp_xyz(atoms: Atoms, filename: str = "test.xyz") -> tuple[str, str]:
        """Create a temporary XYZ file and return (filepath, tempdir)."""
        tempdir = tempfile.mkdtemp()
        filepath = os.path.join(tempdir, filename)
        write(filepath, atoms)
        return filepath, tempdir

    @staticmethod
    def cleanup_temp_dir(tempdir: str) -> None:
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(tempdir, ignore_errors=True)


class BackendTestMixin:
    """Mixin class providing common backend testing functionality."""

    @staticmethod
    def check_backend_availability(backend: str) -> bool:
        """Check if a backend is truly available for testing."""
        if backend == "mock":
            return True

        # Use the sophisticated backend availability checker
        # which handles dependency conflicts and version issues
        from qme.backend_availability import is_backend_available

        return is_backend_available(backend)

    @staticmethod
    def get_available_backends() -> list[str]:
        """Get list of backends that are actually available for testing."""
        all_backends = [
            "mock",
            "aimnet2",
            "mace",
            "uma",
            "so3lr",
            "torchsim_mace",
            "torchsim_uma",
        ]
        return [b for b in all_backends if BackendTestMixin.check_backend_availability(b)]

    @staticmethod
    def require_backend(backend: str) -> None:
        """Skip test if backend is not available."""
        if not BackendTestMixin.check_backend_availability(backend):
            pytest.skip(f"Backend {backend} not available")


class TestResultHandler:
    """Utility class for handling test results consistently."""

    @staticmethod
    def normalize_result(result) -> dict:
        """Normalize optimization result to standard dictionary format."""
        # Handle list return format from run() method
        if isinstance(result, list) and len(result) > 0:
            strategy_result = result[0]
        else:
            strategy_result = result
        return strategy_result

    @staticmethod
    def extract_atoms(result) -> Atoms:
        """Extract Atoms object from optimization result."""
        normalized_result = TestResultHandler.normalize_result(result)
        return normalized_result["optimized_atoms"]

    @staticmethod
    def process_result(result, backend: str) -> dict:
        """Process optimization result and return standardized dictionary with atoms and metadata.

        This method handles multiple return formats from QME optimization strategies:

        1. **Dict format**: Direct dictionary from local strategies (minima, TS)
           - Contains: optimized_atoms, steps_taken, converged, strategy

        2. **List of dicts format**: Legacy format from older implementations
           - Takes the first dictionary from the list

        3. **List of atoms format**: NEB/CI-NEB strategies return trajectory
           - Converts to dict format with optimized_atoms as the full trajectory
           - Sets strategy to "neb_or_cineb" and converged to True

        Args:
            result: Raw result from optimization strategy (dict, list of dicts, or list of atoms)
            backend: Backend name for error reporting

        Returns:
            Standardized dictionary with keys: optimized_atoms, steps_taken, converged, strategy

        Raises:
            AssertionError: If result format is unexpected or missing required keys
        """
        # Handle list return format from run() method
        if isinstance(result, list) and len(result) > 0:
            # Check if it's a list of dictionaries (old format) or list of atoms (NEB/CI-NEB)
            if isinstance(result[0], dict):
                strategy_result = result[0]
            else:
                # NEB/CI-NEB returns list of atoms - convert to expected format
                strategy_result = {
                    "optimized_atoms": result,  # The entire trajectory
                    "steps_taken": None,
                    "converged": True,
                    "strategy": "neb_or_cineb",
                }
        else:
            strategy_result = result

        # Ensure we have the expected structure
        assert isinstance(
            strategy_result, dict
        ), f"Expected dict result, got {type(strategy_result)}"
        assert "optimized_atoms" in strategy_result, "Missing 'optimized_atoms' in result"

        return strategy_result


class StandardTestAssertions:
    """Standardized assertion methods for common test patterns."""

    @staticmethod
    def assert_optimization_result(result: dict, expected_keys: list[str] | None = None) -> None:
        """Assert that optimization result has expected structure."""
        assert isinstance(result, dict), "Result should be a dict"

        if expected_keys is None:
            expected_keys = ["converged", "optimized_atoms", "strategy"]

        for key in expected_keys:
            assert key in result, f"Missing key '{key}' in optimization result"

        # Handle both Python bool and numpy bool types for converged field
        import numpy as np

        converged = result["converged"]
        if isinstance(converged, list):
            for c in converged:
                assert isinstance(c, (bool, np.bool_)), "converged should be boolean"
        else:
            assert isinstance(converged, (bool, np.bool_)), "converged should be boolean"

        # Handle optimized_atoms - can be single Atoms or list of Atoms
        optimized_atoms = result["optimized_atoms"]
        if isinstance(optimized_atoms, list):
            for atoms in optimized_atoms:
                assert isinstance(atoms, Atoms), "optimized_atoms should be Atoms object(s)"
        else:
            assert isinstance(optimized_atoms, Atoms), "optimized_atoms should be Atoms object"

        # steps_taken may be None in some implementations
        if "steps_taken" in result:
            steps_taken = result["steps_taken"]
            if steps_taken is not None:
                if isinstance(steps_taken, list):
                    for steps in steps_taken:
                        assert isinstance(steps, int), "steps_taken should be integer(s) or None"
                        assert steps >= 0, "steps_taken should be non-negative"
                else:
                    assert isinstance(steps_taken, int), "steps_taken should be integer or None"
                    assert steps_taken >= 0, "steps_taken should be non-negative"

    @staticmethod
    def assert_reasonable_geometry(atoms, backend: str = "mock") -> None:
        """Assert that molecular geometry is physically reasonable."""
        # Handle both single atoms and list of atoms (trajectory)
        if isinstance(atoms, list):
            # For trajectory, check the first and last frames
            atoms_to_check = [atoms[0], atoms[-1]]
        else:
            atoms_to_check = [atoms]

        for atoms_frame in atoms_to_check:
            # Check for overlapping atoms
            from ase.geometry import get_distances

            distances = get_distances(atoms_frame.get_positions())[1]

            # Remove diagonal (self-distances)
            n_atoms = len(atoms_frame)
            for i in range(n_atoms):
                distances[i, i] = float("inf")

            min_distance = distances.min()
            if backend == "mock":
                # Mock calculator can produce non-physical geometries
                assert min_distance > 0.1, f"Atoms too close: {min_distance:.3f} Å"
            else:
                # Real ML potentials should be more accurate
                assert min_distance > 0.5, f"Atoms too close: {min_distance:.3f} Å"

    @staticmethod
    def assert_energy_reasonable(energy: float, backend: str = "mock") -> None:
        """Assert that energy is reasonable."""
        assert energy == energy, "Energy should not be NaN"
        assert energy != float("inf"), "Energy should not be infinite"
        assert energy != float("-inf"), "Energy should not be negative infinite"

        if backend != "mock":
            # Real ML potentials should give reasonable energies
            assert -1000 < energy < 1000, f"Energy out of reasonable range: {energy:.3f} eV"

    @staticmethod
    def assert_forces_reasonable(forces, backend: str = "mock") -> None:
        """Assert that forces are reasonable."""
        assert not (forces != forces).any(), "Forces should not contain NaN"
        assert not (forces == float("inf")).any(), "Forces should not contain infinity"
        assert not (forces == float("-inf")).any(), "Forces should not contain negative infinity"

        if backend != "mock":
            # Real ML potentials should give reasonable forces
            max_force = abs(forces).max()
            assert max_force < 100, f"Maximum force too large: {max_force:.3f} eV/Å"


# Pytest fixtures for common test patterns
@pytest.fixture
def test_molecules():
    """Provide standard test molecules."""
    return {
        "h2": TestMoleculeFactory.get_h2_stretched(),
        "water": TestMoleculeFactory.get_water_distorted(),
        "methane": TestMoleculeFactory.get_methane_distorted(),
        "benzene": TestMoleculeFactory.get_benzene(),
    }


@pytest.fixture
def available_backends():
    """Provide list of available backends."""
    return BackendTestMixin.get_available_backends()


@pytest.fixture
def temp_xyz_file():
    """Provide a temporary XYZ file for testing."""
    atoms = TestMoleculeFactory.get_water_distorted()
    filepath, tempdir = TestFileManager.create_temp_xyz(atoms)
    yield filepath
    TestFileManager.cleanup_temp_dir(tempdir)


class BackendTestRunner:
    """Utility class for running tests across multiple backends with graceful failure handling."""

    @staticmethod
    def run_with_warnings(test_func, backends=None, include_mock=False, **test_kwargs):
        """
        Run a test function across multiple backends with warning-based error handling.

        Args:
            test_func: The test function to run
            backends: List of backends to test. If None, uses all available backends.
            include_mock: Whether to include the mock backend in testing.
            **test_kwargs: Additional keyword arguments to pass to the test function.

        Returns:
            Dictionary mapping backend names to their test results.
        """
        from tests.backend_test_helpers import run_backend_test_with_warnings

        return run_backend_test_with_warnings(
            test_func, backends=backends, include_mock=include_mock, **test_kwargs
        )

    @staticmethod
    def assert_backend_results(results, min_successful=1):
        """
        Assert that at least a minimum number of backends succeeded.

        Args:
            results: Dictionary from run_with_warnings
            min_successful: Minimum number of backends that must succeed
        """
        successful = [b for b, r in results.items() if r["success"]]
        failed = [b for b, r in results.items() if not r["success"]]

        if len(successful) < min_successful:
            error_summary = "\n".join(
                [f"  {b}: {r['error']}" for b, r in results.items() if not r["success"]]
            )
            pytest.fail(
                f"Not enough backends succeeded. Required: {min_successful}, "
                f"Got: {len(successful)}\nFailed backends:\n{error_summary}"
            )

        # Log warnings for failed backends
        for backend, result in results.items():
            if not result["success"]:
                warning_msg = f"Backend '{backend}' failed: {result['error']}"
                warnings.warn(warning_msg, UserWarning, stacklevel=2)

        return successful, failed
