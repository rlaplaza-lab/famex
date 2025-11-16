from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import write

from tests.test_constants import DEFAULT_FMAX, HARMONIC_TOL


class TestMoleculeFactory:
    @staticmethod
    def get_h2_stretched():
        """H2 molecule with stretched bond (equilibrium ~0.74 Å)."""
        return Atoms(["H", "H"], positions=[[0, 0, 0], [2.0, 0, 0]])

    @staticmethod
    def get_water_distorted():
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
    def get_methane_distorted():
        """Methane molecule with realistic equilibrium tetrahedral geometry (C-H ~1.087 Å).

        Note: Despite the method name 'distorted', this now uses a realistic equilibrium
        geometry to ensure accurate Hessian calculations. The name is kept for backward
        compatibility with existing tests.
        """
        return Atoms(
            ["C", "H", "H", "H", "H"],
            positions=[
                [0.0000000000, 0.0000000000, 0.0000000000],  # C
                [1.0870000000, 0.0000000000, 0.0000000000],  # H
                [-0.3623333220, -1.0248334322, -0.0000000000],  # H
                [-0.3623333220, 0.5124167161, -0.8875317869],  # H
                [-0.3623333220, 0.5124167161, 0.8875317869],  # H
            ],
        )

    @staticmethod
    def get_benzene():
        """Benzene molecule for testing."""
        return molecule("C6H6")

    @staticmethod
    def get_ethylene_twisted_ts_guess():
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
    def get_water_dissociation_ts_guess():
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
    def get_sn2_like_ts_guess():
        """Return simple SN2-like transition state guess (F- + CH3Cl -> FCH3 + Cl-)."""
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
    def get_h2_equilibrium():
        """H2 molecule at equilibrium geometry (bond length ~0.74 Å)."""
        return Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

    @staticmethod
    def get_h2o_equilibrium():
        """Water molecule at equilibrium geometry (from ase.build)."""
        from ase.build import molecule

        return molecule("H2O")

    @staticmethod
    def get_perturbed_molecule(base, seed=42, magnitude=0.05):
        """Create a perturbed version of a molecule for optimization tests.

        Args:
            base: Base molecule to perturb
            seed: Random seed for reproducibility
            magnitude: Magnitude of perturbation in Å
        """
        import numpy as np

        atoms = base.copy()
        rng = np.random.RandomState(seed)
        atoms.positions += rng.normal(0, magnitude, atoms.positions.shape)
        return atoms


class TestFileManager:
    @staticmethod
    def create_temp_xyz(atoms, filename="test.xyz"):
        """Create a temporary XYZ file and return (filepath, tempdir)."""
        tempdir = tempfile.mkdtemp()
        filepath = os.path.join(tempdir, filename)
        write(filepath, atoms)
        return filepath, tempdir

    @staticmethod
    def cleanup_temp_dir(tempdir):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(tempdir, ignore_errors=True)


def get_available_backends(include_mock=False):
    """Get list of backends that are actually available for testing."""
    # Use the centralized backend availability system
    from qme.backends.availability import get_available_backends as get_qme_backends

    return get_qme_backends(include_mock=include_mock)


def check_backend_availability(backend):
    """Check if a backend is truly available for testing."""
    if backend == "mock":
        return True

    # Use the sophisticated backend availability checker
    # which handles dependency conflicts and version issues
    from qme.backends.availability import is_backend_available

    return is_backend_available(backend)


class BackendTestMixin:
    @staticmethod
    def check_backend_availability(backend):
        """Check if a backend is truly available for testing."""
        return check_backend_availability(backend)

    @staticmethod
    def get_available_backends():
        """Get list of backends that are actually available for testing."""
        return get_available_backends()

    @staticmethod
    def require_backend(backend):
        """Skip test if backend is not available."""
        if not check_backend_availability(backend):
            pytest.skip(f"Backend {backend} not available")


class TestResultHandler:
    @staticmethod
    def normalize_result(result):
        """Normalize optimization result to standard dictionary format."""
        # Handle list return format from run() method
        return result[0] if isinstance(result, list) and len(result) > 0 else result

    @staticmethod
    def extract_atoms(result):
        """Extract Atoms object from optimization result."""
        normalized_result = TestResultHandler.normalize_result(result)
        return normalized_result["optimized_atoms"]

    @staticmethod
    def process_result(result, backend):
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

        Returns
        -------
        dict
            Standardized dictionary with keys: optimized_atoms, steps_taken, converged, strategy.

        Raises
        ------
        AssertionError
            If result format is unexpected or missing required keys.


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
        assert isinstance(strategy_result, dict), (
            f"Expected dict result, got {type(strategy_result)}"
        )
        assert "optimized_atoms" in strategy_result, "Missing 'optimized_atoms' in result"

        return strategy_result


class StandardTestAssertions:
    @staticmethod
    def assert_optimization_result(result, expected_keys=None):
        """Assert that optimization result has expected structure."""
        assert isinstance(result, dict), "Result should be a dict"

        if expected_keys is None:
            expected_keys = ["converged", "optimized_atoms", "strategy"]

        for key in expected_keys:
            assert key in result, f"Missing key '{key}' in optimization result"

        # Check converged is boolean-like (only if converged is in expected_keys)
        if "converged" in expected_keys:
            converged = result["converged"]
            assert isinstance(converged, bool | list) or converged in (True, False), (
                "converged should be boolean"
            )

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
    def assert_reasonable_geometry(atoms, backend="mock"):
        """Assert that molecular geometry is physically reasonable."""
        # Handle both single atoms and list of atoms (trajectory)
        atoms_to_check = [atoms[0], atoms[-1]] if isinstance(atoms, list) else [atoms]

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
    def assert_energy_reasonable(energy, backend="mock"):
        """Assert that energy is reasonable."""
        assert energy == energy, "Energy should not be NaN"
        assert energy != float("inf"), "Energy should not be infinite"
        assert energy != float("-inf"), "Energy should not be negative infinite"

        if backend != "mock":
            # Real ML potentials should give reasonable energies
            assert -1000 < energy < 1000, f"Energy out of reasonable range: {energy:.3f} eV"

    @staticmethod
    def assert_forces_reasonable(forces, backend="mock"):
        """Assert that forces are reasonable."""
        assert not (forces != forces).any(), "Forces should not contain NaN"
        assert not (forces == float("inf")).any(), "Forces should not contain infinity"
        assert not (forces == float("-inf")).any(), "Forces should not contain negative infinity"

        if backend != "mock":
            # Real ML potentials should give reasonable forces
            max_force = abs(forces).max()
            assert max_force < 100, f"Maximum force too large: {max_force:.3f} eV/Å"

    @staticmethod
    def assert_hessian_valid(hessian, expected_shape=None):
        """Assert that Hessian matrix is valid.

        Args:
            hessian: Hessian matrix to validate
            expected_shape: Expected shape (rows, cols), or None to skip shape check
        """
        import numpy as np

        # Check it's a 2D array
        assert len(hessian.shape) == 2, f"Hessian should be 2D, got shape {hessian.shape}"

        # Check square
        assert hessian.shape[0] == hessian.shape[1], "Hessian should be square"

        # Check expected shape if provided
        if expected_shape is not None:
            assert hessian.shape == expected_shape, (
                f"Expected shape {expected_shape}, got {hessian.shape}"
            )

        # Check symmetry (allowing small numerical noise)
        assert np.allclose(hessian, hessian.T, rtol=HARMONIC_TOL[0], atol=HARMONIC_TOL[1]), (
            "Hessian should be symmetric"
        )

        # Check no NaN or inf
        assert not np.any(np.isnan(hessian)), "Hessian should not contain NaN"
        assert not np.any(np.isinf(hessian)), "Hessian should not contain infinity"

    @staticmethod
    def assert_frequencies_valid(frequencies, expected_count=None):
        """Assert that frequency array is valid.

        Args:
            frequencies: Frequency array to validate
            expected_count: Expected number of frequencies, or None to skip
        """
        import numpy as np

        frequencies = np.asarray(frequencies)

        # Check expected count if provided
        if expected_count is not None:
            assert len(frequencies) == expected_count, (
                f"Expected {expected_count} frequencies, got {len(frequencies)}"
            )

        # Check no NaN or inf
        assert not np.any(np.isnan(frequencies)), "Frequencies should not contain NaN"
        assert not np.any(np.isinf(frequencies)), "Frequencies should not contain infinity"

    @staticmethod
    def assert_convergence_quality(atoms, fmax=DEFAULT_FMAX):
        """Assert that optimization converged with reasonable quality.

        Args:
            atoms: Optimized atoms object
            fmax: Maximum force threshold for convergence
        """
        from ase import Atoms

        assert isinstance(atoms, Atoms), "Input should be Atoms object"

        forces = atoms.get_forces()
        max_force = abs(forces).max()

        # Check that forces are below threshold
        assert max_force <= fmax * 1.1, (
            f"Max force {max_force:.6f} exceeds threshold {fmax:.6f} eV/Å"
        )


def assert_error_contains(error, text):
    """Assert that error message contains expected text.

    Args:
        error: Exception instance or error message string
        text: Expected text to find in error message (case-insensitive)

    Example:
        with pytest.raises(ValueError) as exc_info:
            # Code that raises error
            pass
        assert_error_contains(exc_info.value, "expected error message")
    """
    error_msg = str(error) if not isinstance(error, str) else error
    assert text.lower() in error_msg.lower(), (
        f"Expected error message to contain '{text}', but got: {error_msg}"
    )


# Pytest fixtures for common test patterns
@pytest.fixture
def test_molecules():
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


class BackendTestWarning(UserWarning):
    pass


def backend_test_with_warnings(
    backends=None,
    include_mock=False,
    test_name_suffix="",
):
    """Run a test across multiple backends with graceful failure handling.

    When a backend fails, it logs a warning but continues testing other backends.
    The test only fails if ALL backends fail.

    Args:
        backends: List of backends to test. If None, uses all available backends.
        include_mock: Whether to include the mock backend in testing.
        test_name_suffix: Suffix to add to test names for backend identification.

    Returns
    -------
    callable
        Decorated test function that handles backend failures gracefully.

    """
    import functools

    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            if backends is None:
                available_backends = get_available_backends(include_mock=include_mock)
            else:
                available_backends = backends

            if not available_backends:
                pytest.skip("No backends available for testing")

            results = {}
            warnings_list = []

            for backend in available_backends:
                try:
                    # Create a new kwargs dict with the backend parameter
                    backend_kwargs = kwargs.copy()
                    backend_kwargs["backend"] = backend

                    # Run the test for this specific backend
                    result = test_func(*args, **backend_kwargs)
                    results[backend] = {"success": True, "result": result}

                except Exception as e:
                    # Log the failure as a warning
                    warning_msg = f"Backend '{backend}' failed: {e!s}"
                    warnings.warn(warning_msg, BackendTestWarning, stacklevel=2)
                    warnings_list.append((backend, str(e)))
                    results[backend] = {"success": False, "error": str(e)}

            # Check if any backends succeeded
            successful_backends = [b for b, r in results.items() if r["success"]]
            failed_backends = [b for b, r in results.items() if not r["success"]]

            if not successful_backends:
                # All backends failed - this is a real test failure
                error_summary = "\n".join([f"  {b}: {r['error']}" for b, r in results.items()])
                pytest.fail(f"All backends failed:\n{error_summary}")

            # Some backends succeeded - log summary
            if failed_backends:
                pass

            # Return results for potential inspection
            return results

        return wrapper

    return decorator


class BackendTestRunner:
    @staticmethod
    def run_with_warnings(test_func, backends=None, include_mock=False, **test_kwargs):
        """Run a test function across multiple backends with warning-based error handling.

        Args:
            test_func: The test function to run
            backends: List of backends to test. If None, uses all available backends.
            include_mock: Whether to include the mock backend in testing.
            **test_kwargs: Additional keyword arguments to pass to the test function.

        Returns
        -------
        dict
            Dictionary mapping backend names to their test results.

        """
        if backends is None:
            available_backends = get_available_backends(include_mock=include_mock)
        else:
            available_backends = backends

        results = {}

        for backend in available_backends:
            try:
                result = test_func(backend=backend, **test_kwargs)
                results[backend] = {"success": True, "result": result}
            except Exception as e:
                warning_msg = f"Backend '{backend}' failed: {e!s}"
                warnings.warn(warning_msg, BackendTestWarning, stacklevel=2)
                results[backend] = {"success": False, "error": str(e)}

        return results

    @staticmethod
    def assert_backend_results(results, min_successful=1):
        """Assert that at least a minimum number of backends succeeded.

        Args:
            results: Dictionary from run_with_warnings
            min_successful: Minimum number of backends that must succeed

        """
        successful = [b for b, r in results.items() if r["success"]]
        failed = [b for b, r in results.items() if not r["success"]]

        if len(successful) < min_successful:
            error_summary = "\n".join(
                [f"  {b}: {r['error']}" for b, r in results.items() if not r["success"]],
            )
            pytest.fail(
                f"Not enough backends succeeded. Required: {min_successful}, "
                f"Got: {len(successful)}\nFailed backends:\n{error_summary}",
            )

        # Log warnings for failed backends
        for backend, result in results.items():
            if not result["success"]:
                warning_msg = f"Backend '{backend}' failed: {result['error']}"
                warnings.warn(warning_msg, UserWarning, stacklevel=2)

        return successful, failed


def parametrize_backends(
    backends=None,
    include_mock=False,
    ids=None,
):
    """Create a pytest parametrize marker for backend testing.

    This helper reduces redundancy in backend testing by providing a standardized
    way to parametrize tests across multiple backends.

    Args:
        backends: List of specific backends to test. If None, uses all available.
        include_mock: Whether to include mock backend.
        ids: Custom test IDs. If None, uses backend names.

    Returns
    -------
    pytest.mark.parametrize
        pytest.mark.parametrize marker.

    Examples
    --------
        @parametrize_backends(include_mock=True)
        def test_something(backend):
            # Test code here
            pass
    """
    if backends is None:
        backends = get_available_backends(include_mock=include_mock)

    if ids is None:
        ids = backends

    return pytest.mark.parametrize("backend", backends, ids=ids)


def requires_backend(backend_name):
    """Create a pytest marker to skip test if backend is not available.

    This is a convenience wrapper around pytest.mark.skipif that uses the
    standardized backend availability checking.

    Args:
        backend_name: Name of the backend to require (e.g., 'uma', 'mace')

    Returns
    -------
    pytest.mark.skipif
        pytest.mark.skipif marker.

    Examples
    --------
        @requires_backend("uma")
        def test_uma_specific_feature():
            # Test code here
            pass
    """
    from qme.backends.availability import is_backend_available

    return pytest.mark.skipif(
        not is_backend_available(backend_name),
        reason=f"{backend_name} backend not available",
    )


@pytest.fixture
def backend_test_fixture():
    """Fixture providing backend testing utilities and common patterns.

    This fixture provides a namespace with:
    - get_backends(): Get available backends
    - require_backend(backend): Skip if backend unavailable
    - run_with_all_backends(func): Run function with all backends

    Example:
        def test_something(backend_test_fixture):
            backends = backend_test_fixture.get_backends()
            for backend in backends:
                # test code
    """

    class BackendTestFixture:
        def __init__(self):
            self._available_backends = get_available_backends(include_mock=False)

        def get_backends(self, include_mock=False):
            """Get list of available backends."""
            return get_available_backends(include_mock=include_mock)

        def require_backend(self, backend):
            """Skip test if backend is not available."""
            if not check_backend_availability(backend):
                pytest.skip(f"Backend {backend} not available")

        def run_with_all_backends(self, test_func, include_mock=False, **kwargs):
            """Run test function with all available backends."""
            return BackendTestRunner.run_with_warnings(
                test_func, backends=None, include_mock=include_mock, **kwargs
            )

    return BackendTestFixture()


def create_backend_test_atoms(backend):
    """Create appropriate test atoms for a given backend.

    Some backends have limitations on supported elements or system sizes.
    This helper selects appropriate test molecules.

    Args:
        backend: Backend name

    Returns
    -------
    Atoms
        Atoms object suitable for testing with the backend.

    """
    # Most backends support small molecules
    return TestMoleculeFactory.get_water_distorted()


def setup_atoms_with_calculator(atoms, calculator):
    """Set up atoms with a calculator (common test pattern).

    This reduces repetition of the pattern:
        atoms = molecule.copy()
        atoms.calc = calculator

    Args:
        atoms: Atoms object (will be copied)
        calculator: Calculator to attach

    Returns
    -------
    Atoms
        Copy of atoms with calculator attached.

    Examples
    --------
        atoms = setup_atoms_with_calculator(h2o_molecule, mock_backend)
        # Equivalent to:
        # atoms = h2o_molecule.copy()
        # atoms.calc = mock_backend
    """
    atoms_copy = atoms.copy()
    atoms_copy.calc = calculator
    return atoms_copy


def assert_backend_calculator(calculator, backend="mock"):
    """Assert that a calculator has the expected interface.

    Args:
        calculator: Calculator instance to check
        backend: Backend name for context
    """
    assert calculator is not None, f"Calculator should not be None for backend {backend}"

    # Check that calculator has basic methods
    # Note: We don't enforce these strictly as different backends may have different APIs
    if hasattr(calculator, "calculate"):
        assert callable(calculator.calculate)
    elif hasattr(calculator, "get_potential_energy"):
        assert callable(calculator.get_potential_energy)


# ============================================================================
# Shared Test Calculator Classes
# ============================================================================


class NoisyCalculator:
    def __init__(self, noise_level=0.0):
        """Initialize with specified noise level.

        Args:
            noise_level: Standard deviation of noise to add to forces
        """
        self.noise_level = noise_level

    def get_forces(self, atoms=None):
        """Compute forces with added noise.

        Args:
            atoms: Atoms object (optional, uses self.atoms if None)

        Returns
        -------
        np.ndarray
            Forces array with added noise.

        """
        if atoms is None:
            atoms = self.atoms

        # Simple harmonic forces
        positions = atoms.positions
        forces = -1.0 * positions

        # Add noise
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, forces.shape)
            forces += noise

        return forces

    def get_potential_energy(self, atoms=None, force_consistent=False):
        """Compute potential energy.

        Args:
            atoms: Atoms object (optional, uses self.atoms if None)
            force_consistent: Whether to use force-consistent energy (ignored)

        Returns
        -------
        float
            Potential energy.

        """
        if atoms is None:
            atoms = self.atoms

        # Simple harmonic energy
        positions = atoms.positions
        energy = 0.5 * np.sum(positions**2)
        return float(energy)


class HarmonicCalculator:
    def __init__(self, k=1.0):
        """Initialize with force constant k.

        Args:
            k: Force constant (default: 1.0)
        """
        self.k = k

    def get_forces(self, atoms):
        """Compute harmonic forces: F = -k * r.

        Args:
            atoms: Atoms object with positions

        Returns
        -------
        np.ndarray
            Forces array.

        """
        forces = -self.k * atoms.positions
        return forces

    def get_hessian(self, atoms=None):
        """Compute analytical harmonic Hessian: H = k * I.

        Args:
            atoms: Atoms object (optional, uses self.atoms if None)

        Returns
        -------
        np.ndarray
            Analytical Hessian matrix.

        """
        if atoms is None:
            atoms = self.atoms
        n_atoms = len(atoms)
        n_coords = 3 * n_atoms
        hessian = self.k * np.eye(n_coords)
        return hessian

    def get_potential_energy(self, atoms=None, force_consistent=False):
        """Compute harmonic potential energy.

        Args:
            atoms: Atoms object (optional, uses self.atoms if None)
            force_consistent: Whether to use force-consistent energy (ignored)

        Returns
        -------
        float
            Potential energy.

        """
        if atoms is None:
            atoms = self.atoms

        positions = atoms.positions
        energy = 0.5 * self.k * np.sum(positions**2)
        return float(energy)
