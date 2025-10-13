"""
Backend test helpers for graceful handling of backend failures.

This module provides utilities to run tests across multiple backends with
graceful failure handling - individual backend failures result in warnings
rather than test failures.
"""

import functools
import warnings
from collections.abc import Callable
from typing import Any

import pytest

from qme.backend_availability import get_available_backends


class BackendTestWarning(UserWarning):
    """Warning raised when a backend test fails but should not fail the entire test."""


def backend_test_with_warnings(
    backends: list[str] | None = None,
    include_mock: bool = False,
    test_name_suffix: str = "",
) -> Callable:
    """
    Decorator to run a test across multiple backends with graceful failure handling.

    When a backend fails, it logs a warning but continues testing other backends.
    The test only fails if ALL backends fail.

    Args:
        backends: List of backends to test. If None, uses all available backends.
        include_mock: Whether to include the mock backend in testing.
        test_name_suffix: Suffix to add to test names for backend identification.

    Returns:
        Decorated test function that handles backend failures gracefully.
    """

    def decorator(test_func: Callable) -> Callable:
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
                    warning_msg = f"Backend '{backend}' failed: {str(e)}"
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
                print(f"\nBackend test summary for {test_func.__name__}:")
                print(f"  ✅ Successful: {', '.join(successful_backends)}")
                print(f"  ⚠️  Failed: {', '.join(failed_backends)}")
                print(f"  Total: {len(available_backends)} backends tested")

            # Return results for potential inspection
            return results

        return wrapper

    return decorator


# Utility function for existing tests
def run_backend_test_with_warnings(
    test_func: Callable,
    backends: list[str] | None = None,
    include_mock: bool = False,
    **test_kwargs,
) -> dict[str, Any]:
    """
    Run a test function across multiple backends with warning-based error handling.

    This is a utility function that can be called from within existing tests
    to test multiple backends without modifying the test structure.

    Args:
        test_func: The test function to run
        backends: List of backends to test. If None, uses all available backends.
        include_mock: Whether to include the mock backend in testing.
        **test_kwargs: Additional keyword arguments to pass to the test function.

    Returns:
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
            warning_msg = f"Backend '{backend}' failed: {str(e)}"
            warnings.warn(warning_msg, BackendTestWarning, stacklevel=2)
            results[backend] = {"success": False, "error": str(e)}

    return results
