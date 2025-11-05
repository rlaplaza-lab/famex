"""Test that core package dependencies can be imported.

This is a minimal smoke test to verify that the package's declared
dependencies are available and importable.
"""


def test_core_dependencies_importable() -> None:
    """Test that core dependencies listed in pyproject.toml are importable."""
    # Core dependencies: ase, numpy, click, requests, sella
    import ase  # noqa: F401
    import click  # noqa: F401
    import numpy  # noqa: F401
    import requests  # noqa: F401
    import sella  # noqa: F401


def test_qme_package_importable() -> None:
    """Test that the qme package itself can be imported."""
    import qme  # noqa: F401

    # Verify basic package attributes exist
    assert hasattr(qme, "__version__")
    assert hasattr(qme, "Explorer")
