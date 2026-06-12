from __future__ import annotations


class TestPackageDependencies:
    def test_core_dependencies_importable(self):
        # Core dependencies: ase, numpy, click, requests, sella
        import ase  # noqa: F401
        import click  # noqa: F401
        import numpy  # noqa: F401
        import requests  # noqa: F401
        import sella  # noqa: F401

    def test_famex_package_importable(self):
        import famex  # noqa: F401

        # Verify basic package attributes exist
        assert hasattr(famex, "__version__")
        assert hasattr(famex, "Explorer")
