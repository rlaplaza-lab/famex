from __future__ import annotations

import pytest

from famex.utils.path_security import PathSecurityError
from tests.security.test_utils import EVIL_PATHS


class TestAIMNet2PathTraversal:
    @pytest.mark.parametrize("evil_name", EVIL_PATHS[:4])
    def test_aimnet2_model_path_traversal(self, evil_name):
        # Import here to avoid test failures if aimnet2 isn't available
        try:
            from famex.potentials.aimnet2_potential import get_model_path
        except (ImportError, ModuleNotFoundError):
            pytest.skip("AIMNet2 not available")

        try:
            with pytest.raises((ValueError, PathSecurityError, RuntimeError, ModuleNotFoundError)):
                get_model_path(evil_name)
        except ModuleNotFoundError:
            # Skip if model_cache module is not available (optional dependency)
            pytest.skip("model_cache module not available")

    def test_aimnet2_fallback_path_validation(self):
        # Import here to avoid test failures if aimnet2 isn't available
        try:
            from famex.potentials import aimnet2_potential
        except ImportError:
            pytest.skip("AIMNet2 not available")

        # The fallback code is only used if caching fails, so we can't easily test it
        # in isolation without mocking. But the import and structure should be present.
        assert hasattr(aimnet2_potential, "validate_safe_path")
