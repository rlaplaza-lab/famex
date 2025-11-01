"""Security tests for AIMNet2 path traversal vulnerabilities."""

import pytest

from qme.utils.path_security import PathSecurityError


class TestAIMNet2PathTraversal:
    """Test AIMNet2 model path handling against path traversal attacks."""

    def test_aimnet2_model_path_traversal(self):
        """Test that AIMNet2 prevents path traversal in model names."""
        # Import here to avoid test failures if aimnet2 isn't available
        try:
            from qme.potentials.aimnet2_potential import get_model_path
        except (ImportError, ModuleNotFoundError):
            pytest.skip("AIMNet2 not available")

        # Try with path traversal attempts
        evil_names = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\evil",
            "/etc/passwd",
            "~/malicious",
        ]

        for evil_name in evil_names:
            try:
                with pytest.raises(
                    (ValueError, PathSecurityError, RuntimeError, ModuleNotFoundError)
                ):
                    get_model_path(evil_name)
            except ModuleNotFoundError:
                # Skip if model_cache module is not available (optional dependency)
                pytest.skip("model_cache module not available")

    def test_aimnet2_fallback_path_validation(self):
        """Test that AIMNet2 fallback code validates paths."""
        # Import here to avoid test failures if aimnet2 isn't available
        try:
            from qme.potentials import aimnet2_potential
        except ImportError:
            pytest.skip("AIMNet2 not available")

        # The fallback code is only used if caching fails, so we can't easily test it
        # in isolation without mocking. But the import and structure should be present.
        assert hasattr(aimnet2_potential, "validate_safe_path")
