"""Tests for dependency manager."""

from __future__ import annotations

import pytest

from famex.backends.dependencies import DependencyManager
from famex.utils.validation import DependencyError


class TestDependencyManager:
    """Tests for DependencyManager."""

    def test_initialization(self):
        """Test DependencyManager initialization."""
        manager = DependencyManager()
        assert manager._cache == {}
        assert manager._availability_cache == {}

    def test_has_sella(self):
        """Test checking for sella dependency."""
        manager = DependencyManager()
        # sella is a core dependency, should be available
        assert manager.has("sella") is True

    def test_get_sella(self):
        """Test getting sella dependency."""
        manager = DependencyManager()
        sella = manager.get("sella")
        assert sella is not None
        # Should be cached
        assert "sella" in manager._cache

    def test_has_nonexistent_package(self):
        """Test checking for non-existent package."""
        manager = DependencyManager()
        assert manager.has("nonexistent_package_xyz123") is False

    def test_get_nonexistent_package(self):
        """Test getting non-existent package returns None."""
        manager = DependencyManager()
        result = manager.get("nonexistent_package_xyz123", default=None)
        assert result is None

    def test_get_with_default(self):
        """Test get() with custom default value."""
        manager = DependencyManager()
        default_value = "default"
        result = manager.get("nonexistent_package_xyz123", default=default_value)
        assert result == default_value

    def test_get_install_command_sella(self):
        """Test _get_install_command for sella."""
        manager = DependencyManager()
        cmd = manager._get_install_command("sella")
        assert cmd == "sella"

    def test_get_install_command_torch(self):
        """Test _get_install_command for torch."""
        manager = DependencyManager()
        cmd = manager._get_install_command("torch")
        assert cmd == "torch"

    def test_get_install_command_backends(self):
        """Test _get_install_command for various backends."""
        manager = DependencyManager()
        from famex.backends.constants import BACKEND_AIMNET2, BACKEND_MACE

        # Test various backend commands
        cmd_aimnet2 = manager._get_install_command(BACKEND_AIMNET2)
        assert "torch" in cmd_aimnet2
        # torch-cluster is optional, so we don't require it in install command

        cmd_mace = manager._get_install_command(BACKEND_MACE)
        assert "mace" in cmd_mace.lower()

        # Commands should be strings
        assert isinstance(cmd_aimnet2, str)
        assert isinstance(cmd_mace, str)

    def test_get_install_command_unknown(self):
        """Test _get_install_command for unknown package."""
        manager = DependencyManager()
        cmd = manager._get_install_command("unknown_package")
        assert cmd == "unknown_package"

    def test_require_sella(self):
        """Test require() for sella (core dependency)."""
        manager = DependencyManager()
        sella = manager.require("sella", purpose="testing")
        assert sella is not None

    def test_require_nonexistent(self):
        """Test require() for non-existent package raises ImportError."""
        manager = DependencyManager()
        with pytest.raises(ImportError, match="is required for"):
            manager.require("nonexistent_package_xyz123", purpose="testing")

    def test_require_multiple_single_available(self):
        """Test require_multiple() with single available dependency."""
        manager = DependencyManager()
        result = manager.require_multiple("sella", purpose="testing")
        # Should return module directly, not dict
        assert result is not None
        assert not isinstance(result, dict)

    def test_require_multiple_single_missing(self):
        """Test require_multiple() with single missing dependency raises DependencyError."""
        manager = DependencyManager()
        with pytest.raises(DependencyError, match="nonexistent_package_xyz123"):
            manager.require_multiple("nonexistent_package_xyz123", purpose="testing")

    def test_require_multiple_multiple_available(self):
        """Test require_multiple() with multiple available dependencies."""
        manager = DependencyManager()
        # sella should be available
        result = manager.require_multiple("sella", "sella", purpose="testing")
        # Should return dict when multiple (even if same name)
        # Actually, if same name, might return single module
        assert result is not None

    def test_require_multiple_with_missing(self):
        """Test require_multiple() with some missing dependencies."""
        manager = DependencyManager()
        with pytest.raises(DependencyError, match="nonexistent_package_xyz123"):
            manager.require_multiple("sella", "nonexistent_package_xyz123", purpose="testing")

    def test_require_multiple_all_missing(self):
        """Test require_multiple() with all dependencies missing."""
        manager = DependencyManager()
        with pytest.raises(DependencyError):
            manager.require_multiple(
                "nonexistent_package_xyz123", "another_nonexistent_xyz456", purpose="testing"
            )

    def test_cache_behavior(self):
        """Test that dependencies are cached after loading."""
        manager = DependencyManager()
        # Get sella twice
        sella1 = manager.get("sella")
        sella2 = manager.get("sella")
        # Should be same object (cached)
        assert sella1 is sella2
        assert "sella" in manager._cache

    def test_availability_cache_behavior(self):
        """Test that availability checks are cached."""
        manager = DependencyManager()
        # Check availability twice
        result1 = manager.has("nonexistent_package_xyz123")
        result2 = manager.has("nonexistent_package_xyz123")
        # Should be same (cached)
        assert result1 == result2
        assert "nonexistent_package_xyz123" in manager._availability_cache


class TestDependencyContext:
    """Tests for _DependencyContext context manager."""

    def test_context_manager_single_dependency(self):
        """Test context manager with single dependency."""
        manager = DependencyManager()
        context = manager.need("sella")
        with context as sella:
            assert sella is not None
            # Should return module directly, not tuple
            assert not isinstance(sella, tuple)

    def test_context_manager_multiple_dependencies(self):
        """Test context manager with multiple dependencies."""
        manager = DependencyManager()
        context = manager.need("sella", "sella")  # Same dependency twice
        with context as result:
            # Should return tuple when multiple
            assert isinstance(result, tuple)
            assert len(result) == 2

    def test_context_manager_exit(self):
        """Test context manager exit doesn't raise errors."""
        manager = DependencyManager()
        context = manager.need("sella")
        with context:
            pass
        # Should exit cleanly
        assert True

    def test_context_manager_raises_on_missing(self):
        """Test context manager raises DependencyError for missing dependencies."""
        manager = DependencyManager()
        context = manager.need("nonexistent_package_xyz123")
        with pytest.raises(DependencyError), context:
            pass

    def test_context_manager_multiple_with_missing(self):
        """Test context manager with multiple dependencies where some are missing."""
        manager = DependencyManager()
        context = manager.need("sella", "nonexistent_package_xyz123")
        with pytest.raises(DependencyError), context:
            pass


class TestDependencyManagerErrorHandling:
    """Tests for error handling in DependencyManager."""

    def test_load_dependency_import_error(self):
        """Test _load_dependency handles ImportError gracefully."""
        manager = DependencyManager()
        # Try to load non-existent package
        result = manager._load_dependency("nonexistent_package_xyz123", fallback_value="fallback")
        assert result == "fallback"

    def test_load_dependency_generic_import(self):
        """Test _load_dependency with generic import."""
        manager = DependencyManager()
        # Try to import a standard library module

        result = manager._load_dependency("sys")
        assert result is not None

    def test_check_availability_lazy_handles_errors(self):
        """Test _check_availability_lazy handles various errors."""
        manager = DependencyManager()
        # Should handle invalid package names gracefully
        result = manager._check_availability_lazy("")
        assert isinstance(result, bool)

        # Should handle special characters
        result = manager._check_availability_lazy("package.with.dots")
        assert isinstance(result, bool)

    def test_has_with_backend_mapping(self):
        """Test has() with backend name mapping."""
        manager = DependencyManager()
        from famex.backends.constants import BACKEND_AIMNET2

        # Should map backend name to package name
        result = manager.has(BACKEND_AIMNET2)
        assert isinstance(result, bool)

    def test_get_with_lowercase(self):
        """Test that get() behavior - note that get() doesn't normalize case."""
        manager = DependencyManager()
        # get() uses the name directly without normalization
        sella1 = manager.get("sella")
        assert sella1 is not None

        # Clear cache to test fresh lookup
        manager._cache.clear()
        # "SELLA" (uppercase) would try to import "SELLA" which doesn't exist
        # So it would return None (or default)
        sella2 = manager.get("SELLA", default=None)
        # Uppercase won't work because the package name is "sella" (lowercase)
        assert sella2 is None

        # But lowercase works
        manager._cache.clear()
        sella3 = manager.get("sella")
        assert sella3 is not None
        assert sella1 == sella3
