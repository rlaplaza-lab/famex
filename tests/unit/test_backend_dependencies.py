"""Tests for dependency manager."""

from __future__ import annotations

import pytest

from famex.backends.dependencies import PACKAGE_NAME_MAP, DependencyManager


@pytest.fixture
def manager():
    return DependencyManager()


class TestDependencyManager:
    """Tests for DependencyManager."""

    def test_initialization(self, manager):
        """Test DependencyManager initialization."""
        assert manager._cache == {}
        assert manager._availability_cache == {}

    def test_package_name_map_contains_pyscf_dispersion(self):
        assert PACKAGE_NAME_MAP["pyscf_dispersion"] == "pyscf.dispersion.dftd3"
        assert PACKAGE_NAME_MAP["fairchem"] == "fairchem.core"

    def test_has_sella(self, manager):
        """Test checking for sella dependency."""
        # sella is a core dependency, should be available
        assert manager.has("sella") is True

    def test_get_sella(self, manager):
        """Test getting sella dependency."""
        sella = manager.get("sella")
        assert sella is not None
        # Should be cached
        assert "sella" in manager._cache

    def test_has_nonexistent_package(self, manager):
        """Test checking for non-existent package."""
        assert manager.has("nonexistent_package_xyz123") is False

    def test_get_nonexistent_package(self, manager):
        """Test getting non-existent package returns None."""
        result = manager.get("nonexistent_package_xyz123", default=None)
        assert result is None

    def test_get_with_default(self, manager):
        """Test get() with custom default value."""
        default_value = "default"
        result = manager.get("nonexistent_package_xyz123", default=default_value)
        assert result == default_value

    def test_install_package_spec_sella(self):
        from famex.backends.constants import install_package_spec

        assert install_package_spec("sella") == "sella"

    def test_install_package_spec_torch(self):
        from famex.backends.constants import install_package_spec

        assert install_package_spec("torch") == "torch"

    def test_install_package_spec_backends(self):
        from famex.backends.constants import (
            BACKEND_AIMNET2,
            BACKEND_MACE,
            install_package_spec,
        )

        cmd_aimnet2 = install_package_spec(BACKEND_AIMNET2)
        assert "torch" in cmd_aimnet2

        cmd_mace = install_package_spec(BACKEND_MACE)
        assert "mace" in cmd_mace.lower()
        assert isinstance(cmd_aimnet2, str)
        assert isinstance(cmd_mace, str)

    def test_install_package_spec_unknown(self):
        from famex.backends.constants import install_package_spec

        assert install_package_spec("unknown_package") == "unknown_package"

    def test_require_sella(self, manager):
        """Test require() for sella (core dependency)."""
        sella = manager.require("sella", purpose="testing")
        assert sella is not None

    def test_require_nonexistent(self, manager):
        """Test require() for non-existent package raises ImportError."""
        from famex.core.exceptions import DependencyError

        with pytest.raises(ImportError, match="Missing dependency"):
            manager.require("nonexistent_package_xyz123", purpose="testing")
        with pytest.raises(DependencyError, match="Missing dependency"):
            manager.require("nonexistent_package_xyz123", purpose="testing")

    def test_require_so3lr_uses_full_install_command(self, manager):
        from famex.backends.constants import BACKEND_FULL_INSTALL_COMMANDS, BACKEND_SO3LR

        manager._availability_cache["so3lr"] = False
        with pytest.raises(ImportError) as exc_info:
            manager.require(BACKEND_SO3LR, purpose="testing")
        assert BACKEND_FULL_INSTALL_COMMANDS[BACKEND_SO3LR] in str(exc_info.value)

    def test_cache_behavior(self, manager):
        """Test that dependencies are cached after loading."""
        # Get sella twice
        sella1 = manager.get("sella")
        sella2 = manager.get("sella")
        # Should be same object (cached)
        assert sella1 is sella2
        assert "sella" in manager._cache

    def test_availability_cache_behavior(self, manager):
        """Test that availability checks are cached."""
        # Check availability twice
        result1 = manager.has("nonexistent_package_xyz123")
        result2 = manager.has("nonexistent_package_xyz123")
        # Should be same (cached)
        assert result1 == result2
        assert "nonexistent_package_xyz123" in manager._availability_cache


class TestDependencyManagerErrorHandling:
    """Tests for error handling in DependencyManager."""

    def test_load_dependency_import_error(self, manager):
        """Test _load_dependency handles ImportError gracefully."""
        # Try to load non-existent package
        result = manager._load_dependency("nonexistent_package_xyz123", fallback_value="fallback")
        assert result == "fallback"

    def test_load_dependency_generic_import(self, manager):
        """Test _load_dependency with generic import."""
        # Try to import a standard library module

        result = manager._load_dependency("sys")
        assert result is not None

    def test_check_availability_lazy_handles_errors(self, manager):
        """Test _check_availability_lazy handles various errors."""
        # Should handle invalid package names gracefully
        result = manager._check_availability_lazy("")
        assert isinstance(result, bool)

        # Should handle special characters
        result = manager._check_availability_lazy("package.with.dots")
        assert isinstance(result, bool)

    def test_has_with_backend_mapping(self, manager):
        """Test has() with backend name mapping."""
        from famex.backends.constants import BACKEND_AIMNET2, BACKEND_PET

        # Should map backend name to package name
        result = manager.has(BACKEND_AIMNET2)
        assert isinstance(result, bool)

        pet_result = manager.has(BACKEND_PET)
        assert isinstance(pet_result, bool)

    def test_install_package_spec_pet(self):
        from famex.backends.constants import BACKEND_PET, install_package_spec

        assert install_package_spec(BACKEND_PET) == "upet"
        assert install_package_spec("upet") == "upet"

    def test_get_with_lowercase(self, manager):
        """Test that get() behavior - note that get() doesn't normalize case."""
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
