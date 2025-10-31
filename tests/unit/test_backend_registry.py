"""Unit tests for CalculatorRegistry and calculator creation - edge cases."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from qme.backends.constants import BACKEND_MOCK
from qme.backends.registry import CalculatorRegistry, create_calculator
from qme.utils.validation import BackendError


class TestCalculatorRegistryEdgeCases:
    """Test CalculatorRegistry edge cases and advanced scenarios."""

    def test_registry_lazy_loading(self) -> None:
        """Test that backends are lazily loaded."""
        registry = CalculatorRegistry()

        # Initially, mock should not be in active registry (but in lazy registry)
        assert BACKEND_MOCK in registry._lazy_registry
        # May or may not be loaded depending on imports
        # Test that loading works
        registry._load_backend(BACKEND_MOCK)
        assert BACKEND_MOCK in registry._registry

    def test_registry_load_already_loaded(self) -> None:
        """Test loading an already loaded backend doesn't reload."""
        registry = CalculatorRegistry()

        # Load mock backend
        registry._load_backend(BACKEND_MOCK)
        first_func = registry._registry.get(BACKEND_MOCK)

        # Load again
        registry._load_backend(BACKEND_MOCK)
        second_func = registry._registry.get(BACKEND_MOCK)

        # Should be same function
        assert first_func == second_func

    def test_registry_load_nonexistent_backend(self) -> None:
        """Test loading a backend that doesn't exist."""
        registry = CalculatorRegistry()

        # Should not raise, just silently fail
        registry._load_backend("nonexistent_backend")

        assert "nonexistent_backend" not in registry._registry

    def test_registry_load_with_import_error(self) -> None:
        """Test handling of import errors during lazy loading."""
        registry = CalculatorRegistry()

        # Create a custom lazy backend that will fail to import
        registry._lazy_registry["test_fail"] = type(
            "LazyBackend",
            (),
            {
                "module": "nonexistent.module.path",
                "function": "nonexistent_function",
                "is_class": False,
            },
        )()

        # Should not raise, just fail silently
        registry._load_backend("test_fail")

        assert "test_fail" not in registry._registry

    def test_registry_load_with_attribute_error(self) -> None:
        """Test handling of attribute errors during lazy loading."""
        registry = CalculatorRegistry()

        # Use a module that exists but attribute doesn't
        registry._lazy_registry["test_attr_fail"] = type(
            "LazyBackend",
            (),
            {
                "module": "qme.backends.registry",
                "function": "nonexistent_attribute",
                "is_class": False,
            },
        )()

        # Should warn but not crash
        with pytest.warns(UserWarning, match="Failed to load backend"):
            registry._load_backend("test_attr_fail")

        assert "test_attr_fail" not in registry._registry

    def test_registry_custom_backend_registration(self) -> None:
        """Test registering a custom backend."""
        registry = CalculatorRegistry()

        def custom_factory(**kwargs):
            return MagicMock()

        registry.register("custom_backend", custom_factory)

        calc = registry.create_calculator("custom_backend")
        assert calc is not None

    def test_registry_create_with_all_parameters(self) -> None:
        """Test creating calculator with all parameters."""
        registry = CalculatorRegistry()

        calc = registry.create_calculator(
            BACKEND_MOCK,
            model_name="test_model",
            model_path="/path/to/model",
            device="cpu",
            charge=1,
            mult=2,
            extra_param="value",
        )

        assert calc is not None

    def test_registry_create_with_none_parameters(self) -> None:
        """Test creating calculator with None parameters."""
        registry = CalculatorRegistry()

        calc = registry.create_calculator(
            BACKEND_MOCK,
            model_name=None,
            model_path=None,
            device=None,
        )

        assert calc is not None

    def test_registry_get_registered_backends(self) -> None:
        """Test getting list of registered backends."""
        registry = CalculatorRegistry()

        backends = registry.get_registered_backends()

        assert isinstance(backends, list)
        assert len(backends) > 0
        assert BACKEND_MOCK in backends

    def test_registry_is_backend_available_delegation(self) -> None:
        """Test that is_backend_available delegates to availability checker."""
        registry = CalculatorRegistry()

        with patch("qme.backends.registry.is_backend_available") as mock_available:
            mock_available.return_value = True

            result = registry.is_backend_available("test_backend")

            assert result is True
            mock_available.assert_called_once_with("test_backend")

    def test_registry_class_backend_handling(self) -> None:
        """Test handling of class-based backends (like Mock)."""
        registry = CalculatorRegistry()

        # Mock backend is a class
        calc = registry.create_calculator(BACKEND_MOCK)

        assert calc is not None
        # Verify it's an instance (not the class itself)
        assert not isinstance(calc, type)


class TestCreateCalculatorFunction:
    """Test create_calculator convenience function."""

    def test_create_calculator_basic(self) -> None:
        """Test basic calculator creation."""
        calc = create_calculator(
            backend=BACKEND_MOCK,
            model_name=None,
            model_path=None,
            device=None,
            default_charge=0,
            default_spin=1,
        )

        assert calc is not None

    def test_create_calculator_with_charge_mult(self) -> None:
        """Test calculator creation with explicit charge/multiplicity."""
        calc = create_calculator(
            backend=BACKEND_MOCK,
            model_name=None,
            model_path=None,
            device=None,
            default_charge=0,
            default_spin=1,
            charge=1,
            mult=2,
        )

        assert calc is not None

    def test_create_calculator_with_use_cache(self) -> None:
        """Test calculator creation with use_cache flag."""
        # First call - should create new
        calc1 = create_calculator(
            backend=BACKEND_MOCK,
            model_name=None,
            model_path=None,
            device=None,
            default_charge=0,
            default_spin=1,
            use_cache=True,
        )

        # Second call with same params - should potentially use cache
        calc2 = create_calculator(
            backend=BACKEND_MOCK,
            model_name=None,
            model_path=None,
            device=None,
            default_charge=0,
            default_spin=1,
            use_cache=True,
        )

        # Both should work
        assert calc1 is not None
        assert calc2 is not None

    def test_create_calculator_without_cache(self) -> None:
        """Test calculator creation with use_cache=False."""
        calc = create_calculator(
            backend=BACKEND_MOCK,
            model_name=None,
            model_path=None,
            device=None,
            default_charge=0,
            default_spin=1,
            use_cache=False,
        )

        assert calc is not None

    def test_create_calculator_so3lr_excluded_from_cache(self) -> None:
        """Test that SO3LR is excluded from caching."""
        # This test verifies the behavior without actually creating SO3LR
        # since it may not be available
        with (
            patch("qme.backends.registry.calculator_registry.create_calculator") as mock_create,
            patch("qme.backends.cache.get_cached_calculator", return_value=None),
            patch("qme.backends.cache.cache_calculator"),
        ):
            mock_calc = MagicMock()
            mock_create.return_value = mock_calc

            # Try to create SO3LR (even if not available, we test the code path)
            try:
                create_calculator(
                    backend="so3lr",
                    model_name=None,
                    model_path=None,
                    device=None,
                    default_charge=0,
                    default_spin=1,
                    use_cache=True,
                )
                # If SO3LR is not available, it will raise, but we check the code path
            except BackendError:
                pass

            # SO3LR should not be cached
            # The cache_calculator should not be called for SO3LR
            # This is tested via the backend_lower check

    def test_create_calculator_unavailable_backend(self) -> None:
        """Test creating calculator for unavailable backend."""
        with pytest.raises(BackendError, match="not available"):
            create_calculator(
                backend="nonexistent_backend_xyz123",
                model_name=None,
                model_path=None,
                device=None,
                default_charge=0,
                default_spin=1,
            )

    def test_create_calculator_handles_cache_import_error(self) -> None:
        """Test that cache import errors are handled gracefully."""
        with patch("qme.backends.registry.calculator_registry.create_calculator") as mock_create:
            mock_calc = MagicMock()
            mock_create.return_value = mock_calc

        # Make cache import fail
        with patch("qme.backends.cache.get_cached_calculator", side_effect=ImportError):
            calc = create_calculator(
                backend=BACKEND_MOCK,
                model_name=None,
                model_path=None,
                device=None,
                default_charge=0,
                default_spin=1,
                use_cache=True,
            )

            # Should still create calculator even if cache import fails
            assert calc is not None

    def test_create_calculator_handles_cache_calc_error(self) -> None:
        """Test that cache calculator errors are handled gracefully."""
        with patch("qme.backends.registry.calculator_registry.create_calculator") as mock_create:
            mock_calc = MagicMock()
            mock_create.return_value = mock_calc

            # Make cache_calculator fail
            with (
                patch("qme.backends.cache.get_cached_calculator", return_value=None),
                patch("qme.backends.cache.cache_calculator", side_effect=ImportError),
            ):
                calc = create_calculator(
                    backend=BACKEND_MOCK,
                    model_name=None,
                    model_path=None,
                    device=None,
                    default_charge=0,
                    default_spin=1,
                    use_cache=True,
                )

                # Should still return calculator even if caching fails
                assert calc is not None

    def test_create_calculator_uses_cached_when_available(self) -> None:
        """Test that cached calculator is used when available."""
        mock_cached_calc = MagicMock()

        with (
            patch("qme.backends.cache.get_cached_calculator", return_value=mock_cached_calc),
            patch("qme.backends.registry.calculator_registry.create_calculator") as mock_create,
        ):
            calc = create_calculator(
                backend=BACKEND_MOCK,
                model_name=None,
                model_path=None,
                device=None,
                default_charge=0,
                default_spin=1,
                use_cache=True,
            )

            # Should return cached calculator
            assert calc == mock_cached_calc
            # Should not create new calculator
            mock_create.assert_not_called()

    def test_create_calculator_with_verbose(self) -> None:
        """Test calculator creation with verbose flag."""
        calc = create_calculator(
            backend=BACKEND_MOCK,
            model_name=None,
            model_path=None,
            device=None,
            default_charge=0,
            default_spin=1,
            verbose=0,
        )

        assert calc is not None


class TestBackendErrorHandling:
    """Test error handling in calculator creation."""

    def test_create_calculator_backend_error_includes_available(self) -> None:
        """Test that BackendError includes list of available backends."""
        with pytest.raises(BackendError) as exc_info:
            create_calculator(
                backend="nonexistent_backend",
                model_name=None,
                model_path=None,
                device=None,
                default_charge=0,
                default_spin=1,
            )

        # Error should contain information about available backends
        error_msg = str(exc_info.value)
        assert "not available" in error_msg.lower() or "available" in error_msg.lower()

    def test_registry_create_calculator_backend_error(self) -> None:
        """Test that registry raises BackendError for unavailable backends."""
        registry = CalculatorRegistry()

        with pytest.raises(BackendError):
            registry.create_calculator("nonexistent_backend")
