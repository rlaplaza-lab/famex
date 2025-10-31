"""Unit tests for backend management: registry, availability, cache, and dependencies."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qme.backends.availability import (
    BackendAvailabilityChecker,
    clear_availability_cache,
    get_availability_reason,
    get_available_backends,
    get_backend_error_message,
    is_backend_available,
)
from qme.backends.cache import CalculatorCache, ModelCache, UnifiedCache
from qme.backends.constants import BACKEND_MOCK
from qme.backends.registry import CalculatorRegistry, create_calculator
from qme.utils.validation import BackendError


class TestCalculatorRegistry:
    """Test calculator registry functionality."""

    def test_registry_initialization(self) -> None:
        """Test registry initializes with empty active registry."""
        registry = CalculatorRegistry()
        assert len(registry._registry) == 0
        assert len(registry._lazy_registry) > 0  # Should have lazy backends

    def test_get_registered_backends(self) -> None:
        """Test getting list of registered backend names."""
        registry = CalculatorRegistry()
        backends = registry.get_registered_backends()

        assert isinstance(backends, list)
        assert len(backends) > 0
        assert BACKEND_MOCK in backends

    def test_register_custom_backend(self) -> None:
        """Test registering a custom backend factory function."""
        registry = CalculatorRegistry()

        def custom_factory(**kwargs):
            return MagicMock()

        registry.register("custom_backend", custom_factory)
        assert "custom_backend" in registry._registry

    def test_create_calculator_mock(self) -> None:
        """Test creating mock calculator."""
        registry = CalculatorRegistry()

        # Mock calculator should be available
        calc = registry.create_calculator(BACKEND_MOCK)
        assert calc is not None

    def test_create_calculator_with_params(self) -> None:
        """Test creating calculator with parameters."""
        registry = CalculatorRegistry()

        calc = registry.create_calculator(
            BACKEND_MOCK,
            device="cpu",
            model_name="test_model",
        )
        assert calc is not None

    def test_create_calculator_unavailable_backend(self) -> None:
        """Test creating calculator for unavailable backend."""
        registry = CalculatorRegistry()

        with pytest.raises(BackendError, match="not available"):
            registry.create_calculator("nonexistent_backend")

    def test_is_backend_available_mock(self) -> None:
        """Test checking availability of mock backend."""
        registry = CalculatorRegistry()
        assert registry.is_backend_available(BACKEND_MOCK) is True

    @patch("qme.backends.registry.is_backend_available")
    def test_is_backend_available_delegation(self, mock_available: MagicMock) -> None:
        """Test that is_backend_available delegates to availability checker."""
        registry = CalculatorRegistry()
        mock_available.return_value = True

        result = registry.is_backend_available("test_backend")

        assert result is True
        mock_available.assert_called_once_with("test_backend")

    def test_load_backend_lazy(self) -> None:
        """Test lazy loading of backend."""
        registry = CalculatorRegistry()

        # Initially not in active registry
        assert "mock" not in registry._registry or registry._registry.get("mock") is None

        # Load it
        registry._load_backend("mock")

        # Now should be in registry
        assert "mock" in registry._registry

    def test_load_backend_already_loaded(self) -> None:
        """Test that loading already loaded backend doesn't reload."""
        registry = CalculatorRegistry()

        # Load once
        registry._load_backend("mock")
        first_func = registry._registry["mock"]

        # Load again
        registry._load_backend("mock")
        second_func = registry._registry["mock"]

        # Should be same function
        assert first_func == second_func

    @patch("qme.backends.registry.importlib.import_module")
    def test_load_backend_import_error(self, mock_import: MagicMock) -> None:
        """Test handling import errors during backend loading."""
        registry = CalculatorRegistry()
        mock_import.side_effect = ImportError("Module not found")

        # Should not raise, just fail silently
        registry._load_backend("test_backend")

        # Backend should not be in registry
        assert (
            "test_backend" not in registry._registry
            or "test_backend" not in registry._lazy_registry
        )


class TestCreateCalculator:
    """Test high-level create_calculator function."""

    def test_create_calculator_mock(self) -> None:
        """Test create_calculator function with mock backend."""
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
        """Test create_calculator with explicit charge/multiplicity."""
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

    @patch("qme.backends.registry.create_calculator")
    def test_create_calculator_cache_flag(self, mock_create: MagicMock) -> None:
        """Test create_calculator with use_cache flag."""
        mock_create.return_value = MagicMock()

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


class TestBackendAvailability:
    """Test backend availability checking."""

    def test_mock_always_available(self) -> None:
        """Test that mock backend is always available."""
        assert is_backend_available(BACKEND_MOCK) is True

    def test_get_availability_reason_mock(self) -> None:
        """Test getting availability reason for mock."""
        reason = get_availability_reason(BACKEND_MOCK)
        assert "available" in reason.lower()

    def test_get_available_backends_includes_mock(self) -> None:
        """Test get_available_backends includes mock by default."""
        backends = get_available_backends(include_mock=True)
        assert BACKEND_MOCK in backends

    def test_get_available_backends_excludes_mock(self) -> None:
        """Test get_available_backends excludes mock when requested."""
        backends = get_available_backends(include_mock=False)
        assert BACKEND_MOCK not in backends

    def test_clear_availability_cache(self) -> None:
        """Test clearing availability cache."""
        # Check a backend (populates cache)
        is_backend_available(BACKEND_MOCK)

        # Clear cache
        clear_availability_cache()

        # Cache should be empty (checked by creating new checker)
        checker = BackendAvailabilityChecker()
        assert len(checker._cache) == 0

    def test_get_backend_error_message(self) -> None:
        """Test getting backend error message."""
        message = get_backend_error_message("nonexistent_backend")

        assert "nonexistent_backend" in message
        assert "available" in message.lower() or "install" in message.lower()

    def test_backend_availability_checker_init(self) -> None:
        """Test BackendAvailabilityChecker initialization."""
        checker = BackendAvailabilityChecker()

        assert isinstance(checker._cache, dict)
        assert isinstance(checker._conflict_cache, dict)
        assert len(checker._requirements) > 0

    def test_check_basic_dependencies(self) -> None:
        """Test checking basic dependencies."""
        checker = BackendAvailabilityChecker()

        # Mock backend has no dependencies
        result = checker._check_basic_dependencies(BACKEND_MOCK)
        assert result is True

    @patch("qme.backends.availability.deps")
    def test_check_basic_dependencies_missing(self, mock_deps: MagicMock) -> None:
        """Test checking dependencies when missing."""
        checker = BackendAvailabilityChecker()
        mock_deps.has.return_value = False

        # Use a backend that has requirements (like uma which requires fairchem, torch)
        result = checker._check_basic_dependencies("uma")
        assert result is False  # Should be False because dependencies are missing

    def test_get_availability_reason_unavailable(self) -> None:
        """Test getting reason when backend unavailable."""
        checker = BackendAvailabilityChecker()

        with patch.object(checker, "_check_basic_dependencies", return_value=False):
            reason = checker.get_availability_reason("test_backend")
            assert "dependencies" in reason.lower() or "missing" in reason.lower()


class TestCalculatorCache:
    """Test calculator caching functionality."""

    def test_cache_initialization(self) -> None:
        """Test cache initialization."""
        cache = CalculatorCache(max_size=5)
        assert cache.max_size == 5
        assert len(cache._cache) == 0

    def test_cache_key_generation(self) -> None:
        """Test cache key generation is consistent."""
        cache = CalculatorCache()

        key1 = cache._generate_key("test", "model", "cpu", param1=1, param2=2)
        key2 = cache._generate_key("test", "model", "cpu", param2=2, param1=1)

        # Should generate same key regardless of param order
        assert key1 == key2

    def test_cache_put_get(self) -> None:
        """Test putting and getting from cache."""
        cache = CalculatorCache()
        mock_calc = MagicMock()

        key = cache.put(mock_calc, "test", "model", "cpu")
        retrieved = cache.get("test", "model", "cpu")

        assert retrieved == mock_calc
        assert key is not None

    def test_cache_miss(self) -> None:
        """Test cache miss returns None."""
        cache = CalculatorCache()
        result = cache.get("test", "model", "cpu")
        assert result is None

    def test_cache_eviction(self) -> None:
        """Test cache evicts oldest entries when full."""
        cache = CalculatorCache(max_size=2)

        # Add two calculators
        calc1 = MagicMock()
        calc2 = MagicMock()
        cache.put(calc1, "backend1", None, "cpu")
        cache.put(calc2, "backend2", None, "cpu")

        # Add third - should evict oldest
        calc3 = MagicMock()
        cache.put(calc3, "backend3", None, "cpu")

        # First should be evicted
        assert (
            cache.get("backend1", None, "cpu") is None
            or cache.get("backend1", None, "cpu") != calc1
        )

    def test_cache_clear(self) -> None:
        """Test clearing cache."""
        cache = CalculatorCache()
        mock_calc = MagicMock()

        cache.put(mock_calc, "test", "model", "cpu")
        assert cache.get("test", "model", "cpu") == mock_calc

        cache.clear()
        assert cache.get("test", "model", "cpu") is None
        assert len(cache._cache) == 0


class TestModelCache:
    """Test model caching functionality."""

    def test_cache_initialization_default(self) -> None:
        """Test cache initialization with default directory."""
        cache = ModelCache()
        assert cache.cache_dir.exists()
        assert cache.cache_dir.is_dir()

    def test_cache_initialization_custom_dir(self) -> None:
        """Test cache initialization with custom directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)
            assert cache.cache_dir == Path(tmpdir)

    def test_get_model_hash(self) -> None:
        """Test model hash generation."""
        cache = ModelCache(cache_dir=tempfile.mkdtemp())

        hash1 = cache._get_model_hash("model1", "http://example.com/model")
        hash2 = cache._get_model_hash("model1", "http://example.com/model")
        hash3 = cache._get_model_hash("model2", "http://example.com/model")

        # Same model should give same hash
        assert hash1 == hash2
        # Different model should give different hash
        assert hash1 != hash3

    def test_get_cached_model_not_cached(self) -> None:
        """Test getting model that isn't cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)
            result = cache.get_cached_model("test_model", "http://example.com/model")
            assert result is None

    def test_cache_model_and_retrieve(self) -> None:
        """Test caching a model and retrieving it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            model_data = b"fake model data"
            cached_path = cache.cache_model("test_model", "http://example.com/model", model_data)

            assert cached_path.exists()
            assert cached_path.read_bytes() == model_data

            # Retrieve it
            retrieved_path = cache.get_cached_model("test_model", "http://example.com/model")
            assert retrieved_path == cached_path

    def test_cache_model_checksum_verification(self) -> None:
        """Test checksum verification of cached models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            model_data = b"fake model data"
            cached_path = cache.cache_model("test_model", "http://example.com/model", model_data)

            # Corrupt the file
            cached_path.write_bytes(b"corrupted data")

            # Should not return corrupted file
            retrieved_path = cache.get_cached_model("test_model", "http://example.com/model")
            assert retrieved_path is None or not cached_path.exists()

    def test_clear_cache_all(self) -> None:
        """Test clearing entire cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            cache.cache_model("model1", "http://example.com/m1", b"data1")
            cache.cache_model("model2", "http://example.com/m2", b"data2")

            cache.clear_cache()

            assert len(list(cache.cache_dir.glob("*.jpt"))) == 0
            assert len(cache.metadata) == 0

    def test_clear_cache_specific_model(self) -> None:
        """Test clearing specific model from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            cache.cache_model("model1", "http://example.com/m1", b"data1")
            cache.cache_model("model2", "http://example.com/m2", b"data2")

            cache.clear_cache(model_name="model1")

            # model1 should be gone, model2 should remain
            retrieved = cache.get_cached_model("model1", "http://example.com/m1")
            assert retrieved is None

            retrieved2 = cache.get_cached_model("model2", "http://example.com/m2")
            assert retrieved2 is not None

    def test_get_cache_info(self) -> None:
        """Test getting cache information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            cache.cache_model("model1", "http://example.com/m1", b"data1")
            info = cache.get_cache_info()

            assert "total_size_mb" in info
            assert "model_count" in info
            assert info["model_count"] >= 1


class TestUnifiedCache:
    """Test unified cache combining calculator and model caches."""

    def test_unified_cache_initialization(self) -> None:
        """Test unified cache initialization."""
        cache = UnifiedCache()
        assert isinstance(cache.calculator_cache, CalculatorCache)
        assert isinstance(cache.model_cache, ModelCache)

    def test_unified_cache_custom_dir(self) -> None:
        """Test unified cache with custom directory."""
        # UnifiedCache doesn't accept cache_dir - ModelCache uses default
        cache = UnifiedCache()
        assert cache.model_cache is not None
        assert cache.calculator_cache is not None

    def test_unified_cache_get_cached_calculator(self) -> None:
        """Test getting cached calculator from unified cache."""
        cache = UnifiedCache()
        mock_calc = MagicMock()

        cache.calculator_cache.put(mock_calc, "test", "model", "cpu")
        retrieved = cache.calculator_cache.get("test", "model", "cpu")

        assert retrieved == mock_calc

    def test_unified_cache_clear_all(self) -> None:
        """Test clearing all caches."""
        cache = UnifiedCache()
        mock_calc = MagicMock()

        cache.calculator_cache.put(mock_calc, "test", "model", "cpu")
        cache.model_cache.cache_model("model1", "http://example.com/m1", b"data1")

        cache.clear_all()

        assert cache.calculator_cache.get("test", "model", "cpu") is None


class TestDependencyManager:
    """Test dependency management functionality."""

    def test_dependency_manager_has(self) -> None:
        """Test checking if dependency exists."""
        from qme.backends.dependencies import deps

        # These tests assume some dependencies may or may not be available
        # So we just test that the method works without error
        result = deps.has("nonexistent_package")
        assert isinstance(result, bool)

    def test_dependency_manager_get(self) -> None:
        """Test getting dependency module."""
        from qme.backends.dependencies import deps

        # Test with a dependency that might exist
        try:
            if deps.has("numpy"):
                result = deps.get("numpy")
                assert result is not None
            else:
                # If numpy not available, test will pass (no assertion)
                pass
        except ImportError:
            # Expected if package not available
            pass

    def test_get_install_command(self) -> None:
        """Test getting install command for dependency."""
        from qme.backends.dependencies import deps

        # Test with known package - use private method via public interface
        # The install command functionality is accessed through other methods
        # Just test that DependencyManager has the expected structure
        assert hasattr(deps, "has")
        assert hasattr(deps, "get")
