from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from famex.backends.availability import (
    BackendAvailabilityChecker,
    clear_availability_cache,
    get_availability_reason,
    get_available_backends,
    get_backend_error_message,
    is_backend_available,
)
from famex.backends.cache import CalculatorCache, ModelCache, UnifiedCache
from famex.backends.constants import BACKEND_MOCK
from famex.backends.registry import _BACKEND_CLASSES, CalculatorRegistry, create_calculator
from famex.utils.validation import BackendError


class TestCalculatorRegistry:
    def test_registry_initialization(self):
        registry = CalculatorRegistry()
        assert len(registry._registry) == 0
        assert len(_BACKEND_CLASSES) > 0

    def test_get_registered_backends(self):
        registry = CalculatorRegistry()
        backends = registry.get_registered_backends()

        assert isinstance(backends, list)
        assert len(backends) > 0
        assert BACKEND_MOCK in backends

    def test_register_custom_backend(self):
        registry = CalculatorRegistry()

        def custom_factory(**kwargs):
            return MagicMock()

        registry.register("custom_backend", custom_factory)
        assert "custom_backend" in registry._registry

        calc = registry.create_calculator("custom_backend")
        assert calc is not None

    @pytest.mark.parametrize(
        "params",
        [
            {"model_name": "test_model", "device": "cpu"},
            {"model_name": None, "model_path": None, "device": None},
            {
                "model_name": "test_model",
                "model_path": "/path/to/model",
                "device": "cpu",
                "charge": 1,
                "mult": 2,
            },
        ],
        ids=["with_params", "with_none", "all_params"],
    )
    def test_create_calculator_with_params(self, params):
        registry = CalculatorRegistry()
        calc = registry.create_calculator(BACKEND_MOCK, **params)
        assert calc is not None

    def test_create_calculator_mock(self):
        registry = CalculatorRegistry()
        calc = registry.create_calculator(BACKEND_MOCK)
        assert calc is not None
        # Verify it's an instance (not the class itself)
        assert not isinstance(calc, type)

    def test_create_calculator_unavailable_backend(self):
        registry = CalculatorRegistry()
        with pytest.raises(BackendError, match="not available"):
            registry.create_calculator("nonexistent_backend")

    def test_is_backend_available_mock(self):
        registry = CalculatorRegistry()
        assert registry.is_backend_available(BACKEND_MOCK) is True

    @patch("famex.backends.registry.is_backend_available")
    def test_is_backend_available_delegation(self, mock_available):
        registry = CalculatorRegistry()
        mock_available.return_value = True

        result = registry.is_backend_available("test_backend")
        assert result is True
        mock_available.assert_called_once_with("test_backend")


class TestRegistryLazyLoading:
    def test_load_backend_lazy(self):
        registry = CalculatorRegistry()
        assert BACKEND_MOCK in _BACKEND_CLASSES

        registry._load_backend(BACKEND_MOCK)
        assert BACKEND_MOCK in registry._registry

    def test_load_backend_already_loaded(self):
        registry = CalculatorRegistry()

        registry._load_backend(BACKEND_MOCK)
        first_func = registry._registry.get(BACKEND_MOCK)

        registry._load_backend(BACKEND_MOCK)
        second_func = registry._registry.get(BACKEND_MOCK)

        assert first_func == second_func

    def test_load_nonexistent_backend(self):
        registry = CalculatorRegistry()
        registry._load_backend("nonexistent_backend")
        assert "nonexistent_backend" not in registry._registry

    def test_load_with_import_error(self):
        registry = CalculatorRegistry()

        with patch.dict(_BACKEND_CLASSES, {"test_fail": ("nonexistent.module.path", "Missing")}):
            registry._load_backend("test_fail")
            assert "test_fail" not in registry._registry

    def test_load_with_attribute_error(self):
        registry = CalculatorRegistry()

        with patch.dict(
            _BACKEND_CLASSES,
            {"test_attr_fail": ("famex.backends.registry", "nonexistent_attribute")},
        ):
            with pytest.warns(UserWarning, match="Failed to load backend"):
                registry._load_backend("test_attr_fail")

            assert "test_attr_fail" not in registry._registry

    @patch("famex.backends.registry.importlib.import_module")
    def test_load_backend_import_error(self, mock_import):
        registry = CalculatorRegistry()
        mock_import.side_effect = ImportError("Module not found")

        with patch.dict(_BACKEND_CLASSES, {"test_backend": ("fake.module", "FakeClass")}):
            registry._load_backend("test_backend")
            assert "test_backend" not in registry._registry


class TestCreateCalculatorFunction:
    @pytest.mark.parametrize(
        ("use_cache", "charge", "mult", "verbose"),
        [
            (False, 0, 1, 0),
            (True, 0, 1, 0),
            (False, 1, 2, 0),
            (False, 0, 1, 1),
        ],
        ids=["no_cache", "with_cache", "with_charge_mult", "verbose"],
    )
    def test_create_calculator_variations(self, use_cache, charge, mult, verbose):
        calc = create_calculator(
            backend=BACKEND_MOCK,
            model_name=None,
            model_path=None,
            device=None,
            default_charge=0,
            default_spin=1,
            charge=charge,
            mult=mult,
            use_cache=use_cache,
            verbose=verbose,
        )
        assert calc is not None

    def test_create_calculator_unavailable_backend(self):
        with pytest.raises(BackendError, match="not available"):
            create_calculator(
                backend="nonexistent_backend_xyz123",
                model_name=None,
                model_path=None,
                device=None,
                default_charge=0,
                default_spin=1,
            )

    def test_create_calculator_cache_behavior(self):
        calc1 = create_calculator(
            backend=BACKEND_MOCK,
            model_name=None,
            model_path=None,
            device=None,
            default_charge=0,
            default_spin=1,
            use_cache=True,
        )

        calc2 = create_calculator(
            backend=BACKEND_MOCK,
            model_name=None,
            model_path=None,
            device=None,
            default_charge=0,
            default_spin=1,
            use_cache=True,
        )

        # Both should work (may or may not be same object depending on cache)
        assert calc1 is not None
        assert calc2 is not None

    def test_create_calculator_uses_cached_when_available(self):
        mock_cached_calc = MagicMock()

        with (
            patch("famex.backends.cache.get_cached_calculator", return_value=mock_cached_calc),
            patch("famex.backends.registry.calculator_registry.create_calculator") as mock_create,
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

            assert calc == mock_cached_calc
            mock_create.assert_not_called()

    def test_create_calculator_handles_cache_errors(self):
        with patch("famex.backends.registry.calculator_registry.create_calculator") as mock_create:
            mock_calc = MagicMock()
            mock_create.return_value = mock_calc

            # Test cache import error
            with patch("famex.backends.cache.get_cached_calculator", side_effect=ImportError):
                calc = create_calculator(
                    backend=BACKEND_MOCK,
                    model_name=None,
                    model_path=None,
                    device=None,
                    default_charge=0,
                    default_spin=1,
                    use_cache=True,
                )
                assert calc is not None

            # Test cache calculator error
            with (
                patch("famex.backends.cache.get_cached_calculator", return_value=None),
                patch("famex.backends.cache.cache_calculator", side_effect=ImportError),
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
                assert calc is not None

    def test_create_calculator_backend_error_includes_available(self):
        with pytest.raises(BackendError) as exc_info:
            create_calculator(
                backend="nonexistent_backend",
                model_name=None,
                model_path=None,
                device=None,
                default_charge=0,
                default_spin=1,
            )

        error_msg = str(exc_info.value)
        assert "not available" in error_msg.lower() or "available" in error_msg.lower()


class TestBackendAvailability:
    def test_mock_always_available(self):
        assert is_backend_available(BACKEND_MOCK) is True

    def test_get_availability_reason_mock(self):
        reason = get_availability_reason(BACKEND_MOCK)
        assert "available" in reason.lower()

    @pytest.mark.parametrize("include_mock", [True, False], ids=["include", "exclude"])
    def test_get_available_backends(self, include_mock):
        backends = get_available_backends(include_mock=include_mock)
        assert isinstance(backends, list)
        if include_mock:
            assert BACKEND_MOCK in backends
        else:
            assert BACKEND_MOCK not in backends

    def test_clear_availability_cache(self):
        is_backend_available(BACKEND_MOCK)
        clear_availability_cache()

        checker = BackendAvailabilityChecker()
        assert len(checker._cache) == 0

    def test_get_backend_error_message(self):
        message = get_backend_error_message("nonexistent_backend")
        assert "nonexistent_backend" in message
        assert "available" in message.lower() or "install" in message.lower()

    def test_backend_availability_checker_init(self):
        checker = BackendAvailabilityChecker()
        assert isinstance(checker._cache, dict)
        assert isinstance(checker._conflict_cache, dict)
        assert len(checker._requirements) > 0

    def test_check_basic_dependencies(self):
        checker = BackendAvailabilityChecker()
        result = checker._check_basic_dependencies(BACKEND_MOCK)
        assert result is True

    @patch("famex.backends.availability.deps")
    def test_check_basic_dependencies_missing(self, mock_deps):
        checker = BackendAvailabilityChecker()
        mock_deps.has.return_value = False
        result = checker._check_basic_dependencies("uma")
        assert result is False

    def test_get_availability_reason_unavailable(self):
        checker = BackendAvailabilityChecker()
        with patch.object(checker, "_check_basic_dependencies", return_value=False):
            reason = checker.get_availability_reason("test_backend")
            assert "dependencies" in reason.lower() or "missing" in reason.lower()


class TestCalculatorCache:
    def test_cache_initialization(self):
        cache = CalculatorCache(max_size=5)
        assert cache.max_size == 5
        assert len(cache._cache) == 0

    def test_cache_key_generation(self):
        cache = CalculatorCache()
        key1 = cache._generate_key("test", "model", "cpu", param1=1, param2=2)
        key2 = cache._generate_key("test", "model", "cpu", param2=2, param1=1)
        assert key1 == key2

    def test_cache_put_get(self):
        cache = CalculatorCache()
        mock_calc = MagicMock()

        cache.put(mock_calc, "test", "model", "cpu")
        retrieved = cache.get("test", "model", "cpu")

        assert retrieved == mock_calc

    def test_cache_miss(self):
        cache = CalculatorCache()
        result = cache.get("test", "model", "cpu")
        assert result is None

    def test_cache_eviction(self):
        cache = CalculatorCache(max_size=2)

        calc1 = MagicMock()
        calc2 = MagicMock()
        cache.put(calc1, "backend1", None, "cpu")
        cache.put(calc2, "backend2", None, "cpu")

        calc3 = MagicMock()
        cache.put(calc3, "backend3", None, "cpu")

        assert (
            cache.get("backend1", None, "cpu") is None
            or cache.get("backend1", None, "cpu") != calc1
        )

    def test_cache_clear(self):
        cache = CalculatorCache()
        mock_calc = MagicMock()

        cache.put(mock_calc, "test", "model", "cpu")
        assert cache.get("test", "model", "cpu") == mock_calc

        cache.clear()
        assert cache.get("test", "model", "cpu") is None
        assert len(cache._cache) == 0


class TestModelCache:
    @pytest.mark.parametrize("custom_dir", [True, False], ids=["custom", "default"])
    def test_cache_initialization(self, custom_dir):
        if custom_dir:
            with tempfile.TemporaryDirectory() as tmpdir:
                cache = ModelCache(cache_dir=tmpdir)
                assert cache.cache_dir == Path(tmpdir)
        else:
            cache = ModelCache()
            assert cache.cache_dir.exists()
            assert cache.cache_dir.is_dir()

    def test_get_model_hash(self):
        cache = ModelCache(cache_dir=tempfile.mkdtemp())

        hash1 = cache._get_model_hash("model1", "http://example.com/model")
        hash2 = cache._get_model_hash("model1", "http://example.com/model")
        hash3 = cache._get_model_hash("model2", "http://example.com/model")

        assert hash1 == hash2
        assert hash1 != hash3

    def test_get_cached_model_not_cached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)
            result = cache.get_cached_model("test_model", "http://example.com/model")
            assert result is None

    def test_cache_model_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            model_data = b"fake model data"
            cached_path = cache.cache_model("test_model", "http://example.com/model", model_data)

            assert cached_path.exists()
            assert cached_path.read_bytes() == model_data

            retrieved_path = cache.get_cached_model("test_model", "http://example.com/model")
            assert retrieved_path == cached_path

    def test_cache_model_checksum_verification(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            model_data = b"fake model data"
            cached_path = cache.cache_model("test_model", "http://example.com/model", model_data)
            cached_path.write_bytes(b"corrupted data")

            retrieved_path = cache.get_cached_model("test_model", "http://example.com/model")
            assert retrieved_path is None or not cached_path.exists()

    def test_clear_cache_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            cache.cache_model("model1", "http://example.com/m1", b"data1")
            cache.cache_model("model2", "http://example.com/m2", b"data2")

            cache.clear_cache()

            assert len(list(cache.cache_dir.glob("*.jpt"))) == 0
            assert len(cache.metadata) == 0

    def test_clear_cache_specific_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            cache.cache_model("model1", "http://example.com/m1", b"data1")
            cache.cache_model("model2", "http://example.com/m2", b"data2")

            cache.clear_cache(model_name="model1")

            assert cache.get_cached_model("model1", "http://example.com/m1") is None
            assert cache.get_cached_model("model2", "http://example.com/m2") is not None

    def test_get_cache_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            cache.cache_model("model1", "http://example.com/m1", b"data1")
            info = cache.get_cache_info()

            assert "total_size_mb" in info
            assert "model_count" in info
            assert info["model_count"] >= 1


class TestUnifiedCache:
    def test_unified_cache_initialization(self):
        cache = UnifiedCache()
        assert isinstance(cache.calculator_cache, CalculatorCache)
        assert isinstance(cache.model_cache, ModelCache)

    def test_unified_cache_get_cached_calculator(self):
        cache = UnifiedCache()
        mock_calc = MagicMock()

        cache.calculator_cache.put(mock_calc, "test", "model", "cpu")
        retrieved = cache.calculator_cache.get("test", "model", "cpu")

        assert retrieved == mock_calc

    def test_unified_cache_clear_all(self):
        cache = UnifiedCache()
        mock_calc = MagicMock()

        cache.calculator_cache.put(mock_calc, "test", "model", "cpu")
        cache.model_cache.cache_model("model1", "http://example.com/m1", b"data1")

        cache.clear_all()

        assert cache.calculator_cache.get("test", "model", "cpu") is None


class TestDependencyManager:
    def test_dependency_manager_has(self):
        from famex.backends.dependencies import deps

        result = deps.has("nonexistent_package")
        assert isinstance(result, bool)

    def test_dependency_manager_get(self):
        from famex.backends.dependencies import deps

        if deps.has("numpy"):
            result = deps.get("numpy")
            assert result is not None

    def test_dependency_manager_structure(self):
        from famex.backends.dependencies import deps

        assert hasattr(deps, "has")
        assert hasattr(deps, "get")
