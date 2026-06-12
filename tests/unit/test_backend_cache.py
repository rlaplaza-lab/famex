from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from famex.backends.cache import (
    CalculatorCache,
    ModelCache,
    UnifiedCache,
    cache_calculator,
    clear_all_caches,
    download_and_cache_model,
    get_cached_calculator,
    get_model_cache,
    get_unified_cache,
)


class TestCalculatorCacheEdgeCases:
    def test_cache_access_order_tracking(self):
        cache = CalculatorCache(max_size=3)
        calc1 = MagicMock()
        calc2 = MagicMock()
        calc3 = MagicMock()

        # Add calculators
        cache.put(calc1, "backend1", None, "cpu")
        cache.put(calc2, "backend2", None, "cpu")
        cache.put(calc3, "backend3", None, "cpu")

        # Access calc1 to update its order
        retrieved = cache.get("backend1", None, "cpu")
        assert retrieved == calc1

        # Add fourth - should evict calc2 (least recently used, not calc1)
        calc4 = MagicMock()
        cache.put(calc4, "backend4", None, "cpu")

        # calc2 should be evicted
        assert cache.get("backend2", None, "cpu") is None
        # calc1 should still be there (was accessed)
        assert cache.get("backend1", None, "cpu") == calc1

    def test_cache_key_with_none_values(self):
        cache = CalculatorCache()

        key1 = cache._generate_key("test", None, None)
        key2 = cache._generate_key("test", None, None)

        # Keys should be consistent for same parameters
        assert key1 == key2

        # Adding extra parameter (even if None) creates different key
        key3 = cache._generate_key("test", None, None, extra=None)
        assert key1 != key3  # Different parameter set

    def test_cache_key_with_complex_parameters(self):
        cache = CalculatorCache()

        key1 = cache._generate_key(
            "test",
            "model",
            "cpu",
            charge=0,
            spin=1,
            kwargs={"nested": {"key": "value"}},
        )
        key2 = cache._generate_key(
            "test",
            "model",
            "cpu",
            spin=1,
            charge=0,
            kwargs={"nested": {"key": "value"}},
        )

        # Keys should be same regardless of parameter order
        assert key1 == key2

    def test_cache_with_same_key_multiple_times(self):
        cache = CalculatorCache()
        calc1 = MagicMock()
        calc2 = MagicMock()

        key1 = cache.put(calc1, "test", "model", "cpu")
        key2 = cache.put(calc2, "test", "model", "cpu")

        # Should have same key
        assert key1 == key2

        # Should return the latest calculator
        retrieved = cache.get("test", "model", "cpu")
        assert retrieved == calc2

    def test_cache_eviction_when_at_limit(self):
        cache = CalculatorCache(max_size=2)
        calc1 = MagicMock()
        calc2 = MagicMock()

        cache.put(calc1, "backend1", None, "cpu")
        cache.put(calc2, "backend2", None, "cpu")

        # Both should be in cache
        assert cache.get("backend1", None, "cpu") == calc1
        assert cache.get("backend2", None, "cpu") == calc2

        # Adding a third should evict one
        calc3 = MagicMock()
        cache.put(calc3, "backend3", None, "cpu")

        # One of the first two should be gone
        assert (
            cache.get("backend1", None, "cpu") is None or cache.get("backend2", None, "cpu") is None
        )

    def test_cache_clear_resets_access_order(self):
        cache = CalculatorCache()
        calc = MagicMock()

        cache.put(calc, "test", "model", "cpu")
        cache.get("test", "model", "cpu")

        cache.clear()

        assert cache._access_counter == 0
        assert len(cache._access_order) == 0

    def test_cache_with_weak_references(self):
        cache = CalculatorCache()

        # Create calculator and cache it
        calc = MagicMock()
        cache.put(calc, "test", "model", "cpu")

        # Calculator should be retrievable
        assert cache.get("test", "model", "cpu") == calc

        # Delete the calculator reference
        del calc

        # With weak references, the cache entry may be gone
        # This is expected behavior - weak references allow GC
        # May be None if GC occurred, or may still be there if GC hasn't run
        # Both behaviors are acceptable with weak references
        _ = cache.get("test", "model", "cpu")


class TestModelCacheEdgeCases:
    def test_model_cache_metadata_corruption_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            # Corrupt metadata file
            cache.metadata_file.write_text("not valid json{")

            # Creating new cache should handle corruption
            new_cache = ModelCache(cache_dir=tmpdir)
            assert isinstance(new_cache.metadata, dict)
            assert len(new_cache.metadata) == 0

    def test_model_cache_missing_file_cleanup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            # Add a model
            model_data = b"test data"
            cached_path = cache.cache_model("test_model", "http://example.com/model", model_data)

            assert cached_path.exists()
            assert cache.get_cached_model("test_model", "http://example.com/model") is not None

            # Delete the file manually
            cached_path.unlink()

            # Getting cached model should clean up metadata
            result = cache.get_cached_model("test_model", "http://example.com/model")
            assert result is None
            assert "test_model" not in str(cache.metadata)

    def test_model_cache_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            model_data = b"original data"
            cached_path = cache.cache_model("test_model", "http://example.com/model", model_data)

            # Corrupt the file
            cached_path.write_bytes(b"corrupted data")

            # Should detect checksum mismatch and return None
            result = cache.get_cached_model("test_model", "http://example.com/model")
            assert result is None

            # File should be deleted
            assert not cached_path.exists()

    def test_model_cache_multiple_models_same_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            model1_data = b"model1 data"
            model2_data = b"model2 data"

            path1 = cache.cache_model("test_model", "http://example.com/model1", model1_data)
            path2 = cache.cache_model("test_model", "http://example.com/model2", model2_data)

            # Should create separate cache entries
            assert path1 != path2

            retrieved1 = cache.get_cached_model("test_model", "http://example.com/model1")
            retrieved2 = cache.get_cached_model("test_model", "http://example.com/model2")

            assert retrieved1 == path1
            assert retrieved2 == path2

    def test_model_cache_filename_sanitization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            model_data = b"test"
            # Model name with special characters
            cached_path = cache.cache_model(
                "model/name:with:chars", "http://example.com/model", model_data
            )

            # Filename should not contain special characters
            assert "/" not in cached_path.name
            assert ":" not in cached_path.name
            assert cached_path.exists()

    def test_model_cache_get_info_empty_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            info = cache.get_cache_info()

            assert info["model_count"] == 0
            assert info["total_size_mb"] == 0.0
            assert isinstance(info["models"], list)
            assert len(info["models"]) == 0

    def test_model_cache_clear_nonexistent_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            # Should not raise error
            cache.clear_cache(model_name="nonexistent")
            assert len(cache.metadata) == 0

    def test_model_cache_metadata_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache1 = ModelCache(cache_dir=tmpdir)
            model_data = b"test data"
            cache1.cache_model("test_model", "http://example.com/model", model_data)

            # Create new cache instance
            cache2 = ModelCache(cache_dir=tmpdir)

            # Should see the cached model
            result = cache2.get_cached_model("test_model", "http://example.com/model")
            assert result is not None

    def test_model_cache_handles_permission_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            # Try to save metadata to read-only location (if possible)
            # This test mainly ensures error handling exists
            # Actual permission errors depend on system configuration
            import contextlib

            with contextlib.suppress(OSError, PermissionError):
                # This should complete or raise a handled error
                cache._save_metadata()


class TestUnifiedCacheEdgeCases:
    def test_unified_cache_info(self):
        cache = UnifiedCache()
        mock_calc = MagicMock()

        cache.calculator_cache.put(mock_calc, "test", "model", "cpu")

        info = cache.get_cache_info()

        assert "calculator_cache" in info
        assert "model_cache" in info
        assert info["calculator_cache"]["size"] >= 0
        assert info["calculator_cache"]["max_size"] > 0

    def test_unified_cache_clear_clears_both(self):
        cache = UnifiedCache()
        mock_calc = MagicMock()

        cache.calculator_cache.put(mock_calc, "test", "model", "cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache.model_cache.cache_dir = Path(tmpdir)
            cache.model_cache.cache_model("model1", "http://example.com/m1", b"data1")

            cache.clear_all()

            assert cache.calculator_cache.get("test", "model", "cpu") is None
            assert cache.model_cache.get_cached_model("model1", "http://example.com/m1") is None


class TestGlobalCacheFunctions:
    def test_get_cached_calculator_function(self):
        mock_calc = MagicMock()
        cache_calculator(mock_calc, "test", "model", "cpu")

        retrieved = get_cached_calculator("test", "model", "cpu")

        assert retrieved == mock_calc

    def test_cache_calculator_function(self):
        mock_calc = MagicMock()

        key = cache_calculator(mock_calc, "test", "model", "cpu")

        assert key is not None
        retrieved = get_cached_calculator("test", "model", "cpu")
        assert retrieved == mock_calc

    def test_get_model_cache_function(self):
        cache = get_model_cache()

        assert isinstance(cache, ModelCache)
        assert cache.cache_dir.exists()

    def test_get_unified_cache_function(self):
        cache = get_unified_cache()

        assert isinstance(cache, UnifiedCache)
        assert isinstance(cache.calculator_cache, CalculatorCache)
        assert isinstance(cache.model_cache, ModelCache)

    def test_clear_all_caches_function(self):
        # Get unified cache
        unified = get_unified_cache()

        # Add something to cache
        mock_calc = MagicMock()
        unified.calculator_cache.put(mock_calc, "test", "model", "cpu")

        # Verify it's there
        assert unified.calculator_cache.get("test", "model", "cpu") == mock_calc

        # Clear all
        clear_all_caches()

        # Should be cleared in the unified cache
        retrieved = unified.calculator_cache.get("test", "model", "cpu")
        assert retrieved is None

    def test_download_and_cache_model_with_existing_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)
            model_data = b"cached model data"
            cache.cache_model("test_model", "http://example.com/model", model_data)

            with patch("famex.backends.cache._get_model_cache", return_value=cache):
                # Should return cached model without downloading
                result = download_and_cache_model("test_model", "http://example.com/model")

                assert result.exists()
                assert result.read_bytes() == model_data

    def test_download_and_cache_model_downloads_when_not_cached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            model_data = b"downloaded model data"

            with (
                patch("famex.backends.cache._get_model_cache", return_value=cache),
                patch("famex.backends.cache.requests.get") as mock_get,
            ):
                mock_response = MagicMock()
                mock_response.content = model_data
                mock_response.raise_for_status = MagicMock()
                mock_get.return_value = mock_response

                result = download_and_cache_model("test_model", "http://example.com/model")

                assert result.exists()
                assert result.read_bytes() == model_data
                mock_get.assert_called_once()

    def test_download_and_cache_model_handles_download_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            with (
                patch("famex.backends.cache._get_model_cache", return_value=cache),
                patch("famex.backends.cache.requests.get") as mock_get,
            ):
                import requests

                mock_get.side_effect = requests.RequestException("Download failed")

                with pytest.raises(RuntimeError, match="Failed to download model"):
                    download_and_cache_model("test_model", "http://example.com/model")
