"""Unified caching utilities for FAMEX backends.

This module provides both calculator instance caching and model file caching
to avoid recreating calculators and re-downloading models on every run.

Note: SO3LR calculators are excluded from caching due to internal state
issues that cause "vmap got inconsistent sizes" errors when reused.
This is a known limitation of the SO3LR backend's internal implementation.
"""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
from weakref import WeakValueDictionary

import requests

from famex.utils.logging import get_famex_logger
from famex.utils.path_security import PathSecurityError, sanitize_filename, validate_safe_path

logger = get_famex_logger(__name__)


class CalculatorCache:
    """Simple calculator cache using weak references.

    Important: This cache uses WeakValueDictionary, which means:
    - Calculators are stored with weak references
    - If no other references exist, calculators may be garbage collected
    - Cache entries will automatically disappear when calculators are GC'd
    - This prevents memory leaks but means cache may miss even if "cached"

    This design choice prioritizes memory safety over cache persistence.
    For guaranteed persistence, maintain a strong reference to your calculator.
    """

    def __init__(self, max_size: int = 10) -> None:
        """Initialize calculator cache.

        Parameters
        ----------
        max_size : int
            Maximum number of calculators to cache

        """
        self.max_size = max_size
        self._cache: WeakValueDictionary = WeakValueDictionary()
        self._access_order: dict[str, int] = {}
        self._access_counter = 0

    def _generate_key(
        self,
        backend: str,
        model_name: str | None,
        device: str | None,
        **kwargs: Any,
    ) -> str:
        """Generate a cache key for calculator parameters."""
        # Create a sorted dictionary of parameters for consistent hashing
        params = {
            "backend": backend,
            "model_name": model_name,
            "device": device,
            **kwargs,
        }

        # Sort parameters for consistent key generation
        sorted_params = sorted(params.items())
        param_str = str(sorted_params)

        # Generate hash
        return hashlib.md5(param_str.encode()).hexdigest()[:16]

    def get(
        self,
        backend: str,
        model_name: str | None,
        device: str | None,
        **kwargs: Any,
    ) -> Any | None:
        """Get cached calculator if available.

        Parameters
        ----------
        backend : str
            Calculator backend
        model_name : str, optional
            Model name
        device : str, optional
            Device specification
        **kwargs
            Additional calculator parameters

        Returns
        -------
        Calculator or None
            Cached calculator if available, None otherwise

        """
        key = self._generate_key(backend, model_name, device, **kwargs)

        if key in self._cache:
            # Update access order
            self._access_order[key] = self._access_counter
            self._access_counter += 1
            return self._cache[key]

        return None

    def put(
        self,
        calculator: Any,
        backend: str,
        model_name: str | None,
        device: str | None,
        **kwargs: Any,
    ) -> str:
        """Cache a calculator.

        Parameters
        ----------
        calculator : Any
            Calculator instance to cache
        backend : str
            Calculator backend
        model_name : str, optional
            Model name
        device : str, optional
            Device specification
        **kwargs
            Additional calculator parameters

        Returns
        -------
        str
            Cache key for the calculator

        """
        key = self._generate_key(backend, model_name, device, **kwargs)

        # Check if we need to evict old entries
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_oldest()

        # Cache the calculator
        self._cache[key] = calculator
        self._access_order[key] = self._access_counter
        self._access_counter += 1

        return key

    def _evict_oldest(self) -> None:
        """Evict the least recently used calculator."""
        if not self._access_order:
            return

        # Find the oldest entry
        oldest_key = min(self._access_order.keys(), key=lambda k: self._access_order[k])

        # Remove from cache (weak reference will handle cleanup)
        if oldest_key in self._cache:
            del self._cache[oldest_key]
        del self._access_order[oldest_key]

    def clear(self) -> None:
        """Clear all cached calculators."""
        self._cache.clear()
        self._access_order.clear()
        self._access_counter = 0


class ModelCache:
    """Persistent model cache with version checking and integrity validation."""

    def __init__(self, cache_dir: str | None = None) -> None:
        """Initialize model cache.

        Parameters
        ----------
        cache_dir : str, optional
            Cache directory. Defaults to ~/.famex/cache/models

        """
        if cache_dir is None:
            cache_dir = str(Path.home() / ".famex" / "cache" / "models")

        self.cache_dir: Path = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache metadata file
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict[str, Any]:
        """Load cache metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
                    return {}
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _save_metadata(self) -> None:
        """Save cache metadata to file."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except OSError as e:
            logger.warning("Could not save cache metadata: %s", e)

    def _get_model_hash(self, model_name: str, model_url: str) -> str:
        """Generate a hash for model identification."""
        content = f"{model_name}:{model_url}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_cached_model(self, model_name: str, model_url: str) -> Path | None:
        """Get cached model if available and valid.

        Parameters
        ----------
        model_name : str
            Name of the model
        model_url : str
            URL where the model can be downloaded

        Returns
        -------
        Path or None
            Path to cached model file, or None if not cached/valid

        """
        model_hash = self._get_model_hash(model_name, model_url)

        if model_hash not in self.metadata:
            return None

        cache_entry = self.metadata[model_hash]
        cached_path: Path = self.cache_dir / cache_entry["filename"]

        # Check if file exists and is valid
        if not cached_path.exists():
            # Remove invalid entry
            del self.metadata[model_hash]
            self._save_metadata()
            return None

        # Verify file integrity if checksum is available
        if "checksum" in cache_entry:
            if not self._verify_checksum(cached_path, cache_entry["checksum"]):
                logger.warning("Cached model %s failed checksum verification", model_name)
                cached_path.unlink()
                del self.metadata[model_hash]
                self._save_metadata()
                return None

        logger.info("Using cached model: %s", cached_path)
        return cached_path

    def cache_model(self, model_name: str, model_url: str, model_data: bytes) -> Path:
        """Cache a downloaded model.

        Parameters
        ----------
        model_name : str
            Name of the model (UNTRUSTED - will be sanitized for security)
        model_url : str
            URL where the model was downloaded from
        model_data : bytes
            Model data to cache

        Returns
        -------
        Path
            Path to cached model file

        Raises
        ------
        ValueError
            If model_name contains unsafe characters or path traversal attempts

        """
        model_hash = self._get_model_hash(model_name, model_url)

        # SECURITY: Sanitize model_name to prevent path traversal
        # This removes ALL directory components and unsafe characters
        try:
            safe_name = sanitize_filename(model_name, allow_path_sep=False)
        except PathSecurityError as e:
            raise ValueError(f"Invalid model name: {e}") from e

        filename = f"{safe_name}_{model_hash}.jpt"

        # SECURITY: Validate final path is within cache directory
        try:
            cached_path = validate_safe_path(
                self.cache_dir / filename,
                base_dir=self.cache_dir,
                must_exist=False,
                allow_absolute=False,
            )
        except PathSecurityError as e:
            raise ValueError(f"Cannot create safe cache path: {e}") from e

        # Save model data (now guaranteed safe)
        with open(cached_path, "wb") as f:
            f.write(model_data)

        # Calculate checksum for integrity verification
        checksum = hashlib.sha256(model_data).hexdigest()

        # Update metadata
        self.metadata[model_hash] = {
            "model_name": model_name,  # Store original for reference
            "model_url": model_url,
            "filename": filename,
            "checksum": checksum,
            "size": len(model_data),
        }
        self._save_metadata()

        logger.info("Cached model: %s", cached_path)
        return cached_path

    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file checksum."""
        try:
            with open(file_path, "rb") as f:
                actual_checksum = hashlib.sha256(f.read()).hexdigest()
            return actual_checksum == expected_checksum
        except OSError:
            return False

    def clear_cache(self, model_name: str | None = None) -> None:
        """Clear cache, optionally for a specific model.

        Parameters
        ----------
        model_name : str, optional
            Specific model to clear. If None, clears entire cache.

        """
        if model_name is None:
            # Clear entire cache
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.metadata = {}
            self._save_metadata()
        else:
            # Clear specific model
            to_remove = []
            for model_hash, entry in self.metadata.items():
                if entry["model_name"] == model_name:
                    cached_path = self.cache_dir / entry["filename"]
                    if cached_path.exists():
                        cached_path.unlink()
                    to_remove.append(model_hash)

            for model_hash in to_remove:
                del self.metadata[model_hash]
            self._save_metadata()

    def get_cache_info(self) -> dict:
        """Get information about cached models."""
        total_size = 0
        model_count = 0

        for entry in self.metadata.values():
            cached_path = self.cache_dir / entry["filename"]
            if cached_path.exists():
                total_size += entry.get("size", 0)
                model_count += 1

        return {
            "cache_dir": str(self.cache_dir),
            "model_count": model_count,
            "total_size_mb": total_size / (1024 * 1024),
            "models": list(self.metadata.values()),
        }


class UnifiedCache:
    """Unified cache manager for both calculators and models."""

    def __init__(self) -> None:
        """Initialize the unified cache manager."""
        self.calculator_cache = CalculatorCache()
        self.model_cache = ModelCache()

    def clear_all(self) -> None:
        """Clear both calculator and model caches."""
        self.calculator_cache.clear()
        self.model_cache.clear_cache()

    def get_cache_info(self) -> dict:
        """Get information about all caches."""
        return {
            "calculator_cache": {
                "size": len(self.calculator_cache._cache),
                "max_size": self.calculator_cache.max_size,
            },
            "model_cache": self.model_cache.get_cache_info(),
        }


# Global cache instances
_calculator_cache = None
_model_cache = None
_unified_cache = None


def _get_calculator_cache() -> CalculatorCache:
    """Get the global calculator cache instance."""
    global _calculator_cache
    if _calculator_cache is None:
        _calculator_cache = CalculatorCache()
    return _calculator_cache


def _get_model_cache() -> ModelCache:
    """Get the global model cache instance."""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
    return _model_cache


def _get_unified_cache() -> UnifiedCache:
    """Get the global unified cache instance."""
    global _unified_cache
    if _unified_cache is None:
        _unified_cache = UnifiedCache()
    return _unified_cache


# Calculator cache functions
def get_cached_calculator(
    backend: str,
    model_name: str | None,
    device: str | None,
    **kwargs: Any,
) -> Any | None:
    """Get a cached calculator if available.

    Parameters
    ----------
    backend : str
        Calculator backend
    model_name : str, optional
        Model name
    device : str, optional
        Device specification
    **kwargs
        Additional calculator parameters

    Returns
    -------
    Calculator or None
        Cached calculator if available, None otherwise

    """
    cache = _get_calculator_cache()
    return cache.get(backend, model_name, device, **kwargs)


def cache_calculator(
    calculator: Any,
    backend: str,
    model_name: str | None,
    device: str | None,
    **kwargs: Any,
) -> str:
    """Cache a calculator.

    Parameters
    ----------
    calculator : Any
        Calculator instance to cache
    backend : str
        Calculator backend
    model_name : str, optional
        Model name
    device : str, optional
        Device specification
    **kwargs
        Additional calculator parameters

    Returns
    -------
    str
        Cache key for the calculator

    """
    cache = _get_calculator_cache()
    return cache.put(calculator, backend, model_name, device, **kwargs)


# Model cache functions
def get_model_cache() -> ModelCache:
    """Get the global model cache instance."""
    return _get_model_cache()


def download_and_cache_model(model_name: str, model_url: str) -> Path:
    """Download and cache a model, using cache if available.

    Parameters
    ----------
    model_name : str
        Name of the model
    model_url : str
        URL to download the model from

    Returns
    -------
    Path
        Path to the model file (cached or newly downloaded)

    """
    cache = _get_model_cache()

    # Check if model is already cached
    cached_path = cache.get_cached_model(model_name, model_url)
    if cached_path is not None:
        return cached_path

    # Download model
    logger.info("Downloading model %s from %s", model_name, model_url)
    try:
        response = requests.get(model_url, timeout=30)
        response.raise_for_status()
        model_data = response.content
    except requests.RequestException as e:
        logger.error("Failed to download model %s from %s: %s", model_name, model_url, e)
        msg = f"Failed to download model {model_name}: {e}"
        raise RuntimeError(msg)

    # Cache the downloaded model
    return cache.cache_model(model_name, model_url, model_data)


# Unified cache functions
def get_unified_cache() -> UnifiedCache:
    """Get the global unified cache instance."""
    return _get_unified_cache()


def clear_all_caches() -> None:
    """Clear all caches (calculator and model)."""
    unified = _get_unified_cache()
    unified.clear_all()
