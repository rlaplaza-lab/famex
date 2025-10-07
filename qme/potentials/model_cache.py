"""
Model caching utilities for QME backends.

This module provides persistent model caching to avoid re-downloading
and re-loading models on every run.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests


class ModelCache:
    """Persistent model cache with version checking and integrity validation."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model cache.

        Parameters:
        -----------
        cache_dir : str, optional
            Cache directory. Defaults to ~/.qme/cache/models
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".qme" / "cache" / "models"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache metadata file
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_metadata(self):
        """Save cache metadata to file."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save cache metadata: {e}")

    def _get_model_hash(self, model_name: str, model_url: str) -> str:
        """Generate a hash for model identification."""
        content = f"{model_name}:{model_url}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_cached_model(self, model_name: str, model_url: str) -> Optional[Path]:
        """
        Get cached model if available and valid.

        Parameters:
        -----------
        model_name : str
            Name of the model
        model_url : str
            URL where the model can be downloaded

        Returns:
        --------
        Path or None
            Path to cached model file, or None if not cached/valid
        """
        model_hash = self._get_model_hash(model_name, model_url)

        if model_hash not in self.metadata:
            return None

        cache_entry = self.metadata[model_hash]
        cached_path = self.cache_dir / cache_entry["filename"]

        # Check if file exists and is valid
        if not cached_path.exists():
            # Remove invalid entry
            del self.metadata[model_hash]
            self._save_metadata()
            return None

        # Verify file integrity if checksum is available
        if "checksum" in cache_entry:
            if not self._verify_checksum(cached_path, cache_entry["checksum"]):
                print(f"Warning: Cached model {model_name} failed checksum verification")
                cached_path.unlink()
                del self.metadata[model_hash]
                self._save_metadata()
                return None

        print(f"Using cached model: {cached_path}")
        return cached_path

    def cache_model(self, model_name: str, model_url: str, model_data: bytes) -> Path:
        """
        Cache a downloaded model.

        Parameters:
        -----------
        model_name : str
            Name of the model
        model_url : str
            URL where the model was downloaded from
        model_data : bytes
            Model data to cache

        Returns:
        --------
        Path
            Path to cached model file
        """
        model_hash = self._get_model_hash(model_name, model_url)

        # Generate filename from model name
        safe_name = model_name.replace("/", "_").replace(":", "_")
        filename = f"{safe_name}_{model_hash}.jpt"
        cached_path = self.cache_dir / filename

        # Save model data
        with open(cached_path, "wb") as f:
            f.write(model_data)

        # Calculate checksum for integrity verification
        checksum = hashlib.sha256(model_data).hexdigest()

        # Update metadata
        self.metadata[model_hash] = {
            "model_name": model_name,
            "model_url": model_url,
            "filename": filename,
            "checksum": checksum,
            "size": len(model_data),
        }
        self._save_metadata()

        print(f"Cached model: {cached_path}")
        return cached_path

    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file checksum."""
        try:
            with open(file_path, "rb") as f:
                actual_checksum = hashlib.sha256(f.read()).hexdigest()
            return actual_checksum == expected_checksum
        except IOError:
            return False

    def clear_cache(self, model_name: Optional[str] = None):
        """
        Clear cache, optionally for a specific model.

        Parameters:
        -----------
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

    def get_cache_info(self) -> Dict:
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


# Global model cache instance
_model_cache = None


def get_model_cache() -> ModelCache:
    """Get the global model cache instance."""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
    return _model_cache


def download_and_cache_model(model_name: str, model_url: str) -> Path:
    """
    Download and cache a model, using cache if available.

    Parameters:
    -----------
    model_name : str
        Name of the model
        model_url : str
        URL to download the model from

    Returns:
    --------
    Path
        Path to the model file (cached or newly downloaded)
    """
    cache = get_model_cache()

    # Check if model is already cached
    cached_path = cache.get_cached_model(model_name, model_url)
    if cached_path is not None:
        return cached_path

    # Download model
    print(f"Downloading model {model_name} from {model_url}")
    try:
        response = requests.get(model_url, timeout=30)
        response.raise_for_status()
        model_data = response.content
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download model {model_name}: {e}")

    # Cache the downloaded model
    return cache.cache_model(model_name, model_url, model_data)
