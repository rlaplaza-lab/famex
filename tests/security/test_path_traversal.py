"""Security tests for path traversal vulnerabilities."""

import tempfile
from pathlib import Path

import pytest

from qme.backends.cache import ModelCache
from qme.utils.path_security import (
    PathSecurityError,
    is_safe_relative_path,
    sanitize_filename,
    validate_safe_path,
)


class TestPathSanitization:
    """Test path sanitization functions."""

    def test_sanitize_filename_basic(self):
        assert sanitize_filename("model.jpt") == "model.jpt"
        assert sanitize_filename("model-v2_final.jpt") == "model-v2_final.jpt"

    def test_sanitize_filename_removes_paths(self):
        assert sanitize_filename("../model.jpt") == "model.jpt"
        assert sanitize_filename("../../etc/passwd") == "passwd"
        assert sanitize_filename("/etc/passwd") == "passwd"
        assert sanitize_filename("dir/subdir/model.jpt") == "model.jpt"

    def test_sanitize_filename_removes_dangerous_chars(self):
        assert sanitize_filename("model;rm -rf.txt") == "model_rm_-rf.txt"
        assert sanitize_filename("model`whoami`.txt") == "model_whoami_.txt"
        assert sanitize_filename("model$PATH.txt") == "model_PATH.txt"

    def test_sanitize_filename_null_byte(self):
        # Null bytes should be removed
        result = sanitize_filename("model\x00.jpt")
        assert "\x00" not in result

    def test_sanitize_filename_unicode_attacks(self):
        # Test various Unicode normalization attacks
        assert sanitize_filename("model\u202e.jpt") == "model_.jpt"  # Right-to-left override
        assert sanitize_filename("model\ufeff.jpt") == "model_.jpt"  # Zero-width no-break space

    def test_sanitize_filename_empty_result_raises(self):
        with pytest.raises(PathSecurityError):
            sanitize_filename(".....")
        with pytest.raises(PathSecurityError):
            sanitize_filename("///")

    def test_validate_safe_path_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            safe_path = base / "subdir" / "file.txt"
            validated = validate_safe_path(safe_path, base_dir=base)
            assert validated.is_absolute()
            assert validated.parent.parent == base

    def test_validate_safe_path_prevents_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            evil_path = base / ".." / ".." / "etc" / "passwd"
            with pytest.raises(PathSecurityError, match="escape base directory"):
                validate_safe_path(evil_path, base_dir=base)

    def test_validate_safe_path_rejects_absolute_by_default(self):
        with pytest.raises(PathSecurityError, match="Absolute paths not allowed"):
            validate_safe_path("/etc/passwd")

    def test_validate_safe_path_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            safe_dir = base / "safe"
            safe_dir.mkdir()

            # Create symlink pointing outside base
            link = safe_dir / "evil_link"
            link.symlink_to("/etc/passwd")

            # Should detect symlink escapes base directory
            with pytest.raises(PathSecurityError, match="escape base directory"):
                validate_safe_path(link, base_dir=base, must_exist=False)

    def test_is_safe_relative_path(self):
        assert is_safe_relative_path("model.jpt") is True
        assert is_safe_relative_path("subdir/model.jpt") is True
        assert is_safe_relative_path("../model.jpt") is False
        assert is_safe_relative_path("/etc/passwd") is False
        assert is_safe_relative_path("model$PATH.jpt") is False


class TestModelCachePathTraversal:
    """Test ModelCache against path traversal attacks."""

    def test_cache_model_with_traversal_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            # Try to write outside cache dir
            evil_name = "../../../etc/evil_model"
            model_data = b"malicious data"

            # Should sanitize and write safely
            cached_path = cache.cache_model(evil_name, "http://evil.com/model", model_data)

            # Verify file is within cache dir
            assert str(cached_path).startswith(str(tmpdir))

            # Verify parent directory is cache dir
            assert cached_path.parent == Path(tmpdir)

    def test_cache_model_null_byte_injection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            evil_name = "model\x00../../etc/passwd"
            model_data = b"test"

            # Should either sanitize or reject
            try:
                cached_path = cache.cache_model(evil_name, "http://test.com", model_data)
                assert "\x00" not in str(cached_path)
                assert str(cached_path).startswith(str(tmpdir))
            except ValueError:
                pass  # Also acceptable to reject

    def test_cache_model_windows_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            # Windows path traversal attempts
            evil_names = [
                "..\\..\\windows\\system32\\evil",
                "C:\\windows\\evil",
                "\\\\?\\C:\\evil",
            ]

            for evil_name in evil_names:
                model_data = b"test"
                cached_path = cache.cache_model(evil_name, "http://test.com", model_data)
                assert str(cached_path).startswith(str(tmpdir))

    def test_cache_model_suspicious_patterns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            # Test various suspicious patterns
            evil_names = [
                "~/malicious",
                "$HOME/malicious",
                "model~backup",
                "model$backup",
            ]

            for evil_name in evil_names:
                model_data = b"test"
                # Should all be sanitized
                cached_path = cache.cache_model(evil_name, "http://test.com", model_data)
                # Verify sanitization worked
                assert "~" not in str(cached_path)
                assert "$" not in str(cached_path)
                assert str(cached_path).startswith(str(tmpdir))
