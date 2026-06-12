"""Tests for path security utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from famex.utils.path_security import (
    PathSecurityError,
    is_safe_relative_path,
    sanitize_filename,
    validate_safe_path,
)


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_valid_filename(self):
        """Test that valid filenames pass through."""
        assert sanitize_filename("test.txt") == "test.txt"
        assert sanitize_filename("my_file-123.xyz") == "my_file-123.xyz"
        assert sanitize_filename("simple_name") == "simple_name"

    def test_removes_path_components(self):
        """Test that path components are removed."""
        assert sanitize_filename("../etc/passwd") == "passwd"
        assert sanitize_filename("/absolute/path/file.txt") == "file.txt"
        assert sanitize_filename("dir/subdir/file.xyz") == "file.xyz"
        # On Linux, Windows paths are treated as regular strings
        # os.path.basename will return the whole string if no path separator
        # But the sanitization will still clean it up
        result = sanitize_filename("C:\\Windows\\System32\\file.exe")
        assert result.endswith("file.exe") or "file.exe" in result

    def test_removes_path_traversal(self):
        """Test that path traversal sequences are handled."""
        assert sanitize_filename("../../../etc/passwd") == "passwd"
        assert sanitize_filename("file..txt") == "file_txt"
        assert sanitize_filename("..file..txt..") == "file_txt"

    def test_removes_special_characters(self):
        """Test that special characters are replaced."""
        assert sanitize_filename("file@name.txt") == "file_name.txt"
        assert sanitize_filename("file#name$123") == "file_name_123"
        assert sanitize_filename("file with spaces.txt") == "file_with_spaces.txt"
        assert sanitize_filename("file!@#$%^&*().txt") == "file__________.txt"

    def test_removes_leading_dots(self):
        """Test that leading dots are removed (hidden files)."""
        assert sanitize_filename(".hidden") == "hidden"
        assert sanitize_filename("..hidden") == "hidden"
        assert sanitize_filename("...hidden") == "hidden"
        assert sanitize_filename(".file.txt") == "file.txt"

    def test_removes_leading_trailing_spaces(self):
        """Test that leading/trailing spaces are removed."""
        assert sanitize_filename("  file.txt  ") == "file.txt"
        assert sanitize_filename(" .hidden ") == "hidden"

    def test_empty_after_sanitization(self):
        """Test that empty filenames raise PathSecurityError."""
        with pytest.raises(PathSecurityError, match="no safe characters"):
            sanitize_filename("")
        with pytest.raises(PathSecurityError, match="no safe characters"):
            sanitize_filename("   ")
        # "!@#$%" becomes "_____" after sanitization, so it's not empty
        # Test with something that actually becomes empty after all sanitization steps
        # A string with only dots and spaces that get stripped
        with pytest.raises(PathSecurityError, match="no safe characters"):
            sanitize_filename("   .   ")

    def test_allow_path_sep(self):
        """Test allow_path_sep parameter."""
        # When allow_path_sep=True, path separators should be preserved in basename
        # But since we use os.path.basename first, this mainly affects the character replacement
        result = sanitize_filename("file-name.txt", allow_path_sep=True)
        assert result == "file-name.txt"


class TestValidateSafePath:
    """Tests for validate_safe_path function."""

    def test_valid_relative_path(self):
        """Test that valid relative paths are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test")

            # Should work with base_dir
            result = validate_safe_path("test.txt", base_dir=tmpdir)
            assert result == test_file.resolve()
            assert result.exists()

    def test_rejects_null_bytes(self):
        """Test that null bytes are rejected."""
        with pytest.raises(PathSecurityError, match="null bytes"):
            validate_safe_path("file\x00name.txt")

    def test_rejects_absolute_paths_by_default(self):
        """Test that absolute paths are rejected by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            abs_path = Path(tmpdir) / "test.txt"
            abs_path.write_text("test")

            with pytest.raises(PathSecurityError, match="Absolute paths not allowed"):
                validate_safe_path(str(abs_path), allow_absolute=False)

    def test_allows_absolute_paths_when_requested(self):
        """Test that absolute paths are allowed when requested."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test")

            result = validate_safe_path(str(test_file), allow_absolute=True)
            assert result == test_file.resolve()

    def test_path_traversal_within_base_dir(self):
        """Test that path traversal attempts are blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "base"
            base_dir.mkdir()
            (base_dir / "allowed.txt").write_text("test")

            # Valid path within base_dir
            result = validate_safe_path("allowed.txt", base_dir=base_dir)
            assert result.exists()

            # Path traversal should be blocked
            with pytest.raises(PathSecurityError, match="attempts to escape"):
                validate_safe_path("../allowed.txt", base_dir=base_dir)

            # Absolute path outside base_dir should be blocked
            outside_file = Path(tmpdir) / "outside.txt"
            outside_file.write_text("test")
            with pytest.raises(PathSecurityError, match="attempts to escape"):
                validate_safe_path(str(outside_file), base_dir=base_dir)

    def test_must_exist_validation(self):
        """Test must_exist parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Non-existent file should raise error when must_exist=True
            with pytest.raises(PathSecurityError, match="does not exist"):
                validate_safe_path("nonexistent.txt", base_dir=tmpdir, must_exist=True)

            # Should work when must_exist=False
            result = validate_safe_path(
                "nonexistent.txt",
                base_dir=tmpdir,
                must_exist=False,
            )
            assert result == (Path(tmpdir) / "nonexistent.txt").resolve()

    def test_path_resolution(self):
        """Test that paths are properly resolved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "base"
            base_dir.mkdir()
            subdir = base_dir / "subdir"
            subdir.mkdir()
            (subdir / "file.txt").write_text("test")

            # Using relative path should resolve correctly
            result = validate_safe_path("subdir/file.txt", base_dir=base_dir)
            assert result == (subdir / "file.txt").resolve()
            assert result.exists()

    def test_invalid_path_type(self):
        """Test that invalid path types raise PathSecurityError."""
        with pytest.raises(PathSecurityError, match="Invalid path"):
            validate_safe_path(None)  # type: ignore[arg-type]

    def test_path_with_base_dir_absolute(self):
        """Test validation with base_dir when path is absolute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "base"
            base_dir.mkdir()

            # Absolute path within base_dir should work
            test_file = base_dir / "test.txt"
            test_file.write_text("test")
            result = validate_safe_path(str(test_file), base_dir=base_dir)
            assert result == test_file.resolve()

            # Absolute path outside base_dir should fail
            outside_file = Path(tmpdir) / "outside.txt"
            outside_file.write_text("test")
            with pytest.raises(PathSecurityError, match="attempts to escape"):
                validate_safe_path(str(outside_file), base_dir=base_dir)


class TestIsSafeRelativePath:
    """Tests for is_safe_relative_path function."""

    def test_valid_relative_paths(self):
        """Test that valid relative paths return True."""
        assert is_safe_relative_path("file.txt") is True
        assert is_safe_relative_path("dir/file.txt") is True
        assert is_safe_relative_path("dir/subdir/file.xyz") is True
        assert is_safe_relative_path("file-name_123.txt") is True

    def test_rejects_path_traversal(self):
        """Test that path traversal patterns are rejected."""
        assert is_safe_relative_path("../file.txt") is False
        assert is_safe_relative_path("../../etc/passwd") is False
        assert is_safe_relative_path("dir/../file.txt") is False

    def test_rejects_absolute_paths(self):
        """Test that absolute paths are rejected."""
        assert is_safe_relative_path("/absolute/path") is False
        assert is_safe_relative_path("C:\\Windows\\path") is False
        assert is_safe_relative_path("~/.bashrc") is False

    def test_rejects_suspicious_patterns(self):
        """Test that suspicious patterns are rejected."""
        assert is_safe_relative_path("file$name.txt") is False
        assert is_safe_relative_path("file\x00name.txt") is False
        assert is_safe_relative_path("file%2ename.txt") is False
        assert is_safe_relative_path("file\\xname.txt") is False

    def test_rejects_invalid_types(self):
        """Test that non-string inputs return False."""
        assert is_safe_relative_path(None) is False  # type: ignore[arg-type]
        assert is_safe_relative_path(123) is False  # type: ignore[arg-type]
        assert is_safe_relative_path([]) is False  # type: ignore[arg-type]

    def test_rejects_empty_string(self):
        """Test that empty string returns False."""
        assert is_safe_relative_path("") is False

    def test_allows_safe_characters(self):
        """Test that safe characters are allowed."""
        assert is_safe_relative_path("file.txt") is True
        assert is_safe_relative_path("my_file-123.xyz") is True
        assert is_safe_relative_path("dir/subdir/file.txt") is True


class TestPathSecurityError:
    """Tests for PathSecurityError exception."""

    def test_exception_inheritance(self):
        """Test that PathSecurityError inherits from ValueError."""
        error = PathSecurityError("Test error")
        assert isinstance(error, ValueError)
        assert isinstance(error, Exception)

    def test_exception_message(self):
        """Test that exception message is correct."""
        error = PathSecurityError("Unsafe path detected")
        assert str(error) == "Unsafe path detected"
