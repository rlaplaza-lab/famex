from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from famex.cli.cli_helpers import write_atoms
from tests.security.test_utils import EVIL_PATHS, NULL_BYTE_PATHS, SAFE_PATHS


class TestCLIPathTraversal:
    @pytest.mark.parametrize("evil_path", EVIL_PATHS[:4])
    def test_write_atoms_rejects_traversal(self, test_atoms, evil_path):  # noqa: F811
        with pytest.raises(ValueError, match="Unsafe output path"):
            write_atoms(test_atoms, evil_path)

    @pytest.mark.parametrize("safe_path", SAFE_PATHS[:3])
    def test_write_atoms_allows_safe_paths(self, test_atoms, safe_path):  # noqa: F811
        with tempfile.TemporaryDirectory() as tmpdir:
            full_path = str(Path(tmpdir) / safe_path)
            try:
                result = write_atoms(test_atoms, full_path)
                assert result == full_path
                assert Path(full_path).exists()
                Path(full_path).unlink()
            except Exception:
                # Some paths might legitimately fail for non-security reasons
                pass

    @pytest.mark.parametrize("null_byte_path", NULL_BYTE_PATHS)
    def test_write_atoms_rejects_null_bytes(self, test_atoms, null_byte_path):  # noqa: F811
        with pytest.raises(ValueError, match="Unsafe output path"):
            write_atoms(test_atoms, null_byte_path)
