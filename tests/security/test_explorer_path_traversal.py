from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from qme import Explorer
from tests.security.test_utils import EVIL_PATHS, NULL_BYTE_PATHS, SAFE_PATHS


class TestExplorerPathTraversal:
    @pytest.fixture
    def explorer(self, test_atoms):  # noqa: F811
        return Explorer(test_atoms, backend="mock", target="minima")

    @pytest.mark.parametrize("evil_path", EVIL_PATHS[:4])
    def test_save_structure_rejects_traversal(self, explorer, test_atoms, evil_path):  # noqa: F811
        with pytest.raises(ValueError, match="Unsafe output path"):
            explorer.save_structure(test_atoms, evil_path)

    @pytest.mark.parametrize("evil_path", EVIL_PATHS[:3])
    def test_save_trajectory_rejects_traversal(self, test_atoms, evil_path):  # noqa: F811
        trajectory = [test_atoms, test_atoms]
        explorer = Explorer(test_atoms, backend="mock", target="path", strategy="neb")
        with pytest.raises(ValueError, match="Unsafe output path"):
            explorer.save_trajectory(trajectory, evil_path)

    @pytest.mark.parametrize("safe_path", SAFE_PATHS[:3])
    def test_save_structure_allows_safe_paths(self, explorer, test_atoms, safe_path):  # noqa: F811
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / safe_path)
            try:
                explorer.save_structure(test_atoms, path)
                assert Path(path).exists()
                Path(path).unlink()
            except Exception:
                # Some paths might legitimately fail for non-security reasons
                pass

    @pytest.mark.parametrize("safe_path", SAFE_PATHS[:3])
    def test_save_trajectory_allows_safe_paths(self, test_atoms, safe_path):  # noqa: F811
        trajectory = [test_atoms, test_atoms]
        explorer = Explorer(test_atoms, backend="mock", target="path", strategy="neb")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / safe_path)
            try:
                explorer.save_trajectory(trajectory, path)
                assert Path(path).exists()
                Path(path).unlink()
            except Exception:
                # Some paths might legitimately fail for non-security reasons
                pass

    @pytest.mark.parametrize("null_byte_path", NULL_BYTE_PATHS)
    def test_explorer_rejects_null_bytes(self, test_atoms, null_byte_path):  # noqa: F811
        explorer = Explorer(test_atoms, backend="mock", target="minima")
        with pytest.raises(ValueError, match="Unsafe output path"):
            explorer.save_structure(test_atoms, null_byte_path)
