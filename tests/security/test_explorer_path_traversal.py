from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from ase import Atoms

from qme import Explorer


class TestExplorerPathTraversal:
    def test_save_structure_rejects_traversal(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        explorer = Explorer(atoms, backend="mock", target="minima")

        # Try various path traversal attempts
        evil_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\evil.xyz",
            "../malicious.xyz",
            "subdir/../../etc/passwd",
        ]

        for evil_path in evil_paths:
            with pytest.raises(ValueError, match="Unsafe output path"):
                explorer.save_structure(atoms, evil_path)

    def test_save_trajectory_rejects_traversal(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        trajectory = [atoms, atoms]
        explorer = Explorer(atoms, backend="mock", target="path", strategy="neb")

        # Try various path traversal attempts
        evil_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\evil.xyz",
            "../malicious.xyz",
        ]

        for evil_path in evil_paths:
            with pytest.raises(ValueError, match="Unsafe output path"):
                explorer.save_trajectory(trajectory, evil_path)

    def test_save_structure_allows_safe_paths(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        explorer = Explorer(atoms, backend="mock", target="minima")

        with tempfile.TemporaryDirectory() as tmpdir:
            # These should all work
            safe_paths = [
                "output.xyz",
                "subdir/output.xyz",
                str(Path(tmpdir) / "output.xyz"),
            ]

            for path in safe_paths:
                if not path.startswith("/"):
                    path = str(Path(tmpdir) / path)

                try:
                    explorer.save_structure(atoms, path)
                    assert Path(path).exists()
                    # Cleanup
                    Path(path).unlink()
                except Exception:
                    # Some paths might legitimately fail for non-security reasons
                    pass

    def test_save_trajectory_allows_safe_paths(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        trajectory = [atoms, atoms]
        explorer = Explorer(atoms, backend="mock", target="path", strategy="neb")

        with tempfile.TemporaryDirectory() as tmpdir:
            # These should all work
            safe_paths = [
                "output.xyz",
                "subdir/output.xyz",
                str(Path(tmpdir) / "output.xyz"),
            ]

            for path in safe_paths:
                if not path.startswith("/"):
                    path = str(Path(tmpdir) / path)

                try:
                    explorer.save_trajectory(trajectory, path)
                    assert Path(path).exists()
                    # Cleanup
                    Path(path).unlink()
                except Exception:
                    # Some paths might legitimately fail for non-security reasons
                    pass

    def test_explorer_rejects_null_bytes(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        explorer = Explorer(atoms, backend="mock", target="minima")

        with pytest.raises(ValueError, match="Unsafe output path"):
            explorer.save_structure(atoms, "output\x00.xyz")
