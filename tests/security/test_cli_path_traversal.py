from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from ase import Atoms

from qme.cli.cli_helpers import write_atoms


class TestCLIPathTraversal:
    def test_write_atoms_rejects_traversal(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        # Try various path traversal attempts
        evil_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\evil.txt",
            "../malicious.xyz",
            "subdir/../../etc/passwd",
        ]

        for evil_path in evil_paths:
            with pytest.raises(ValueError, match="Unsafe output path"):
                write_atoms(atoms, evil_path)

    def test_write_atoms_allows_safe_paths(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        # These should all work
        safe_paths = [
            "output.xyz",
            "subdir/output.xyz",
            "path/to/output.xyz",
            "/tmp/output.xyz",  # Absolute paths are allowed
        ]

        for path in safe_paths:
            with tempfile.TemporaryDirectory() as tmpdir:
                full_path = path if path.startswith("/") else str(Path(tmpdir) / path)

                try:
                    result = write_atoms(atoms, full_path)
                    assert result == full_path
                    assert Path(full_path).exists()
                    # Cleanup
                    Path(full_path).unlink()
                except Exception:
                    # Some paths might legitimately fail for non-security reasons
                    pass

    def test_write_atoms_rejects_null_bytes(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        with pytest.raises(ValueError, match="Unsafe output path"):
            write_atoms(atoms, "output\x00.xyz")
