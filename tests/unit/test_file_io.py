"""Tests for qme.core.file_io module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from qme.core.file_io import (
    _create_clean_atoms,
    validate_output_path,
    write_atoms_safely,
    write_trajectory_safely,
)


class TestValidateOutputPath:
    """Test validate_output_path function."""

    def test_valid_path(self):
        """Test that valid paths pass validation."""
        path = validate_output_path("test.xyz")
        assert isinstance(path, Path)
        assert str(path) == "test.xyz"

    def test_path_with_traversal(self):
        """Test that paths with .. are rejected."""
        with pytest.raises(ValueError, match="Unsafe output path"):
            validate_output_path("../test.xyz")

    def test_path_with_null_byte(self):
        """Test that paths with null bytes are rejected."""
        with pytest.raises(ValueError, match="Unsafe output path"):
            validate_output_path("test\x00.xyz")


class TestCreateCleanAtoms:
    """Test _create_clean_atoms function."""

    def test_clean_atoms_basic(self, water_molecule):
        """Test creating clean atoms without info."""
        clean = _create_clean_atoms(water_molecule)
        assert len(clean) == len(water_molecule)
        assert clean.info == {}

    def test_clean_atoms_with_info(self, water_molecule):
        """Test creating clean atoms with charge and spin in info."""
        atoms = water_molecule.copy()
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        atoms.info["other"] = "should_not_be_copied"

        clean = _create_clean_atoms(atoms)
        assert clean.info["charge"] == 0
        assert clean.info["spin"] == 1
        assert "other" not in clean.info

    def test_clean_atoms_with_empty_info(self, water_molecule):
        """Test creating clean atoms with empty info dict."""
        atoms = water_molecule.copy()
        atoms.info = {}
        clean = _create_clean_atoms(atoms)
        assert clean.info == {}

    def test_clean_atoms_with_partial_info(self, water_molecule):
        """Test creating clean atoms with only charge in info."""
        atoms = water_molecule.copy()
        atoms.info["charge"] = 1
        clean = _create_clean_atoms(atoms)
        assert clean.info["charge"] == 1
        assert "spin" not in clean.info


class TestWriteAtomsSafely:
    """Test write_atoms_safely function."""

    def test_write_xyz_success(self, water_molecule, tmp_path):
        """Test successful XYZ write."""
        output_file = tmp_path / "test.xyz"
        write_atoms_safely(water_molecule, output_file)
        assert output_file.exists()

    @pytest.mark.parametrize(
        "exception_class,exception_msg",
        [
            (OSError, "Permission denied"),
            (ValueError, "Invalid data"),
            (TypeError, "Type error"),
        ],
    )
    def test_write_xyz_errors(self, water_molecule, tmp_path, exception_class, exception_msg):
        """Test XYZ write with various error handling."""
        output_file = tmp_path / "test.xyz"
        with (
            patch(
                "qme.core.file_io.write_xyz_with_metadata",
                side_effect=exception_class(exception_msg),
            ),
            pytest.raises(RuntimeError, match="Failed to save XYZ structure"),
        ):
            write_atoms_safely(water_molecule, output_file)

    def test_write_ase_format_success(self, water_molecule, tmp_path):
        """Test successful write with ASE format."""
        output_file = tmp_path / "test.cif"
        write_atoms_safely(water_molecule, output_file, format="cif")
        assert output_file.exists()

    def test_write_ase_oserror_with_clean(self, water_molecule, tmp_path):
        """Test ASE write with OSError that succeeds with cleaned atoms."""
        output_file = tmp_path / "test.cif"
        call_count = 0

        def mock_write(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("First attempt failed")
            # Second attempt succeeds

        with patch("qme.core.file_io.write", side_effect=mock_write):
            write_atoms_safely(water_molecule, output_file, format="cif")
        assert call_count == 2

    def test_write_ase_oserror_both_fail(self, water_molecule, tmp_path):
        """Test ASE write with OSError where both attempts fail."""
        output_file = tmp_path / "test.cif"
        with (
            patch("qme.core.file_io.write", side_effect=OSError("Permission denied")),
            pytest.raises(RuntimeError, match="Clean attempt also failed"),
        ):
            write_atoms_safely(water_molecule, output_file, format="cif")

    @pytest.mark.parametrize(
        "exception_class,exception_msg",
        [
            (ValueError, "Clean attempt also failed"),
            (TypeError, "Clean attempt also failed"),
            (KeyError, "Clean attempt also failed"),
        ],
    )
    def test_write_ase_errors_with_clean(
        self, water_molecule, tmp_path, exception_class, exception_msg
    ):
        """Test ASE write with errors that fail even with cleaned atoms."""
        output_file = tmp_path / "test.cif"
        call_count = 0

        def mock_write(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("First attempt failed")
            raise exception_class(exception_msg)

        with (
            patch("qme.core.file_io.write", side_effect=mock_write),
            pytest.raises(RuntimeError, match="Clean attempt also failed"),
        ):
            write_atoms_safely(water_molecule, output_file, format="cif")

    @pytest.mark.parametrize(
        "exception_class,exception_msg",
        [
            (ValueError, "Unsupported format"),
            (TypeError, "Type error"),
            (KeyError, "Missing key"),
        ],
    )
    def test_write_ase_errors_direct(
        self, water_molecule, tmp_path, exception_class, exception_msg
    ):
        """Test ASE write with errors on first attempt."""
        output_file = tmp_path / "test.cif"
        with (
            patch("qme.core.file_io.write", side_effect=exception_class(exception_msg)),
            pytest.raises(RuntimeError, match="Failed to save structure"),
        ):
            write_atoms_safely(water_molecule, output_file, format="cif")


class TestWriteTrajectorySafely:
    """Test write_trajectory_safely function."""

    def test_write_trajectory_xyz_success(self, water_molecule, tmp_path):
        """Test successful XYZ trajectory write."""
        output_file = tmp_path / "trajectory.xyz"
        trajectory = [water_molecule.copy(), water_molecule.copy()]
        write_trajectory_safely(trajectory, output_file)
        assert output_file.exists()

    @pytest.mark.parametrize(
        "exception_class,exception_msg",
        [
            (OSError, "Permission denied"),
            (ValueError, "Invalid data"),
            (TypeError, "Type error"),
        ],
    )
    def test_write_trajectory_xyz_errors(
        self, water_molecule, tmp_path, exception_class, exception_msg
    ):
        """Test XYZ trajectory write with various error handling."""
        output_file = tmp_path / "trajectory.xyz"
        trajectory = [water_molecule.copy()]
        with (
            patch(
                "qme.core.file_io.write_xyz_with_metadata",
                side_effect=exception_class(exception_msg),
            ),
            pytest.raises(RuntimeError, match="Failed to save XYZ trajectory"),
        ):
            write_trajectory_safely(trajectory, output_file)

    def test_write_trajectory_ase_oserror_both_fail(self, water_molecule, tmp_path):
        """Test ASE trajectory write with OSError where both attempts fail."""
        output_file = tmp_path / "trajectory.cif"
        trajectory = [water_molecule.copy()]
        with (
            patch("qme.core.file_io.write", side_effect=OSError("Permission denied")),
            pytest.raises(RuntimeError, match="Clean attempt also failed"),
        ):
            write_trajectory_safely(trajectory, output_file, format="cif")

    @pytest.mark.parametrize(
        "exception_class,exception_msg",
        [
            (ValueError, "Clean attempt also failed"),
            (TypeError, "Clean attempt also failed"),
            (KeyError, "Clean attempt also failed"),
        ],
    )
    def test_write_trajectory_ase_errors_with_clean(
        self, water_molecule, tmp_path, exception_class, exception_msg
    ):
        """Test ASE trajectory write with errors that fail even with cleaned atoms."""
        output_file = tmp_path / "trajectory.cif"
        trajectory = [water_molecule.copy()]
        call_count = 0

        def mock_write(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("First attempt failed")
            raise exception_class(exception_msg)

        with (
            patch("qme.core.file_io.write", side_effect=mock_write),
            pytest.raises(RuntimeError, match="Clean attempt also failed"),
        ):
            write_trajectory_safely(trajectory, output_file, format="cif")

    @pytest.mark.parametrize(
        "exception_class,exception_msg",
        [
            (ValueError, "Unsupported format"),
            (TypeError, "Type error"),
            (KeyError, "Missing key"),
        ],
    )
    def test_write_trajectory_ase_errors_direct(
        self, water_molecule, tmp_path, exception_class, exception_msg
    ):
        """Test ASE trajectory write with errors on first attempt."""
        output_file = tmp_path / "trajectory.cif"
        trajectory = [water_molecule.copy()]
        with (
            patch("qme.core.file_io.write", side_effect=exception_class(exception_msg)),
            pytest.raises(RuntimeError, match="Failed to save trajectory"),
        ):
            write_trajectory_safely(trajectory, output_file, format="cif")
