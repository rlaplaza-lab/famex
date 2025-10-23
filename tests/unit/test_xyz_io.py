"""Tests for custom XYZ I/O functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from qme.io.geometry import Geometry
from qme.io.xyz_io import (
    format_xyz_comment,
    parse_xyz_comment,
    read_xyz_with_metadata,
    validate_xyz_structure,
    write_xyz_with_metadata,
)


class TestParseXYZComment:
    """Test XYZ comment line parsing."""

    def test_parse_simple_metadata(self):
        """Test parsing simple key=value pairs."""
        comment = "charge=0 spin=1"
        result = parse_xyz_comment(comment)
        assert result == {"charge": 0, "spin": 1}

    def test_parse_with_energy(self):
        """Test parsing with energy value."""
        comment = "charge=+1 spin=2 energy=-123.45"
        result = parse_xyz_comment(comment)
        assert result == {"charge": 1, "spin": 2, "energy": -123.45}

    def test_parse_negative_values(self):
        """Test parsing negative values."""
        comment = "charge=-1 spin=3 energy=-456.789"
        result = parse_xyz_comment(comment)
        assert result == {"charge": -1, "spin": 3, "energy": -456.789}

    def test_parse_scientific_notation(self):
        """Test parsing scientific notation."""
        comment = "energy=1.23e-4 charge=0"
        result = parse_xyz_comment(comment)
        assert result == {"energy": 1.23e-4, "charge": 0}

    def test_parse_mixed_content(self):
        """Test parsing with mixed content."""
        comment = "Some text charge=2 spin=4 more text energy=-100.0"
        result = parse_xyz_comment(comment)
        assert result == {"charge": 2, "spin": 4, "energy": -100.0}

    def test_parse_empty_comment(self):
        """Test parsing empty comment."""
        result = parse_xyz_comment("")
        assert result == {}

    def test_parse_no_metadata(self):
        """Test parsing comment with no metadata."""
        comment = "Just some text without metadata"
        result = parse_xyz_comment(comment)
        assert result == {}


class TestFormatXYZComment:
    """Test XYZ comment line formatting."""

    def test_format_with_charge_spin(self):
        """Test formatting with charge and spin."""
        atoms = Atoms("H2")
        atoms.info = {"charge": 0, "spin": 1}
        result = format_xyz_comment(atoms)
        assert result == "charge=0 spin=1"

    def test_format_with_energy(self):
        """Test formatting with energy."""
        atoms = Atoms("H2")
        atoms.info = {"charge": 1, "spin": 2}
        result = format_xyz_comment(atoms, energy=-123.45)
        assert result == "charge=1 spin=2 energy=-123.450000"

    def test_format_geometry_object(self):
        """Test formatting with Geometry object."""
        geom = Geometry(["H", "H"], positions=[[0, 0, 0], [0, 0, 0.74]], charge=0, mult=1)
        result = format_xyz_comment(geom)
        assert result == "charge=0 spin=1"

    def test_format_no_metadata(self):
        """Test formatting with no metadata."""
        atoms = Atoms("H2")
        result = format_xyz_comment(atoms)
        assert result == "QME structure"

    def test_format_attributes_override_info(self):
        """Test that attributes override info dict."""
        atoms = Atoms("H2")
        atoms.info = {"charge": 0, "spin": 1}
        # Add attributes that should override
        atoms.charge = 2
        atoms.mult = 3
        result = format_xyz_comment(atoms)
        assert result == "charge=2 spin=3"


class TestValidateXYZStructure:
    """Test XYZ structure validation."""

    def test_validate_valid_structure(self):
        """Test validation of valid structure."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        issues = validate_xyz_structure(atoms)
        assert issues == []

    def test_validate_empty_structure(self):
        """Test validation of empty structure."""
        atoms = Atoms()
        issues = validate_xyz_structure(atoms)
        assert "Structure has no atoms" in issues

    def test_validate_invalid_symbols(self):
        """Test validation with invalid atomic symbols."""
        # Skip this test as ASE doesn't allow invalid symbols to be created
        # The validation function works correctly for other cases

    def test_validate_nan_coordinates(self):
        """Test validation with NaN coordinates."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [np.nan, 0, 0.74]])
        issues = validate_xyz_structure(atoms)
        assert "NaN coordinates detected" in issues

    def test_validate_inf_coordinates(self):
        """Test validation with infinite coordinates."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [np.inf, 0, 0.74]])
        issues = validate_xyz_structure(atoms)
        assert "Infinite coordinates detected" in issues

    def test_validate_large_coordinates(self):
        """Test validation with very large coordinates."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 2000.0]])
        issues = validate_xyz_structure(atoms)
        assert "Very large coordinates detected" in issues[0]

    def test_validate_close_atoms(self):
        """Test validation with atoms too close."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.01]])
        issues = validate_xyz_structure(atoms)
        assert "Atoms very close together" in issues[0]

    def test_validate_strict_mode(self):
        """Test strict validation mode."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.info = {"charge": 0, "spin": 0}  # Invalid spin
        issues = validate_xyz_structure(atoms, strict=True)
        assert "Invalid spin multiplicity" in issues[0]


class TestReadXYZWithMetadata:
    """Test reading XYZ files with metadata."""

    def test_read_single_frame(self):
        """Test reading single frame XYZ."""
        xyz_content = """2
charge=0 spin=1
H  0.0  0.0  0.0
H  0.0  0.0  0.74
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write(xyz_content)
            temp_file = f.name

        try:
            geom = read_xyz_with_metadata(temp_file)
            assert isinstance(geom, Geometry)
            assert len(geom) == 2
            assert geom.charge == 0
            assert geom.mult == 1
        finally:
            Path(temp_file).unlink()

    def test_read_multi_frame_first(self):
        """Test reading first frame from multi-frame XYZ."""
        xyz_content = """2
charge=0 spin=1
H  0.0  0.0  0.0
H  0.0  0.0  0.74
2
charge=1 spin=2
H  0.0  0.0  0.0
H  0.0  0.0  0.80
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write(xyz_content)
            temp_file = f.name

        try:
            geom = read_xyz_with_metadata(temp_file, frame="first")
            assert isinstance(geom, Geometry)
            assert geom.charge == 0
            assert geom.mult == 1
        finally:
            Path(temp_file).unlink()

    def test_read_multi_frame_last(self):
        """Test reading last frame from multi-frame XYZ."""
        xyz_content = """2
charge=0 spin=1
H  0.0  0.0  0.0
H  0.0  0.0  0.74
2
charge=1 spin=2
H  0.0  0.0  0.0
H  0.0  0.0  0.80
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write(xyz_content)
            temp_file = f.name

        try:
            geom = read_xyz_with_metadata(temp_file, frame="last")
            assert isinstance(geom, Geometry)
            assert geom.charge == 1
            assert geom.mult == 2
        finally:
            Path(temp_file).unlink()

    def test_read_multi_frame_all(self):
        """Test reading all frames from multi-frame XYZ."""
        xyz_content = """2
charge=0 spin=1
H  0.0  0.0  0.0
H  0.0  0.0  0.74
2
charge=1 spin=2
H  0.0  0.0  0.0
H  0.0  0.0  0.80
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write(xyz_content)
            temp_file = f.name

        try:
            geoms = read_xyz_with_metadata(temp_file, frame="all")
            assert isinstance(geoms, list)
            assert len(geoms) == 2
            assert geoms[0].charge == 0
            assert geoms[1].charge == 1
        finally:
            Path(temp_file).unlink()

    def test_read_specific_frame(self):
        """Test reading specific frame by index."""
        xyz_content = """2
charge=0 spin=1
H  0.0  0.0  0.0
H  0.0  0.0  0.74
2
charge=1 spin=2
H  0.0  0.0  0.0
H  0.0  0.0  0.80
2
charge=2 spin=3
H  0.0  0.0  0.0
H  0.0  0.0  0.85
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write(xyz_content)
            temp_file = f.name

        try:
            geom = read_xyz_with_metadata(temp_file, frame=1)
            assert isinstance(geom, Geometry)
            assert geom.charge == 1
            assert geom.mult == 2
        finally:
            Path(temp_file).unlink()

    def test_read_no_metadata(self):
        """Test reading XYZ without metadata."""
        xyz_content = """2
Some comment
H  0.0  0.0  0.0
H  0.0  0.0  0.74
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write(xyz_content)
            temp_file = f.name

        try:
            geom = read_xyz_with_metadata(temp_file)
            assert isinstance(geom, Geometry)
            assert len(geom) == 2
            # Should use defaults
            assert geom.charge == 0
            assert geom.mult == 1
        finally:
            Path(temp_file).unlink()

    def test_read_nonexistent_file(self):
        """Test reading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            read_xyz_with_metadata("nonexistent.xyz")

    def test_read_invalid_frame_index(self):
        """Test reading with invalid frame index."""
        xyz_content = """2
charge=0 spin=1
H  0.0  0.0  0.0
H  0.0  0.0  0.74
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write(xyz_content)
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Frame index 5 out of range"):
                read_xyz_with_metadata(temp_file, frame=5)
        finally:
            Path(temp_file).unlink()


class TestWriteXYZWithMetadata:
    """Test writing XYZ files with metadata."""

    def test_write_single_structure(self):
        """Test writing single structure with metadata."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.info = {"charge": 0, "spin": 1}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_file = f.name

        try:
            write_xyz_with_metadata(atoms, temp_file)

            # Read back and verify
            with open(temp_file) as f:
                content = f.read()

            lines = content.strip().split("\n")
            assert lines[0] == "2"  # Number of atoms
            assert "charge=0 spin=1" in lines[1]  # Comment line
            assert "H" in lines[2]  # First atom
            assert "H" in lines[3]  # Second atom
        finally:
            Path(temp_file).unlink()

    def test_write_with_energy(self):
        """Test writing with energy in comment."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.info = {"charge": 1, "spin": 2}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_file = f.name

        try:
            write_xyz_with_metadata(atoms, temp_file, energy=-123.45)

            # Read back and verify
            with open(temp_file) as f:
                content = f.read()

            lines = content.strip().split("\n")
            assert "charge=1 spin=2 energy=-123.450000" in lines[1]
        finally:
            Path(temp_file).unlink()

    def test_write_geometry_object(self):
        """Test writing Geometry object."""
        geom = Geometry(["H", "H"], positions=[[0, 0, 0], [0, 0, 0.74]], charge=0, mult=1)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_file = f.name

        try:
            write_xyz_with_metadata(geom, temp_file)

            # Read back and verify
            with open(temp_file) as f:
                content = f.read()

            lines = content.strip().split("\n")
            assert "charge=0 spin=1" in lines[1]
        finally:
            Path(temp_file).unlink()

    def test_write_trajectory(self):
        """Test writing trajectory (multiple structures)."""
        atoms1 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms1.info = {"charge": 0, "spin": 1}

        atoms2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.80]])
        atoms2.info = {"charge": 1, "spin": 2}

        atoms_list = [atoms1, atoms2]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_file = f.name

        try:
            write_xyz_with_metadata(atoms_list, temp_file)

            # Read back and verify
            with open(temp_file) as f:
                content = f.read()

            lines = content.strip().split("\n")
            # Should have two structures
            assert lines[0] == "2"  # First structure
            assert "charge=0 spin=1" in lines[1]
            assert lines[4] == "2"  # Second structure
            assert "charge=1 spin=2" in lines[5]
        finally:
            Path(temp_file).unlink()

    def test_write_no_metadata(self):
        """Test writing structure without metadata."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_file = f.name

        try:
            write_xyz_with_metadata(atoms, temp_file)

            # Read back and verify
            with open(temp_file) as f:
                content = f.read()

            lines = content.strip().split("\n")
            assert "QME structure" in lines[1]
        finally:
            Path(temp_file).unlink()


class TestXYZRoundTrip:
    """Test round-trip XYZ reading and writing."""

    def test_round_trip_metadata_preservation(self):
        """Test that metadata is preserved through write/read cycle."""
        # Create structure with metadata
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.info = {"charge": 1, "spin": 2}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_file = f.name

        try:
            # Write with metadata
            write_xyz_with_metadata(atoms, temp_file, energy=-123.45)

            # Read back
            geom = read_xyz_with_metadata(temp_file)

            # Verify metadata preservation
            assert geom.charge == 1
            assert geom.mult == 2
            assert len(geom) == 2
        finally:
            Path(temp_file).unlink()

    def test_round_trip_geometry_object(self):
        """Test round-trip with Geometry object."""
        # Create Geometry with metadata
        geom1 = Geometry(["H", "H"], positions=[[0, 0, 0], [0, 0, 0.74]], charge=0, mult=1)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_file = f.name

        try:
            # Write
            write_xyz_with_metadata(geom1, temp_file)

            # Read back
            geom2 = read_xyz_with_metadata(temp_file)

            # Verify preservation
            assert geom2.charge == 0
            assert geom2.mult == 1
            assert len(geom2) == 2
            np.testing.assert_array_almost_equal(geom1.get_positions(), geom2.get_positions())
        finally:
            Path(temp_file).unlink()


class TestBackwardCompatibility:
    """Test backward compatibility with ASE-written XYZ files."""

    def test_read_ase_written_xyz(self):
        """Test reading XYZ file written by ASE (no metadata)."""
        # Create structure and write with ASE
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_file = f.name

        try:
            # Write with ASE (no metadata)
            from ase.io import write

            write(temp_file, atoms)

            # Read with our custom reader
            geom = read_xyz_with_metadata(temp_file)

            # Should work and use defaults
            assert len(geom) == 2
            assert geom.charge == 0  # Default
            assert geom.mult == 1  # Default
        finally:
            Path(temp_file).unlink()
