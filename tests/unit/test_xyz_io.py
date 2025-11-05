from __future__ import annotations

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
    @pytest.mark.parametrize(
        ("comment", "expected"),
        [
            ("charge=0 spin=1", {"charge": 0, "spin": 1}),
            ("charge=+1 spin=2 energy=-123.45", {"charge": 1, "spin": 2, "energy": -123.45}),
            ("charge=-1 energy=-456.789", {"charge": -1, "energy": -456.789}),
            ("energy=1.23e-4 charge=0", {"energy": 1.23e-4, "charge": 0}),
            ("Some text charge=2 spin=4 energy=-100.0", {"charge": 2, "spin": 4, "energy": -100.0}),
            ("", {}),
            ("Just some text without metadata", {}),
        ],
        ids=["basic", "with_energy", "partial", "reordered", "with_text", "empty", "no_metadata"],
    )
    def test_parse_variations(self, comment, expected):
        result = parse_xyz_comment(comment)
        assert result == expected


class TestFormatXYZComment:
    @pytest.mark.parametrize(
        ("obj_type", "info", "charge_attr", "mult_attr", "energy", "expected"),
        [
            ("atoms", {"charge": 0, "spin": 1}, None, None, None, "charge=0 spin=1"),
            (
                "atoms",
                {"charge": 1, "spin": 2},
                None,
                None,
                -123.45,
                "charge=1 spin=2 energy=-123.450000",
            ),
            ("geometry", None, 0, 1, None, "charge=0 spin=1"),
            ("atoms", {}, None, None, None, "QME structure"),
            (
                "atoms",
                {"charge": 0, "spin": 1},
                2,
                3,
                None,
                "charge=2 spin=3",
            ),  # Attributes override
        ],
        ids=[
            "atoms_basic",
            "atoms_with_energy",
            "geometry",
            "atoms_no_metadata",
            "atoms_attr_override",
        ],
    )
    def test_format_variations(self, obj_type, info, charge_attr, mult_attr, energy, expected):
        if obj_type == "atoms":
            obj = Atoms("H2")
            if info:
                obj.info = info
            if charge_attr is not None:
                obj.charge = charge_attr
            if mult_attr is not None:
                obj.mult = mult_attr
        else:
            obj = Geometry(
                ["H", "H"],
                positions=[[0, 0, 0], [0, 0, 0.74]],
                charge=charge_attr or 0,
                mult=mult_attr or 1,
            )

        kwargs = {"energy": energy} if energy is not None else {}
        result = format_xyz_comment(obj, **kwargs)
        assert result == expected


class TestValidateXYZStructure:
    def test_validate_valid_structure(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        issues = validate_xyz_structure(atoms)
        assert issues == []

    @pytest.mark.parametrize(
        ("atoms_factory", "expected_msg"),
        [
            (lambda: Atoms(), "Structure has no atoms"),
            (
                lambda: Atoms("H2", positions=[[0, 0, 0], [np.nan, 0, 0.74]]),
                "NaN coordinates detected",
            ),
            (
                lambda: Atoms("H2", positions=[[0, 0, 0], [np.inf, 0, 0.74]]),
                "Infinite coordinates detected",
            ),
            (
                lambda: Atoms("H2", positions=[[0, 0, 0], [0, 0, 2000.0]]),
                "Very large coordinates detected",
            ),
            (lambda: Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.01]]), "Atoms very close together"),
        ],
        ids=["no_atoms", "nan_coords", "inf_coords", "large_coords", "close_atoms"],
    )
    def test_validate_coordinate_errors(self, atoms_factory, expected_msg):
        atoms = atoms_factory()
        issues = validate_xyz_structure(atoms)
        assert any(expected_msg in issue for issue in issues)

    def test_validate_strict_mode(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.info = {"charge": 0, "spin": 0}  # Invalid spin
        issues = validate_xyz_structure(atoms, strict=True)
        assert "Invalid spin multiplicity" in issues[0]


class TestReadXYZWithMetadata:
    def test_read_single_frame(self):
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

    def test_read_multi_frame(self):
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
            # Test first frame
            geom = read_xyz_with_metadata(temp_file, frame="first")
            assert isinstance(geom, Geometry)
            assert geom.charge == 0

            # Test last frame
            geom = read_xyz_with_metadata(temp_file, frame="last")
            assert geom.charge == 2

            # Test all frames
            geoms = read_xyz_with_metadata(temp_file, frame="all")
            assert isinstance(geoms, list)
            assert len(geoms) == 3
            assert geoms[1].charge == 1

            # Test specific frame
            geom = read_xyz_with_metadata(temp_file, frame=1)
            assert geom.charge == 1
        finally:
            Path(temp_file).unlink()

    def test_read_no_metadata(self):
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
        with pytest.raises(FileNotFoundError):
            read_xyz_with_metadata("nonexistent.xyz")

    def test_read_invalid_frame_index(self):
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
    @pytest.mark.parametrize(
        ("obj_type", "info", "energy", "expected_comment"),
        [
            ("atoms", {"charge": 0, "spin": 1}, None, "charge=0 spin=1"),
            ("atoms", {"charge": 1, "spin": 2}, -123.45, "charge=1 spin=2 energy=-123.450000"),
            ("geometry", None, None, "charge=0 spin=1"),
        ],
        ids=["atoms_basic", "atoms_with_energy", "geometry"],
    )
    def test_write_variations(self, obj_type, info, energy, expected_comment):
        if obj_type == "atoms":
            obj = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
            if info:
                obj.info = info
        else:
            obj = Geometry(["H", "H"], positions=[[0, 0, 0], [0, 0, 0.74]], charge=0, mult=1)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_file = f.name

        try:
            kwargs = {"energy": energy} if energy is not None else {}
            write_xyz_with_metadata(obj, temp_file, **kwargs)

            with open(temp_file) as f:
                content = f.read()

            lines = content.strip().split("\n")
            assert lines[0] == "2"  # Number of atoms
            assert expected_comment in lines[1]
            if obj_type == "atoms":
                assert "H" in lines[2]
        finally:
            Path(temp_file).unlink()

    def test_write_trajectory(self):
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
    def test_round_trip_metadata_preservation(self):
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
    def test_read_ase_written_xyz(self):
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
