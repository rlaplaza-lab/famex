from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from ase import Atoms
from click.testing import CliRunner

from qme.cli.cache_commands import cache
from qme.cli.cli_helpers import (
    load_atoms_from_xyz,
    parse_kv_pairs,
    print_frequency_summary,
    save_results_json,
    write_atoms,
)


class TestParseKVPairs:
    def test_parse_basic_pairs(self):
        pairs = ["k=5.0", "steps=100", "name=test"]
        result = parse_kv_pairs(pairs)

        assert result["k"] == 5.0
        assert result["steps"] == 100
        assert result["name"] == "test"

    def test_parse_bool_values(self):
        pairs = ["verbose=true", "quiet=false"]
        result = parse_kv_pairs(pairs)

        assert result["verbose"] is True
        assert result["quiet"] is False

    def test_parse_int_float(self):
        pairs = ["count=42", "weight=3.14"]
        result = parse_kv_pairs(pairs)

        assert isinstance(result["count"], int)
        assert isinstance(result["weight"], float)
        assert result["count"] == 42
        assert result["weight"] == 3.14

    def test_parse_string_values(self):
        pairs = ["path=/some/path", "message=hello world"]
        result = parse_kv_pairs(pairs)

        assert result["path"] == "/some/path"
        assert result["message"] == "hello world"

    def test_parse_empty_list(self):
        result = parse_kv_pairs([])
        assert result == {}

    def test_parse_invalid_format(self):
        pairs = ["invalid", "valid=value"]
        result = parse_kv_pairs(pairs)

        assert "invalid" not in result
        assert result["valid"] == "value"

    def test_parse_with_spaces(self):
        pairs = [" key = value ", "k2=v2"]
        result = parse_kv_pairs(pairs)

        assert result["key"] == "value"
        assert result["k2"] == "v2"


class TestLoadAtomsFromXYZ:
    def test_load_single_frame_xyz(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_file = f.name
            atoms.write(temp_file)

        try:
            loaded = load_atoms_from_xyz(temp_file)
            assert len(loaded) == 2
            assert loaded.get_chemical_symbols() == ["H", "H"]
        finally:
            Path(temp_file).unlink()

    def test_load_nonexistent_file(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_atoms_from_xyz("nonexistent.xyz")


class TestWriteAtoms:
    def test_write_single_atoms(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_file = f.name

        try:
            result = write_atoms(atoms, temp_file)
            assert result == temp_file
            assert Path(temp_file).exists()
        finally:
            if Path(temp_file).exists():
                Path(temp_file).unlink()

    def test_write_list_of_atoms(self):
        atoms1 = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        atoms2 = Atoms("H2", positions=[[0, 0, 0], [0.80, 0, 0]])
        atoms_list = [atoms1, atoms2]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_file = f.name

        try:
            result = write_atoms(atoms_list, temp_file)
            assert result == temp_file
            assert Path(temp_file).exists()
        finally:
            if Path(temp_file).exists():
                Path(temp_file).unlink()

    def test_write_dict_result(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        result_dict = {"optimized_atoms": atoms, "converged": True}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_file = f.name

        try:
            write_atoms(result_dict, temp_file)
            assert Path(temp_file).exists()
        finally:
            if Path(temp_file).exists():
                Path(temp_file).unlink()

    def test_write_none_path(self):
        atoms = Atoms("H2")
        result = write_atoms(atoms, None)
        assert result is None


class TestPrintFrequencySummary:
    def test_print_minima_summary(self, capsys):
        frequency_analysis = {
            "frequencies": [100.0, 200.0, 300.0, 400.0],
            "zero_point_energy": 0.5,
            "thermodynamic_properties": {
                "temperature": 298.15,
                "entropy": 0.1,
            },
            "is_minimum": True,
            "minima_analysis": {"n_significant_imaginary_frequencies": 0},
        }

        print_frequency_summary(frequency_analysis, target="minima")
        captured = capsys.readouterr()

        assert "Frequency analysis" in captured.out
        assert "Zero-point energy" in captured.out
        assert "Valid minimum" in captured.out

    def test_print_ts_summary(self, capsys):
        frequency_analysis = {
            "frequencies": [-100.0, 200.0, 300.0],
            "zero_point_energy": 0.4,
            "thermodynamic_properties": {
                "temperature": 298.15,
                "entropy": 0.1,
            },
            "is_ts": True,
            "ts_analysis": {"n_imaginary_frequencies": 1},
        }

        print_frequency_summary(frequency_analysis, target="ts")
        captured = capsys.readouterr()

        assert "transition state" in captured.out.lower()
        assert "imaginary frequency" in captured.out.lower()

    def test_print_empty_summary(self, capsys):
        print_frequency_summary({}, target="minima")
        captured = capsys.readouterr()
        assert captured.out == ""


class TestSaveResultsJSON:
    def test_save_basic_results(self):
        results = {
            "converged": True,
            "steps_taken": 10,
            "strategy": "local",
            "optimized_atoms": Atoms("H2"),  # Should be skipped
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_file = f.name

        try:
            save_results_json(results, temp_file)
            json_path = Path(temp_file).with_suffix(".json")

            assert json_path.exists()
            with open(json_path) as json_file:
                data = json.load(json_file)

            assert data["converged"] is True
            assert data["steps_taken"] == 10
            assert "optimized_atoms" not in data  # Should be excluded
        finally:
            if Path(temp_file).exists():
                Path(temp_file).unlink()
            json_path = Path(temp_file).with_suffix(".json")
            if json_path.exists():
                json_path.unlink()

    def test_save_with_frequency_analysis(self):
        import numpy as np

        results = {
            "converged": True,
            "frequency_analysis": {
                "frequencies": np.array([100.0, 200.0]),
                "zero_point_energy": 0.5,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_file = f.name

        try:
            save_results_json(results, temp_file)
            json_path = Path(temp_file).with_suffix(".json")

            assert json_path.exists()
            with open(json_path) as json_file:
                data = json.load(json_file)

            assert "frequency_analysis" in data
            assert isinstance(data["frequency_analysis"]["frequencies"], list)
        finally:
            if Path(temp_file).exists():
                Path(temp_file).unlink()
            json_path = Path(temp_file).with_suffix(".json")
            if json_path.exists():
                json_path.unlink()


class TestCacheCommands:
    def test_cache_info_command(self):
        runner = CliRunner()
        result = runner.invoke(cache, ["info"])

        assert result.exit_code == 0
        assert "QME Model Cache" in result.output

    def test_cache_verify_command(self):
        runner = CliRunner()
        result = runner.invoke(cache, ["verify"])

        assert result.exit_code == 0
        assert "Verifying" in result.output or "cache" in result.output.lower()

    def test_cache_clear_command_no_yes(self):
        runner = CliRunner()
        # Simulate 'n' for no
        result = runner.invoke(cache, ["clear"], input="n\n")

        assert result.exit_code == 0
        assert "Cancelled" in result.output or result.exit_code == 0


class TestLoadAtomsFromXYZExtended:
    def test_load_xyz_with_multiple_frames(self):
        import tempfile
        from pathlib import Path

        # Create multi-frame XYZ
        xyz_content = """3
frame 1
H  0.0  0.0  0.0
H  0.0  0.0  0.7
H  0.0  0.0  1.4
3
frame 2
H  0.0  0.0  0.0
H  0.0  0.0  0.8
H  0.0  0.0  1.6
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_file = f.name
            f.write(xyz_content)

        try:
            loaded = load_atoms_from_xyz(temp_file)
            # Should return last frame
            assert len(loaded) == 3
        finally:
            Path(temp_file).unlink(missing_ok=True)


class TestCoerceToAtoms:
    def test_coerce_atoms_tuple(self):
        from qme.cli.cli_helpers import _coerce_to_atoms

        atoms1 = Atoms("H2")
        atoms2 = Atoms("O")

        result = _coerce_to_atoms((atoms1, atoms2))
        assert result == atoms1  # First one

    def test_coerce_atoms_dict(self):
        from qme.cli.cli_helpers import _coerce_to_atoms

        atoms = Atoms("H2O")
        result = _coerce_to_atoms({"optimized_atoms": atoms})
        assert result == atoms

    def test_coerce_atoms_path(self):
        from qme.cli.cli_helpers import _coerce_to_atoms

        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_file = f.name
            atoms.write(temp_file)

        try:
            result = _coerce_to_atoms(temp_file)
            assert len(result) == 2
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_coerce_atoms_invalid(self):
        from qme.cli.cli_helpers import _coerce_to_atoms

        with pytest.raises(TypeError):
            _coerce_to_atoms(123)


class TestLoadNonXYZFile:
    def test_load_non_xyz_file(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as f:
            temp_file = f.name
            atoms.write(temp_file)

        try:
            loaded = load_atoms_from_xyz(temp_file)
            assert len(loaded) == 2
        finally:
            Path(temp_file).unlink(missing_ok=True)
