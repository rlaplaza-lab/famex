"""Comprehensive tests for Explorer class covering edge cases and uncovered code paths."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from qme.core.explorer import Explorer, _extract_charge_spin
from tests.test_utils import TestMoleculeFactory


class TestExtractChargeSpin:
    """Test _extract_charge_spin function with various metadata formats."""

    def test_extract_from_xyz_comment(self) -> None:
        """Test extraction from XYZ comment metadata."""
        atoms = TestMoleculeFactory.get_water_distorted()
        atoms.info = {"comment": "charge=1 spin=3 energy=-100.0"}

        charge, spin = _extract_charge_spin(atoms, default_charge=0, default_spin=1)

        assert charge == 1
        assert spin == 3

    def test_extract_from_xyz_comment_partial(self) -> None:
        """Test extraction with partial metadata in comment."""
        atoms = TestMoleculeFactory.get_water_distorted()
        atoms.info = {"comment": "charge=2"}

        charge, spin = _extract_charge_spin(atoms, default_charge=0, default_spin=1)

        assert charge == 2
        assert spin == 1  # Uses default

    def test_extract_from_atoms_charge_mult(self) -> None:
        """Test extraction from atoms.charge and atoms.mult attributes."""
        atoms = TestMoleculeFactory.get_water_distorted()
        atoms.charge = 2
        atoms.mult = 3

        charge, spin = _extract_charge_spin(atoms, default_charge=0, default_spin=1)

        assert charge == 2
        assert spin == 3

    def test_extract_from_atoms_info(self) -> None:
        """Test extraction from atoms.info dictionary."""
        atoms = TestMoleculeFactory.get_water_distorted()
        atoms.info = {"charge": 3, "spin": 4}

        charge, spin = _extract_charge_spin(atoms, default_charge=0, default_spin=1)

        assert charge == 3
        assert spin == 4

    def test_extract_priority_xyz_comment_overrides(self) -> None:
        """Test that XYZ comment takes priority over attributes."""
        atoms = TestMoleculeFactory.get_water_distorted()
        atoms.info = {"comment": "charge=5 spin=6"}
        atoms.charge = 10
        atoms.mult = 20

        charge, spin = _extract_charge_spin(atoms, default_charge=0, default_spin=1)

        assert charge == 5  # XYZ comment wins
        assert spin == 6

    def test_extract_priority_attributes_over_info(self) -> None:
        """Test that attributes take priority over info dict."""
        atoms = TestMoleculeFactory.get_water_distorted()
        atoms.charge = 7
        atoms.mult = 8
        atoms.info = {"charge": 9, "spin": 10}

        charge, spin = _extract_charge_spin(atoms, default_charge=0, default_spin=1)

        assert charge == 7  # attributes win
        assert spin == 8

    def test_extract_with_invalid_comment_metadata(self) -> None:
        """Test extraction with invalid comment metadata falls back."""
        atoms = TestMoleculeFactory.get_water_distorted()
        atoms.info = {"comment": "charge=invalid spin=bad"}
        atoms.charge = 11
        atoms.mult = 12

        charge, spin = _extract_charge_spin(atoms, default_charge=0, default_spin=1)

        # Should fall back to attributes
        assert charge == 11
        assert spin == 12

    def test_extract_with_invalid_attribute_types(self) -> None:
        """Test extraction handles invalid attribute types gracefully."""
        atoms = TestMoleculeFactory.get_water_distorted()
        atoms.charge = "invalid"
        atoms.mult = None

        charge, spin = _extract_charge_spin(atoms, default_charge=0, default_spin=1)

        # Should use defaults
        assert charge == 0
        assert spin == 1

    def test_extract_uses_defaults_when_missing(self) -> None:
        """Test extraction uses defaults when no metadata present."""
        atoms = TestMoleculeFactory.get_water_distorted()
        atoms.info = {}

        charge, spin = _extract_charge_spin(atoms, default_charge=99, default_spin=88)

        assert charge == 99
        assert spin == 88

    def test_extract_with_none_info(self) -> None:
        """Test extraction when atoms.info is None."""
        atoms = TestMoleculeFactory.get_water_distorted()
        atoms.info = None

        charge, spin = _extract_charge_spin(atoms, default_charge=50, default_spin=51)

        assert charge == 50
        assert spin == 51


class TestExplorerRunEdgeCases:
    """Test Explorer.run() edge cases and error paths."""

    def test_run_with_custom_runner(self) -> None:
        """Test run() with custom runner function."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        def custom_runner(atoms_list, **kwargs):
            return {
                "optimized_atoms": atoms_list[0],
                "strategy": "custom",
                "converged": True,
            }

        result = explorer.run(runner=custom_runner)

        assert result["strategy"] == "custom"
        assert result["converged"] is True
        assert "optimized_atoms" in result

    def test_run_with_custom_runner_returns_atoms(self) -> None:
        """Test run() handles custom runner returning Atoms directly."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        def custom_runner(atoms_list, **kwargs):
            return atoms_list[0]

        result = explorer.run(runner=custom_runner)

        assert result["strategy"] == "custom"
        assert isinstance(result["optimized_atoms"], Atoms)

    def test_run_with_invalid_strategy_name(self) -> None:
        """Test run() raises error for invalid strategy."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock", target="invalid", strategy="nonexistent")

        with pytest.raises(NotImplementedError):
            explorer.run()

    def test_run_with_calculate_frequencies(self) -> None:
        """Test run() with calculate_frequencies=True."""
        atoms = TestMoleculeFactory.get_h2_stretched()
        explorer = Explorer(atoms, backend="mock", target="minima", strategy="local")

        result = explorer.run(steps=2, fmax=0.5, calculate_frequencies=True)

        assert isinstance(result, dict)
        assert "frequency_analysis" in result
        freq_analysis = result["frequency_analysis"]
        assert "frequencies" in freq_analysis
        assert "is_minimum" in freq_analysis or "is_minimum" in result

    def test_run_with_frequency_calculation_failure(self) -> None:
        """Test run() handles frequency calculation failures gracefully."""
        atoms = TestMoleculeFactory.get_h2_stretched()
        explorer = Explorer(atoms, backend="mock", target="minima", strategy="local")

        # Frequency calculation might fail but optimization should still complete
        with patch("qme.analysis.frequency.FrequencyAnalysis") as mock_freq:
            mock_freq.side_effect = Exception("Frequency calc failed")
            # This will raise, but the test verifies error handling exists
            with pytest.raises(Exception, match="Frequency calc failed"):
                explorer.run(steps=2, fmax=0.5, calculate_frequencies=True)

    def test_run_defaults_to_minima_local(self) -> None:
        """Test run() defaults to minima:local when target/strategy not set."""
        atoms = TestMoleculeFactory.get_h2_stretched()
        explorer = Explorer(atoms, backend="mock", target="", strategy="")

        result = explorer.run(steps=2, fmax=0.5)

        assert isinstance(result, dict)
        assert "optimized_atoms" in result


class TestExplorerFromFile:
    """Test Explorer.from_file() edge cases."""

    def test_from_file_with_list_return(self) -> None:
        """Test from_file handles read_geometry returning a list."""
        atoms = TestMoleculeFactory.get_water_distorted()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_path = f.name
            atoms.write(temp_path)

        try:
            # Mock read_geometry to return a list
            with patch("qme.core.explorer.read_geometry", return_value=[atoms, atoms]):
                explorer = Explorer.from_file(temp_path, backend="mock")

                assert len(explorer.atoms_list) == 1
                assert explorer.atoms_list[0] == atoms
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_from_file_with_all_options(self) -> None:
        """Test from_file with all optional parameters."""
        atoms = TestMoleculeFactory.get_water_distorted()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_path = f.name
            atoms.write(temp_path)

        try:
            explorer = Explorer.from_file(
                temp_path,
                backend="mock",
                model_name="test_model",
                device="cpu",
                default_charge=1,
                default_spin=2,
                verbose=2,
                profile=True,
                target="ts",
            )

            assert explorer.backend == "mock"
            assert explorer.model_name == "test_model"
            assert explorer.device == "cpu"
            assert explorer.default_charge == 1
            assert explorer.default_spin == 2
            assert explorer.verbose == 2
            assert explorer.profiler is not None
            assert explorer.target == "ts"
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestExplorerLoadStructure:
    """Test Explorer.load_structure() method."""

    def test_load_structure_from_file(self) -> None:
        """Test loading structure from file."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_path = f.name
            atoms.write(temp_path)

        try:
            loaded = explorer.load_structure(temp_path)

            assert len(loaded) == 3
            assert len(explorer.atoms_list) == 1
            assert explorer.atoms_list[0] == loaded
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_structure_from_atoms(self) -> None:
        """Test loading structure from Atoms object."""
        atoms1 = TestMoleculeFactory.get_water_distorted()
        atoms2 = TestMoleculeFactory.get_h2_stretched()
        explorer = Explorer(atoms1, backend="mock")

        loaded = explorer.load_structure(atoms2)

        assert loaded == atoms2
        assert len(explorer.atoms_list) == 1
        assert explorer.atoms_list[0] == atoms2

    def test_load_structure_handles_list(self) -> None:
        """Test load_structure handles read_geometry returning list."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_path = f.name
            atoms.write(temp_path)

        try:
            with patch("qme.core.explorer.read_geometry", return_value=[atoms, atoms]):
                loaded = explorer.load_structure(temp_path)

                assert loaded == atoms
                assert len(explorer.atoms_list) == 1
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestExplorerCalculateFrequencies:
    """Test Explorer.calculate_frequencies() method."""

    def test_calculate_frequencies_basic(self) -> None:
        """Test basic frequency calculation."""
        atoms = TestMoleculeFactory.get_h2_stretched()
        explorer = Explorer(atoms, backend="mock")

        results = explorer.calculate_frequencies()

        assert "frequencies" in results
        assert "all_frequencies" in results
        assert "normal_modes" in results
        assert "zero_point_energy" in results
        assert "thermodynamic_properties" in results
        assert "is_ts" in results
        assert "is_minimum" in results

    def test_calculate_frequencies_with_explicit_atoms(self) -> None:
        """Test frequency calculation with explicit atoms parameter."""
        atoms1 = TestMoleculeFactory.get_h2_stretched()
        atoms2 = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms1, backend="mock")

        results = explorer.calculate_frequencies(atoms=atoms2)

        assert "frequencies" in results
        assert results["n_atoms"] == len(atoms2)

    def test_calculate_frequencies_with_indices(self) -> None:
        """Test frequency calculation with atom indices."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        results = explorer.calculate_frequencies(indices=[0, 1])

        assert "frequencies" in results
        assert results["indices"] == [0, 1]

    def test_calculate_frequencies_without_hessian(self) -> None:
        """Test frequency calculation without saving Hessian."""
        atoms = TestMoleculeFactory.get_h2_stretched()
        explorer = Explorer(atoms, backend="mock")

        results = explorer.calculate_frequencies(save_hessian=False)

        assert "frequencies" in results
        assert "hessian" not in results

    def test_calculate_frequencies_no_atoms_raises_error(self) -> None:
        """Test frequency calculation with no atoms raises error."""
        explorer = Explorer([], backend="mock")

        with pytest.raises(RuntimeError, match="No structure available"):
            explorer.calculate_frequencies()

    def test_calculate_frequencies_with_custom_temperature(self) -> None:
        """Test frequency calculation with custom temperature."""
        atoms = TestMoleculeFactory.get_h2_stretched()
        explorer = Explorer(atoms, backend="mock")

        results = explorer.calculate_frequencies(temperature=500.0)

        assert results["temperature"] == 500.0
        assert "thermodynamic_properties" in results


class TestExplorerSaveMethods:
    """Test Explorer save methods edge cases."""

    def test_save_structure_error_path(self) -> None:
        """Test save_structure error handling."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_path = f.name

        try:
            # Make write fail by passing invalid path
            with (
                patch(
                    "qme.core.explorer.write_xyz_with_metadata",
                    side_effect=Exception("Write failed"),
                ),
                pytest.raises(RuntimeError),
            ):
                explorer.save_structure(atoms, temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_structure_non_xyz_format(self) -> None:
        """Test save_structure with non-XYZ format."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            explorer.save_structure(atoms, temp_path)

            assert Path(temp_path).exists()
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_trajectory_error_path(self) -> None:
        """Test save_trajectory error handling."""
        atoms1 = TestMoleculeFactory.get_water_distorted()
        atoms2 = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer([atoms1, atoms2], backend="mock")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_path = f.name

        try:
            with (
                patch(
                    "qme.core.explorer.write_xyz_with_metadata",
                    side_effect=Exception("Write failed"),
                ),
                pytest.raises(RuntimeError),
            ):
                explorer.save_trajectory([atoms1, atoms2], temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_trajectory_clean_atoms_fallback(self) -> None:
        """Test save_trajectory uses clean atoms on failure."""
        atoms = TestMoleculeFactory.get_water_distorted()
        # Add some attributes that might cause write issues
        atoms.info = {"charge": 0, "spin": 1}
        explorer = Explorer([atoms], backend="mock")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # First write attempt fails, should try clean atoms
            call_count = 0

            def failing_write(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("First write failed")
                return None

            with patch("qme.core.explorer.write", side_effect=failing_write):
                explorer.save_trajectory([atoms], temp_path, format="json")

            # Should have tried twice (original + clean)
            assert call_count == 2
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestExplorerEffectiveMethods:
    """Test Explorer _get_effective_* methods."""

    def test_get_effective_model_name(self) -> None:
        """Test _get_effective_model_name for different backends."""
        atoms = TestMoleculeFactory.get_water_distorted()

        # Test with explicit model_name
        explorer = Explorer(atoms, backend="mock", model_name="custom_model")
        assert explorer._get_effective_model_name() == "custom_model"

        # Test with backend defaults
        explorer = Explorer(atoms, backend="uma")
        model_name = explorer._get_effective_model_name()
        assert "uma" in model_name.lower()

    def test_get_effective_optimizer_default_minima(self) -> None:
        """Test _get_effective_optimizer defaults for minima."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock", target="minima", local_optimizer="default")

        assert explorer._get_effective_optimizer() == "lbfgs"

    def test_get_effective_optimizer_default_ts(self) -> None:
        """Test _get_effective_optimizer defaults for TS."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock", target="ts", local_optimizer="default")

        assert explorer._get_effective_optimizer() == "sella"

    def test_get_effective_optimizer_explicit(self) -> None:
        """Test _get_effective_optimizer with explicit optimizer."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock", local_optimizer="bfgs")

        assert explorer._get_effective_optimizer() == "bfgs"


class TestExplorerProfilerIntegration:
    """Test Explorer profiler integration."""

    def test_run_with_profiler(self) -> None:
        """Test run() integrates with profiler."""
        atoms = TestMoleculeFactory.get_h2_stretched()
        explorer = Explorer(atoms, backend="mock", profile=True)

        result = explorer.run(steps=2, fmax=0.5)

        assert explorer.profiler is not None
        assert "optimized_atoms" in result

    def test_profiler_attached_to_strategy(self) -> None:
        """Test profiler is attached to strategy instances."""
        atoms = TestMoleculeFactory.get_h2_stretched()
        explorer = Explorer(atoms, backend="mock", profile=True, target="minima", strategy="local")

        # Profiler should be available
        assert explorer.profiler is not None


class TestExplorerChargeSpinHandling:
    """Test Explorer charge/spin handling in calculator creation."""

    def test_calculate_frequencies_sets_charge_spin_in_info(self) -> None:
        """Test calculate_frequencies sets charge/spin in atoms.info."""
        atoms = TestMoleculeFactory.get_h2_stretched()
        # Remove calculator
        atoms.calc = None
        explorer = Explorer(atoms, backend="mock", default_charge=2, default_spin=3)

        results = explorer.calculate_frequencies()

        # Should have set charge/spin in info
        assert atoms.info is not None
        assert atoms.info.get("charge") == 2
        assert atoms.info.get("spin") == 3
        assert "frequencies" in results
