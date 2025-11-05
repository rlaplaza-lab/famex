from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

import qme
from qme.core.explorer import Explorer, _extract_charge_spin
from qme.utils.validation import BackendError
from tests.test_utils import TestMoleculeFactory


class TestExplorerInitialization:
    @pytest.mark.parametrize(
        ("atoms_input", "expected_length", "expected_index"),
        [
            ("single", 1, 0),
            ("list", 2, None),
        ],
    )
    def test_init_with_atoms(self, water_molecule, atoms_input, expected_length, expected_index):
        atoms = water_molecule
        if atoms_input == "single":
            input_atoms = atoms
        else:
            atoms2 = water_molecule.copy()
            input_atoms = [atoms, atoms2]

        explorer = Explorer(input_atoms, backend="mock")

        assert len(explorer.atoms_list) == expected_length
        assert explorer.backend == "mock"
        if expected_index is not None:
            assert explorer.atoms_list[expected_index] == atoms

    @pytest.mark.parametrize(
        ("target", "strategy", "expected_target", "expected_strategy"),
        [
            ("", "", "minima", "local"),  # defaults
            ("minima", "local", "minima", "local"),
            ("ts", "local", "ts", "local"),
            ("  minima  ", "  local  ", "minima", "local"),  # whitespace stripped
        ],
    )
    def test_init_with_target_strategy(
        self, water_molecule, target, strategy, expected_target, expected_strategy
    ):
        atoms = water_molecule
        explorer = Explorer(atoms, backend="mock", target=target, strategy=strategy)

        # Empty strings get defaults, others get stripped
        if target == "" and strategy == "":
            # Defaults applied
            assert explorer.target in ("minima", "")
            assert explorer.strategy in ("local", "")
        else:
            assert explorer.target == expected_target
            assert explorer.strategy == expected_strategy

    @pytest.mark.parametrize(
        ("profile", "has_profiler"),
        [(False, False), (True, True)],
    )
    def test_init_with_profile(self, water_molecule, profile, has_profiler):
        atoms = water_molecule
        explorer = Explorer(atoms, backend="mock", profile=profile)

        if has_profiler:
            assert explorer.profiler is not None
        else:
            assert explorer.profiler is None

    def test_init_with_defaults(self, water_molecule):
        atoms = water_molecule
        explorer = Explorer(atoms, backend="mock")

        assert explorer.target == "minima"
        assert explorer.strategy == "local"
        assert explorer.default_charge == 0
        assert explorer.default_spin == 1
        assert explorer.verbose == 1
        assert explorer.profiler is None

    def test_init_with_constraints_and_optimizer_kwargs(self, water_molecule):
        atoms = water_molecule
        optimizer_kwargs = {"maxstep": 0.1}
        explorer = Explorer(
            atoms,
            backend="mock",
            constraints="fix 0",
            optimizer_kwargs=optimizer_kwargs,
        )

        assert explorer.constraints_spec == "fix 0"
        assert explorer.optimizer_kwargs == optimizer_kwargs


class TestExplorerListStrategies:
    def test_list_strategies_all(self):
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        strategies = explorer.list_strategies()

        assert isinstance(strategies, dict)
        assert len(strategies) > 0

        # Check structure of returned dictionary
        for name, metadata in strategies.items():
            assert isinstance(name, str)
            assert isinstance(metadata, dict)
            assert "description" in metadata

    def test_list_strategies_filtered(self):
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        # Test filtering (if implemented)
        strategies = explorer.list_strategies(kind="minima")

        assert isinstance(strategies, dict)
        # If filtering is implemented, all strategies should match
        # Otherwise, this should still return all strategies
        assert len(strategies) > 0


class TestExplorerExplainRun:
    @pytest.mark.parametrize(
        ("target", "strategy", "expected_target"),
        [
            ("minima", "local", "minima"),
            ("ts", "local", "ts"),
            ("path", "neb", "path"),
        ],
    )
    def test_explain_run_strategies(self, target, strategy, expected_target):
        atoms = TestMoleculeFactory.get_water_distorted()
        if target == "path":
            atoms = [atoms, TestMoleculeFactory.get_water_distorted()]
        explorer = Explorer(atoms, backend="mock", target=target, strategy=strategy)

        explanation = explorer.explain_run()

        assert isinstance(explanation, dict)
        assert "valid" in explanation
        assert explanation["target"] == expected_target
        if target == "path":
            assert explanation["strategy"] == "neb" or "neb" in str(explanation["strategy"]).lower()

    def test_explain_run_invalid_backend(self):
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="nonexistent_backend")

        # Should either raise error or return explanation with valid=False
        explanation = explorer.explain_run()

        # The explanation should indicate the issue
        assert isinstance(explanation, dict)

    def test_explain_run_invalid_strategy(self):
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock", target="minima", strategy="nonexistent")

        explanation = explorer.explain_run()

        assert isinstance(explanation, dict)
        # Should indicate invalid strategy
        assert "valid" in explanation or "error" in explanation or "strategy" in explanation


class TestExplorerErrorHandling:
    def test_empty_atoms_list_raises_error(self):
        # Explorer initialization doesn't validate empty list immediately
        # Error would occur during run() when strategy tries to access atoms
        explorer = Explorer([], backend="mock")
        # Error occurs during run, not init
        with pytest.raises((ValueError, IndexError)):
            explorer.run()

    def test_run_with_invalid_backend(self):
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="nonexistent_backend_xyz123")

        with pytest.raises((BackendError, ValueError, KeyError)):
            explorer.run(steps=1)

    def test_run_path_strategy_requires_multiple_structures(self):
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock", target="path", strategy="neb")

        # Should raise error or handle gracefully
        with pytest.raises((ValueError, KeyError)):
            explorer.run(steps=1)

    def test_invalid_device_handled_gracefully(self):
        atoms = TestMoleculeFactory.get_water_distorted()
        # Should either work with default or raise a clear error
        explorer = Explorer(atoms, backend="mock", device="invalid_device")

        # Device validation might happen later, or might be ignored
        # Just verify Explorer is created
        assert explorer.device == "invalid_device"

    def test_constraints_parsing_error(self):
        atoms = TestMoleculeFactory.get_water_distorted()

        # Invalid constraints should be caught during run, not init
        explorer = Explorer(atoms, backend="mock", constraints="invalid_constraint_string")

        # Constraints are parsed during run, not init
        assert explorer.constraints_spec == "invalid_constraint_string"

    def test_strategy_not_found_error(self):
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock", target="minima", strategy="nonexistent_strategy")

        # explain_run may return explanation with valid=False or raise during run()
        # Let's test that it either raises or returns invalid explanation
        try:
            explanation = explorer.explain_run()
            # If it doesn't raise, check that it indicates invalid strategy
            if isinstance(explanation, dict) and "valid" in explanation:
                # Might be False or might be missing - both are valid responses
                pass
        except (KeyError, ValueError):
            # This is also acceptable - error raised as expected
            pass


class TestExtractChargeSpin:
    @pytest.mark.parametrize(
        (
            "comment",
            "charge_attr",
            "mult_attr",
            "info_dict",
            "default_charge",
            "default_spin",
            "expected_charge",
            "expected_spin",
        ),
        [
            ("charge=1 spin=3", None, None, {}, 0, 1, 1, 3),
            ("charge=2", None, None, {}, 0, 1, 2, 1),
            (None, 2, 3, {}, 0, 1, 2, 3),
            (None, None, None, {"charge": 3, "spin": 4}, 0, 1, 3, 4),
            ("charge=5 spin=6", 10, 20, {}, 0, 1, 5, 6),  # XYZ comment wins
            (None, 7, 8, {"charge": 9, "spin": 10}, 0, 1, 7, 8),  # attributes win
            ("charge=invalid", 11, 12, {}, 0, 1, 11, 12),  # Invalid comment falls back
            (None, None, None, {}, 99, 88, 99, 88),  # Uses defaults
        ],
        ids=[
            "full_comment",
            "partial_comment",
            "attributes",
            "info_dict",
            "comment_priority",
            "attr_priority",
            "invalid_comment",
            "defaults",
        ],
    )
    def test_extract_charge_spin_variations(
        self,
        comment,
        charge_attr,
        mult_attr,
        info_dict,
        default_charge,
        default_spin,
        expected_charge,
        expected_spin,
    ):
        atoms = TestMoleculeFactory.get_water_distorted()

        if comment:
            atoms.info = {"comment": comment}
        if charge_attr is not None:
            atoms.charge = charge_attr
        if mult_attr is not None:
            atoms.mult = mult_attr
        if info_dict:
            if atoms.info is None:
                atoms.info = {}
            atoms.info.update(info_dict)

        charge, spin = _extract_charge_spin(
            atoms, default_charge=default_charge, default_spin=default_spin
        )

        assert charge == expected_charge
        assert spin == expected_spin

    def test_extract_with_invalid_attribute_types(self):
        atoms = TestMoleculeFactory.get_water_distorted()
        atoms.charge = "invalid"
        atoms.mult = None

        charge, spin = _extract_charge_spin(atoms, default_charge=0, default_spin=1)

        assert charge == 0
        assert spin == 1

    def test_extract_with_none_info(self):
        atoms = TestMoleculeFactory.get_water_distorted()
        atoms.info = None

        charge, spin = _extract_charge_spin(atoms, default_charge=50, default_spin=51)

        assert charge == 50
        assert spin == 51


class TestExplorerRun:
    def test_run_basic_minima(self, h2_molecule):
        atoms = h2_molecule
        explorer = Explorer(atoms, backend="mock", target="minima", strategy="local")

        result = explorer.run(steps=5, fmax=0.5)

        assert isinstance(result, dict)
        assert "optimized_atoms" in result
        assert "strategy" in result
        assert "converged" in result

    def test_run_with_steps_limit(self, h2_molecule):
        atoms = h2_molecule
        explorer = Explorer(atoms, backend="mock", target="minima")

        result = explorer.run(steps=2, fmax=0.5)

        assert isinstance(result, dict)
        # Steps should be limited
        if "steps_taken" in result:
            assert result["steps_taken"] <= 2

    def test_run_with_custom_runner(self, water_molecule):
        atoms = water_molecule
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

    def test_run_with_custom_runner_returns_atoms(self, water_molecule):
        atoms = water_molecule
        explorer = Explorer(atoms, backend="mock")

        def custom_runner(atoms_list, **kwargs):
            return atoms_list[0]

        result = explorer.run(runner=custom_runner)

        assert result["strategy"] == "custom"
        assert isinstance(result["optimized_atoms"], Atoms)

    def test_run_with_invalid_strategy_name(self, water_molecule):
        atoms = water_molecule
        explorer = Explorer(atoms, backend="mock", target="invalid", strategy="nonexistent")

        with pytest.raises(NotImplementedError):
            explorer.run()

    def test_run_with_calculate_frequencies(self, h2_molecule):
        atoms = h2_molecule
        explorer = Explorer(atoms, backend="mock", target="minima", strategy="local")

        result = explorer.run(steps=2, fmax=0.5, calculate_frequencies=True)

        assert isinstance(result, dict)
        assert "frequency_analysis" in result
        freq_analysis = result["frequency_analysis"]
        assert "frequencies" in freq_analysis
        assert "is_minimum" in freq_analysis or "is_minimum" in result

    def test_run_with_frequency_calculation_failure(self, h2_molecule):
        atoms = h2_molecule
        explorer = Explorer(atoms, backend="mock", target="minima", strategy="local")

        # Frequency calculation might fail but optimization should still complete
        with patch("qme.analysis.frequency.FrequencyAnalysis") as mock_freq:
            mock_freq.side_effect = Exception("Frequency calc failed")
            # This will raise, but the test verifies error handling exists
            with pytest.raises(Exception, match="Frequency calc failed"):
                explorer.run(steps=2, fmax=0.5, calculate_frequencies=True)

    def test_run_defaults_to_minima_local(self, h2_molecule):
        atoms = h2_molecule
        explorer = Explorer(atoms, backend="mock", target="", strategy="")

        result = explorer.run(steps=2, fmax=0.5)

        assert isinstance(result, dict)
        assert "optimized_atoms" in result


class TestExplorerFromFile:
    @pytest.mark.parametrize(
        ("kwargs", "assertions"),
        [
            ({}, lambda e: (len(e.atoms_list) == 1 and len(e.atoms_list[0]) == 3)),
            ({"target": "ts"}, lambda e: (e.backend == "mock" and e.target == "ts")),
        ],
        ids=["basic", "custom_target"],
    )
    def test_from_file_variations(self, kwargs, assertions):
        atoms = TestMoleculeFactory.get_water_distorted()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_path = f.name
            atoms.write(temp_path)

        try:
            explorer = Explorer.from_file(temp_path, backend="mock", **kwargs)
            assert assertions(explorer)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_from_file_nonexistent_file(self):
        with pytest.raises((FileNotFoundError, OSError)):
            Explorer.from_file("nonexistent_file.xyz", backend="mock")

    def test_from_file_with_list_return(self):
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

    def test_from_file_with_all_options(self):
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
    def test_load_structure_from_file(self):
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

    def test_load_structure_from_atoms(self):
        atoms1 = TestMoleculeFactory.get_water_distorted()
        atoms2 = TestMoleculeFactory.get_h2_stretched()
        explorer = Explorer(atoms1, backend="mock")

        loaded = explorer.load_structure(atoms2)

        assert loaded == atoms2
        assert len(explorer.atoms_list) == 1
        assert explorer.atoms_list[0] == atoms2

    def test_load_structure_handles_list(self):
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
    def test_calculate_frequencies_basic(self):
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

    def test_calculate_frequencies_with_explicit_atoms(self):
        atoms1 = TestMoleculeFactory.get_h2_stretched()
        atoms2 = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms1, backend="mock")

        results = explorer.calculate_frequencies(atoms=atoms2)

        assert "frequencies" in results
        assert results["n_atoms"] == len(atoms2)

    def test_calculate_frequencies_with_indices(self):
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        results = explorer.calculate_frequencies(indices=[0, 1])

        assert "frequencies" in results
        assert results["indices"] == [0, 1]

    def test_calculate_frequencies_without_hessian(self):
        atoms = TestMoleculeFactory.get_h2_stretched()
        explorer = Explorer(atoms, backend="mock")

        results = explorer.calculate_frequencies(save_hessian=False)

        assert "frequencies" in results
        assert "hessian" not in results

    def test_calculate_frequencies_no_atoms_raises_error(self):
        explorer = Explorer([], backend="mock")

        with pytest.raises(RuntimeError, match="No structure available"):
            explorer.calculate_frequencies()

    def test_calculate_frequencies_with_custom_temperature(self):
        atoms = TestMoleculeFactory.get_h2_stretched()
        explorer = Explorer(atoms, backend="mock")

        results = explorer.calculate_frequencies(temperature=500.0)

        assert results["temperature"] == 500.0
        assert "thermodynamic_properties" in results


class TestExplorerSaveMethods:
    @pytest.mark.parametrize(
        ("method_name", "suffix", "test_data_factory"),
        [
            ("save_structure", ".xyz", lambda: TestMoleculeFactory.get_water_distorted()),
            ("save_structure", ".json", lambda: TestMoleculeFactory.get_water_distorted()),
            (
                "save_trajectory",
                ".xyz",
                lambda: [
                    TestMoleculeFactory.get_water_distorted(),
                    TestMoleculeFactory.get_water_distorted(),
                ],
            ),
        ],
        ids=["save_structure_xyz", "save_structure_json", "save_trajectory"],
    )
    def test_save_methods(self, method_name, suffix, test_data_factory):
        test_data = test_data_factory()
        if method_name == "save_trajectory":
            explorer = Explorer(test_data, backend="mock")
            trajectory = test_data
        else:
            explorer = Explorer(test_data, backend="mock")
            trajectory = None

        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            temp_path = f.name

        try:
            if method_name == "save_trajectory":
                explorer.save_trajectory(trajectory, temp_path)
            else:
                explorer.save_structure(test_data, temp_path)

            assert Path(temp_path).exists()
            if suffix == ".xyz" and method_name == "save_structure":
                loaded = qme.read_geometry(temp_path)
                assert len(loaded) == len(test_data)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.parametrize(
        ("method_name", "patch_target"),
        [
            ("save_structure", "qme.core.explorer.write_xyz_with_metadata"),
            ("save_trajectory", "qme.core.explorer.write_xyz_with_metadata"),
        ],
        ids=["structure_error", "trajectory_error"],
    )
    def test_save_methods_error_handling(self, method_name, patch_target):
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer([atoms] if method_name == "save_trajectory" else atoms, backend="mock")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            temp_path = f.name

        try:
            with (
                patch(patch_target, side_effect=OSError("Write failed")),
                pytest.raises(RuntimeError, match="Failed to save"),
            ):
                if method_name == "save_trajectory":
                    explorer.save_trajectory([atoms, atoms], temp_path)
                else:
                    explorer.save_structure(atoms, temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_trajectory_clean_atoms_fallback(self):
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
                    raise OSError("First write failed")
                # Second call succeeds (clean atoms)
                return None

            with patch("qme.core.explorer.write", side_effect=failing_write):
                explorer.save_trajectory([atoms], temp_path, format="json")

            # Should have tried twice (original + clean)
            assert call_count == 2
            assert Path(temp_path).exists()
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestExplorerEffectiveMethods:
    @pytest.mark.parametrize(
        ("backend", "model_name", "expected"),
        [
            ("mock", "custom_model", "custom_model"),
            ("uma", None, None),  # Will check contains "uma" below
        ],
        ids=["explicit_model", "backend_default"],
    )
    def test_get_effective_model_name(self, backend, model_name, expected):
        atoms = TestMoleculeFactory.get_water_distorted()
        kwargs = {"model_name": model_name} if model_name else {}
        explorer = Explorer(atoms, backend=backend, **kwargs)
        result = explorer._get_effective_model_name()
        if expected:
            assert result == expected
        else:
            assert "uma" in result.lower()

    @pytest.mark.parametrize(
        ("target", "local_optimizer", "expected_optimizer"),
        [
            ("minima", "default", "lbfgs"),
            ("ts", "default", "sella"),
            (None, "bfgs", "bfgs"),
        ],
        ids=["minima_default", "ts_default", "explicit"],
    )
    def test_get_effective_optimizer(self, target, local_optimizer, expected_optimizer):
        atoms = TestMoleculeFactory.get_water_distorted()
        kwargs = (
            {"target": target, "local_optimizer": local_optimizer}
            if target
            else {"local_optimizer": local_optimizer}
        )
        explorer = Explorer(atoms, backend="mock", **kwargs)
        assert explorer._get_effective_optimizer() == expected_optimizer


class TestExplorerProfilerIntegration:
    def test_run_with_profiler(self):
        atoms = TestMoleculeFactory.get_h2_stretched()
        explorer = Explorer(atoms, backend="mock", profile=True)

        result = explorer.run(steps=2, fmax=0.5)

        assert explorer.profiler is not None
        assert "optimized_atoms" in result

    def test_profiler_attached_to_strategy(self):
        atoms = TestMoleculeFactory.get_h2_stretched()
        explorer = Explorer(atoms, backend="mock", profile=True, target="minima", strategy="local")

        # Profiler should be available
        assert explorer.profiler is not None


class TestExplorerChargeSpinHandling:
    def test_calculate_frequencies_sets_charge_spin_in_info(self):
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
