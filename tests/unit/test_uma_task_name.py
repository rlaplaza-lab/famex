"""Tests for UMA task_name and DFT-D3 plumbing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from famex.backends.cache import CalculatorCache
from famex.backends.constants import DEFAULT_UMA_MODEL, DEFAULT_UMA_TASK
from famex.backends.registry import create_calculator
from famex.core.calculator_manager import CalculatorManager


def _import_uma_potential():
    # uma_potential imports torch at module scope; keep collection torch-free.
    pytest.importorskip("torch")
    from famex.potentials.uma_potential import UMAPotential

    return UMAPotential


class TestUMATaskNameDefaults:
    def test_default_task_is_omol(self):
        UMAPotential = _import_uma_potential()
        pot = UMAPotential(model_name=DEFAULT_UMA_MODEL, device="cpu")
        assert pot.task_name == DEFAULT_UMA_TASK
        assert pot.dftd3_functional is None

    def test_custom_task_name_normalized(self):
        UMAPotential = _import_uma_potential()
        pot = UMAPotential(model_name=DEFAULT_UMA_MODEL, task_name="OMAT", device="cpu")
        assert pot.task_name == "omat"

    def test_invalid_dftd3_functional_raises(self):
        UMAPotential = _import_uma_potential()
        with pytest.raises(ValueError, match="Unsupported dftd3_functional"):
            UMAPotential(
                model_name=DEFAULT_UMA_MODEL,
                dftd3_functional="b3lyp",
                device="cpu",
            )

    def test_non_omol_effective_spin_defaults_to_zero(self):
        UMAPotential = _import_uma_potential()
        pot = UMAPotential(
            model_name=DEFAULT_UMA_MODEL,
            task_name="omat",
            default_spin=1,
            device="cpu",
        )
        assert pot._effective_default_spin() == 0

    def test_omol_keeps_default_spin(self):
        UMAPotential = _import_uma_potential()
        pot = UMAPotential(
            model_name=DEFAULT_UMA_MODEL,
            task_name="omol",
            default_spin=1,
            device="cpu",
        )
        assert pot._effective_default_spin() == 1


class TestUMATaskNameLoad:
    def test_load_passes_task_name_to_fairchem(self):
        UMAPotential = _import_uma_potential()
        pot = UMAPotential(
            model_name=DEFAULT_UMA_MODEL,
            task_name="omat",
            needs_hessian=True,
            device="cpu",
        )

        mock_predictor = MagicMock()
        mock_pretrained = MagicMock()
        mock_pretrained.get_predict_unit.return_value = mock_predictor
        mock_fairchem_cls = MagicMock()
        mock_fairchem_cls.return_value = MagicMock()
        mock_settings_cls = MagicMock()
        mock_settings_cls.return_value = MagicMock(name="settings")

        with (
            patch("famex.potentials.uma_potential.deps.has", return_value=True),
            patch(
                "famex.potentials.uma_potential.deps.get",
                side_effect=lambda name, default=None: {
                    "fairchem_pretrained_mlip": mock_pretrained,
                    "fairchem_calculator": mock_fairchem_cls,
                    "fairchem_inference_settings": mock_settings_cls,
                    "fairchem_dftd3_calculator": None,
                }.get(name, default),
            ),
            patch.object(pot, "_set_model_precision"),
        ):
            pot._load_calculator()

        mock_settings_cls.assert_called_once()
        settings_kwargs = mock_settings_cls.call_args.kwargs
        assert settings_kwargs["predict_untrained_hessian"] == {"omat"}
        mock_fairchem_cls.assert_called_once_with(mock_predictor, task_name="omat")

    def test_load_wraps_dftd3_when_requested(self):
        UMAPotential = _import_uma_potential()
        pot = UMAPotential(
            model_name=DEFAULT_UMA_MODEL,
            task_name="omat",
            dftd3_functional="r2scan",
            device="cpu",
        )

        mock_predictor = MagicMock()
        mock_pretrained = MagicMock()
        mock_pretrained.get_predict_unit.return_value = mock_predictor
        mock_fairchem_calc = MagicMock(name="fairchem_calc")
        mock_fairchem_cls = MagicMock(return_value=mock_fairchem_calc)
        mock_dftd3_calc = MagicMock(name="dftd3_calc")
        mock_dftd3_cls = MagicMock(return_value=mock_dftd3_calc)

        with (
            patch("famex.potentials.uma_potential.deps.has", return_value=True),
            patch(
                "famex.potentials.uma_potential.deps.get",
                side_effect=lambda name, default=None: {
                    "fairchem_pretrained_mlip": mock_pretrained,
                    "fairchem_calculator": mock_fairchem_cls,
                    "fairchem_inference_settings": None,
                    "fairchem_dftd3_calculator": mock_dftd3_cls,
                }.get(name, default),
            ),
            patch.object(pot, "_set_model_precision"),
        ):
            pot._load_calculator()

        mock_dftd3_cls.assert_called_once_with(
            mock_fairchem_calc,
            functional="r2scan",
            device="cpu",
        )
        assert pot._fairchem_calc is mock_fairchem_calc
        assert pot._calc is mock_dftd3_calc


class TestCreateCalculatorUMATask:
    def test_create_calculator_forwards_task_kwargs(self):
        mock_cls = MagicMock(return_value=MagicMock())
        with patch(
            "famex.backends.registry.calculator_registry.create_calculator",
            return_value=mock_cls.return_value,
        ) as mock_create:
            create_calculator(
                backend="uma",
                model_name=DEFAULT_UMA_MODEL,
                model_path=None,
                device="cpu",
                default_charge=0,
                default_spin=1,
                use_cache=False,
                needs_hessian=True,
                task_name="oc25",
                dftd3_functional="pbe",
            )

        kwargs = mock_create.call_args.kwargs
        assert kwargs["needs_hessian"] is True
        assert kwargs["task_name"] == "oc25"
        assert kwargs["dftd3_functional"] == "pbe"

    def test_create_calculator_ignores_task_on_non_uma(self):
        with (
            patch(
                "famex.backends.registry.calculator_registry.create_calculator",
                return_value=MagicMock(),
            ) as mock_create,
            patch("famex.backends.registry.logger") as mock_logger,
        ):
            create_calculator(
                backend="aimnet2",
                model_name="aimnet2",
                model_path=None,
                device="cpu",
                default_charge=0,
                default_spin=1,
                use_cache=False,
                task_name="omat",
                dftd3_functional="r2scan",
            )

        kwargs = mock_create.call_args.kwargs
        assert "task_name" not in kwargs
        assert "dftd3_functional" not in kwargs
        mock_logger.warning.assert_called_once()


class TestCalculatorManagerTask:
    def test_manager_forwards_task_name(self):
        manager = CalculatorManager(
            backend="uma",
            model_name=DEFAULT_UMA_MODEL,
            device="cpu",
            task_name="odac",
            dftd3_functional="pbe",
        )
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        atoms.info["charge"] = 0
        atoms.info["spin"] = 0

        with patch(
            "famex.core.calculator_manager.create_calculator",
            return_value=MagicMock(),
        ) as mock_create:
            manager.create_and_attach_calculator(atoms, needs_hessian=False)

        kwargs = mock_create.call_args.kwargs
        assert kwargs["task_name"] == "odac"
        assert kwargs["dftd3_functional"] == "pbe"


class TestUMATaskCacheKeys:
    def test_cache_key_differs_for_task_name(self):
        cache = CalculatorCache()
        key_omol = cache._generate_key(
            "uma",
            DEFAULT_UMA_MODEL,
            "cpu",
            task_name="omol",
        )
        key_omat = cache._generate_key(
            "uma",
            DEFAULT_UMA_MODEL,
            "cpu",
            task_name="omat",
        )
        assert key_omol != key_omat

    def test_cache_key_differs_for_dftd3(self):
        cache = CalculatorCache()
        key_bare = cache._generate_key(
            "uma",
            DEFAULT_UMA_MODEL,
            "cpu",
            task_name="omat",
        )
        key_d3 = cache._generate_key(
            "uma",
            DEFAULT_UMA_MODEL,
            "cpu",
            task_name="omat",
            dftd3_functional="r2scan",
        )
        assert key_bare != key_d3
