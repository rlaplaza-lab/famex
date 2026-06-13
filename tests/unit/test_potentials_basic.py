from __future__ import annotations

from unittest.mock import patch

import pytest
from ase import Atoms

from famex.potentials.base_potential import BasePotential


class TestBasePotential:
    """Test BasePotential class."""

    def test_get_logger_with_import_error(self):
        """Test get_module_logger() with ImportError fallback."""
        from famex.utils.lazy_imports import get_module_logger

        with patch("famex.utils.logging.get_famex_logger", side_effect=ImportError("No module")):
            logger = get_module_logger("test.module")
            # Should fallback to standard logging
            assert logger is not None

    def test_init_implemented_properties_from_kwargs(self):
        """Test initialization with implemented_properties from kwargs when not hasattr (line 83)."""
        potential = BasePotential(implemented_properties=["energy", "forces"])
        assert potential.implemented_properties == ["energy", "forces"]

    def test_calculate_with_none_atoms(self):
        """Test calculate() method with atoms=None (lines 102-105)."""
        potential = BasePotential()
        # Should not raise error
        potential.calculate(atoms=None)
        assert potential.atoms is None

    def test_calculate_with_none_properties(self):
        """Test calculate() method with properties=None."""
        potential = BasePotential(implemented_properties=["energy"])
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        potential.calculate(atoms=atoms, properties=None)
        # Should use implemented_properties
        assert potential.atoms == atoms

    def test_ensure_loaded_with_load_calculator(self):
        """Test ensure_loaded() with _load_calculator method (lines 148-151)."""
        potential = BasePotential()
        # Create a mock potential with _load_calculator
        potential._calc = None
        call_count = 0

        def mock_load():
            nonlocal call_count
            call_count += 1
            potential._calc = "mock_calculator"

        potential._load_calculator = mock_load
        result = potential.ensure_loaded()
        assert result == "mock_calculator"
        assert call_count == 1

    def test_ensure_loaded_without_load_calculator(self):
        """Test ensure_loaded() without _load_calculator method."""
        potential = BasePotential()
        potential._calc = None
        # Should return None if no _load_calculator
        result = potential.ensure_loaded()
        assert result is None

    def test_ensure_loaded_already_loaded(self):
        """Test ensure_loaded() when calculator is already loaded."""
        potential = BasePotential()
        potential._calc = "existing_calculator"
        result = potential.ensure_loaded()
        assert result == "existing_calculator"


class TestPotentialsInitBasic:
    @pytest.mark.parametrize(
        "class_name",
        ["BasePotential", "MockCalculator"],
    )
    def test_basic_imports(self, class_name):
        module = __import__("famex.potentials", fromlist=[class_name])
        cls = getattr(module, class_name)
        assert cls is not None

    @pytest.mark.parametrize(
        "factory_name",
        [
            "get_uma_calculator",
            "get_so3lr_calculator",
            "get_aimnet2_calculator",
            "get_mace_calculator",
            "get_orb_calculator",
            "get_pet_calculator",
            "get_tblite_calculator",
        ],
    )
    def test_calculator_factory_functions_exist(self, factory_name):
        module = __import__("famex.potentials", fromlist=[factory_name])
        factory_func = getattr(module, factory_name)
        assert callable(factory_func)

    @pytest.mark.parametrize(
        "backend",
        ["definitely_unavailable_backend_xyz", "unknown_backend_not_in_mapping"],
    )
    def test_get_calculator_generic_with_invalid_backend(self, backend):
        from famex.potentials import _get_calculator_generic

        with pytest.raises(ImportError):
            _get_calculator_generic(backend)


class TestPETPotentialParsing:
    @pytest.mark.parametrize(
        ("model_name", "expected_model", "expected_version"),
        [
            (None, "pet-mad-s", "latest"),
            ("pet-mad-s", "pet-mad-s", "latest"),
            ("pet-mad-s@1.5.0", "pet-mad-s", "1.5.0"),
            ("pet-oam-xl@1.0.0", "pet-oam-xl", "1.0.0"),
        ],
    )
    def test_parse_pet_model_name(self, model_name, expected_model, expected_version):
        from famex.potentials.pet_potential import parse_pet_model_name

        model, version = parse_pet_model_name(model_name)
        assert model == expected_model
        assert version == expected_version

    def test_pet_potential_parses_version_suffix(self):
        from famex.potentials.pet_potential import PETPotential

        potential = PETPotential(model_name="pet-mad-s@1.5.0")
        assert potential.pet_model == "pet-mad-s"
        assert potential.pet_version == "1.5.0"

    def test_pet_load_calculator_uses_model_and_version(self):
        import sys
        from unittest.mock import MagicMock

        from famex.potentials.pet_potential import PETPotential

        mock_upet_calc = MagicMock()
        mock_calculator_module = MagicMock()
        mock_calculator_module.UPETCalculator = mock_upet_calc
        mock_upet_module = MagicMock()
        mock_upet_module.calculator = mock_calculator_module

        with (
            patch("famex.potentials.pet_potential.deps") as mock_deps,
            patch.dict(
                sys.modules, {"upet": mock_upet_module, "upet.calculator": mock_calculator_module}
            ),
        ):
            mock_deps.has.return_value = True
            potential = PETPotential(model_name="pet-mad-s@1.5.0", device="cpu")
            potential._load_calculator()

        mock_upet_calc.assert_called_once_with(
            model="pet-mad-s",
            version="1.5.0",
            device="cpu",
        )

    def test_pet_load_calculator_uses_checkpoint_path(self):
        import sys
        from unittest.mock import MagicMock

        from famex.potentials.pet_potential import PETPotential

        mock_upet_calc = MagicMock()
        mock_calculator_module = MagicMock()
        mock_calculator_module.UPETCalculator = mock_upet_calc
        mock_upet_module = MagicMock()
        mock_upet_module.calculator = mock_calculator_module

        with (
            patch("famex.potentials.pet_potential.deps") as mock_deps,
            patch.dict(
                sys.modules, {"upet": mock_upet_module, "upet.calculator": mock_calculator_module}
            ),
        ):
            mock_deps.has.return_value = True
            potential = PETPotential(model_path="/path/to/model.ckpt", device="cpu")
            potential._load_calculator()

        mock_upet_calc.assert_called_once_with(
            checkpoint_path="/path/to/model.ckpt",
            device="cpu",
        )
