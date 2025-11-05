from __future__ import annotations

import pytest


class TestPotentialsInitBasic:
    @pytest.mark.parametrize(
        "class_name",
        ["BasePotential", "MockCalculator"],
    )
    def test_basic_imports(self, class_name):
        module = __import__("qme.potentials", fromlist=[class_name])
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
            "get_tblite_calculator",
            "get_torchsim_mace_calculator",
            "get_torchsim_uma_calculator",
        ],
    )
    def test_calculator_factory_functions_exist(self, factory_name):
        module = __import__("qme.potentials", fromlist=[factory_name])
        factory_func = getattr(module, factory_name)
        assert callable(factory_func)

    @pytest.mark.parametrize(
        "backend",
        ["definitely_unavailable_backend_xyz", "unknown_backend_not_in_mapping"],
    )
    def test_get_calculator_generic_with_invalid_backend(self, backend):
        from qme.potentials import _get_calculator_generic

        with pytest.raises(ImportError):
            _get_calculator_generic(backend)
