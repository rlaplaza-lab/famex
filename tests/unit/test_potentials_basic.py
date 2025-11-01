"""Basic tests for potentials module initialization."""

import pytest


class TestPotentialsInitBasic:
    """Basic tests for potentials module."""

    def test_base_potential_import(self):
        """Test BasePotential can be imported."""
        from qme.potentials import BasePotential

        assert BasePotential is not None

    def test_mock_calculator_import(self):
        """Test MockCalculator can be imported."""
        from qme.potentials import MockCalculator

        assert MockCalculator is not None

    def test_calculator_factory_functions_exist(self):
        """Test calculator factory functions exist."""
        from qme.potentials import (
            get_aimnet2_calculator,
            get_mace_calculator,
            get_orb_calculator,
            get_so3lr_calculator,
            get_tblite_calculator,
            get_torchsim_mace_calculator,
            get_torchsim_uma_calculator,
            get_uma_calculator,
        )

        assert callable(get_uma_calculator)
        assert callable(get_so3lr_calculator)
        assert callable(get_aimnet2_calculator)
        assert callable(get_mace_calculator)
        assert callable(get_orb_calculator)
        assert callable(get_tblite_calculator)
        assert callable(get_torchsim_mace_calculator)
        assert callable(get_torchsim_uma_calculator)

    def test_get_calculator_generic_with_unavailable_backend(self):
        """Test _get_calculator_generic with unavailable backend."""
        from qme.potentials import _get_calculator_generic

        with pytest.raises(ImportError):
            _get_calculator_generic("definitely_unavailable_backend_xyz")

    def test_get_calculator_generic_with_unknown_backend(self):
        """Test _get_calculator_generic with unknown backend."""
        from qme.potentials import _get_calculator_generic

        # Find a backend that's available but not in _BACKEND_MODULES
        # Actually, all known backends are in the mapping
        # So we test with definitely unknown one
        with pytest.raises(ImportError):
            _get_calculator_generic("unknown_backend_not_in_mapping")
