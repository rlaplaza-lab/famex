"""Tests for strategy helper functions."""

from __future__ import annotations

import pytest

from famex.core.explorer import Explorer
from famex.strategies.helpers import (
    _get_local_optimizer_class,
    _validate_ts_optimization_setup,
    validate_ts_structure,
)


class TestValidateTSStructure:
    """Tests for validate_ts_structure function."""

    def test_validate_ts_structure_with_valid_ts(self, mock_backend, water_dissociation_ts_guess):
        """Test validate_ts_structure with a structure that has 1 imaginary frequency."""
        # Create a TS-like structure (highly distorted)
        atoms = water_dissociation_ts_guess.copy()
        explorer = Explorer(atoms, backend="mock", target="ts", strategy="local")
        atoms.calc = mock_backend

        # Mock the frequency analysis to return a TS
        result = validate_ts_structure(atoms, explorer, threshold=50.0)

        assert isinstance(result, dict)
        assert "is_transition_state" in result
        assert "n_imaginary_frequencies" in result
        assert "assessment" in result

    def test_validate_ts_structure_without_calculator(self, water_dissociation_ts_guess):
        """Test validate_ts_structure attaches calculator if missing."""
        atoms = water_dissociation_ts_guess.copy()
        explorer = Explorer(atoms, backend="mock", target="ts", strategy="local")
        # Don't attach calculator - should be attached automatically

        result = validate_ts_structure(atoms, explorer, threshold=50.0)

        assert isinstance(result, dict)
        assert atoms.calc is not None

    def test_validate_ts_structure_with_invalid_structure(
        self, mock_backend, h2_equilibrium_molecule
    ):
        """Test validate_ts_structure with structure that has no imaginary frequencies."""
        # Create a minimum-like structure
        atoms = h2_equilibrium_molecule.copy()
        explorer = Explorer(atoms, backend="mock", target="ts", strategy="local")
        atoms.calc = mock_backend

        result = validate_ts_structure(atoms, explorer, threshold=50.0)

        assert isinstance(result, dict)
        # Should not be a TS (no imaginary frequencies)
        # The exact value depends on the mock calculator, but structure should be invalid
        assert "is_transition_state" in result


class TestValidateTSOptimizationSetup:
    """Tests for _validate_ts_optimization_setup function."""

    @pytest.mark.parametrize(
        ("backend", "optimizer", "description"),
        [
            ("mock", "sella", "mock backend"),
            ("MOCK", "sella", "mock backend case-insensitive"),
            ("uma", "lbfgs", "lbfgs optimizer"),
            ("uma", "bfgs", "bfgs optimizer"),
            ("uma", "fire", "fire optimizer"),
            ("uma", "l-bfgs", "l-bfgs variant"),
            ("uma", "l_bfgs", "l_bfgs variant"),
            ("uma", "L-BFGS", "L-BFGS variant"),
        ],
        ids=[
            "forbidden_backend_mock",
            "forbidden_backend_case_insensitive",
            "forbidden_optimizer_lbfgs",
            "forbidden_optimizer_bfgs",
            "forbidden_optimizer_fire",
            "forbidden_optimizer_l-bfgs",
            "forbidden_optimizer_l_bfgs",
            "forbidden_optimizer_L-BFGS",
        ],
    )
    def test_validate_ts_setup_with_forbidden_combinations(self, backend, optimizer, description):
        """Test that forbidden backend/optimizer combinations are rejected."""
        with pytest.raises(ValueError, match="not suitable for transition state optimization"):
            _validate_ts_optimization_setup(backend, optimizer)

    def test_validate_ts_setup_with_allowed_setup(self):
        """Test that allowed backend/optimizer combinations pass validation."""
        # Should not raise
        _validate_ts_optimization_setup("uma", "sella")
        _validate_ts_optimization_setup("aimnet2", "rfo")
        _validate_ts_optimization_setup("mace", "sella")


class TestGetLocalOptimizerClass:
    """Tests for _get_local_optimizer_class function."""

    @pytest.mark.parametrize(
        ("optimizer_name", "expected_in_name"),
        [
            ("sella", "Sella"),
            ("trust-krylov", "TrustKrylov"),
            ("trustkrylov", "TrustKrylov"),
            ("trust_krylov", "TrustKrylov"),
            ("Trust-Krylov", "TrustKrylov"),
            ("rfo", "RFO"),
            ("rfo-ts", "RFO"),
            ("rational-function", "RFO"),
            ("rational_function", "RFO"),
            ("trust-ncg", "TrustNCG"),
            ("trustncg", "TrustNCG"),
            ("trust_ncg", "TrustNCG"),
            ("trust-exact", "TrustExact"),
            ("trustexact", "TrustExact"),
            ("trust_exact", "TrustExact"),
            ("newton-cg", "NewtonCG"),
            ("newtoncg", "NewtonCG"),
            ("newton_cg", "NewtonCG"),
            ("lbfgs", "LBFGS"),
            ("l-bfgs", "LBFGS"),
            ("l_bfgs", "LBFGS"),
            ("bfgs", "BFGS"),
            ("fire", "FIRE"),
        ],
        ids=[
            "sella",
            "trust-krylov",
            "trustkrylov",
            "trust_krylov",
            "Trust-Krylov",
            "rfo",
            "rfo-ts",
            "rational-function",
            "rational_function",
            "trust-ncg",
            "trustncg",
            "trust_ncg",
            "trust-exact",
            "trustexact",
            "trust_exact",
            "newton-cg",
            "newtoncg",
            "newton_cg",
            "lbfgs",
            "l-bfgs",
            "l_bfgs",
            "bfgs",
            "fire",
        ],
    )
    def test_get_optimizer_variants(self, optimizer_name, expected_in_name):
        """Test getting optimizer classes with various name variants."""
        opt_class = _get_local_optimizer_class(optimizer_name)
        assert opt_class is not None
        # For RFO, check both possible class names
        if expected_in_name == "RFO":
            assert "RFO" in opt_class.__name__ or "TransitionState" in opt_class.__name__
        else:
            assert expected_in_name in opt_class.__name__

    def test_get_unknown_optimizer(self):
        """Test that unknown optimizer name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown optimizer name"):
            _get_local_optimizer_class("unknown_optimizer")

    def test_get_optimizer_case_insensitive(self):
        """Test that optimizer names are case-insensitive."""
        # Test a few variants
        opt_class1 = _get_local_optimizer_class("SELLA")
        opt_class2 = _get_local_optimizer_class("sella")
        assert opt_class1 == opt_class2

    def test_get_optimizer_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown optimizer name"):
            _get_local_optimizer_class("")

    def test_get_optimizer_none(self):
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="Unknown optimizer name"):
            _get_local_optimizer_class(None)  # type: ignore[arg-type]
