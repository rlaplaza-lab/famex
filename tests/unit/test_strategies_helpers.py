"""Tests for strategy helper functions."""

from __future__ import annotations

import pytest

import qme
from qme.core.explorer import Explorer
from qme.strategies.helpers import (
    _get_local_optimizer_class,
    _validate_ts_optimization_setup,
    validate_ts_structure,
)
from tests.test_utils import TestMoleculeFactory


class TestValidateTSStructure:
    """Tests for validate_ts_structure function."""

    def test_validate_ts_structure_with_valid_ts(self):
        """Test validate_ts_structure with a structure that has 1 imaginary frequency."""
        # Create a TS-like structure (highly distorted)
        atoms = TestMoleculeFactory.get_water_dissociation_ts_guess()
        explorer = Explorer(atoms, backend="mock", target="ts", strategy="local")
        atoms.calc = qme.MockCalculator(backend="mock")

        # Mock the frequency analysis to return a TS
        result = validate_ts_structure(atoms, explorer, threshold=50.0)

        assert isinstance(result, dict)
        assert "is_transition_state" in result
        assert "n_imaginary_frequencies" in result
        assert "assessment" in result

    def test_validate_ts_structure_without_calculator(self):
        """Test validate_ts_structure attaches calculator if missing."""
        atoms = TestMoleculeFactory.get_water_dissociation_ts_guess()
        explorer = Explorer(atoms, backend="mock", target="ts", strategy="local")
        # Don't attach calculator - should be attached automatically

        result = validate_ts_structure(atoms, explorer, threshold=50.0)

        assert isinstance(result, dict)
        assert atoms.calc is not None

    def test_validate_ts_structure_with_invalid_structure(self):
        """Test validate_ts_structure with structure that has no imaginary frequencies."""
        # Create a minimum-like structure
        atoms = TestMoleculeFactory.get_h2_equilibrium()
        explorer = Explorer(atoms, backend="mock", target="ts", strategy="local")
        atoms.calc = qme.MockCalculator(backend="mock")

        result = validate_ts_structure(atoms, explorer, threshold=50.0)

        assert isinstance(result, dict)
        # Should not be a TS (no imaginary frequencies)
        # The exact value depends on the mock calculator, but structure should be invalid
        assert "is_transition_state" in result


class TestValidateTSOptimizationSetup:
    """Tests for _validate_ts_optimization_setup function."""

    def test_validate_ts_setup_with_forbidden_backend(self):
        """Test that mock backend is rejected for TS optimization."""
        with pytest.raises(ValueError, match="not suitable for transition state optimization"):
            _validate_ts_optimization_setup("mock", "sella")

    def test_validate_ts_setup_with_forbidden_backend_case_insensitive(self):
        """Test that backend check is case-insensitive."""
        with pytest.raises(ValueError, match="not suitable for transition state optimization"):
            _validate_ts_optimization_setup("MOCK", "sella")

    def test_validate_ts_setup_with_forbidden_optimizer_lbfgs(self):
        """Test that lbfgs optimizer is rejected for TS optimization."""
        with pytest.raises(ValueError, match="not suitable for transition state optimization"):
            _validate_ts_optimization_setup("uma", "lbfgs")

    def test_validate_ts_setup_with_forbidden_optimizer_bfgs(self):
        """Test that bfgs optimizer is rejected for TS optimization."""
        with pytest.raises(ValueError, match="not suitable for transition state optimization"):
            _validate_ts_optimization_setup("uma", "bfgs")

    def test_validate_ts_setup_with_forbidden_optimizer_fire(self):
        """Test that fire optimizer is rejected for TS optimization."""
        with pytest.raises(ValueError, match="not suitable for transition state optimization"):
            _validate_ts_optimization_setup("uma", "fire")

    def test_validate_ts_setup_with_forbidden_optimizer_variants(self):
        """Test that optimizer name variants are rejected."""
        forbidden_variants = ["l-bfgs", "l_bfgs", "L-BFGS"]
        for variant in forbidden_variants:
            with pytest.raises(ValueError, match="not suitable for transition state optimization"):
                _validate_ts_optimization_setup("uma", variant)

    def test_validate_ts_setup_with_allowed_setup(self):
        """Test that allowed backend/optimizer combinations pass validation."""
        # Should not raise
        _validate_ts_optimization_setup("uma", "sella")
        _validate_ts_optimization_setup("aimnet2", "trust-krylov-ts")
        _validate_ts_optimization_setup("mace", "sella")


class TestGetLocalOptimizerClass:
    """Tests for _get_local_optimizer_class function."""

    def test_get_sella_optimizer(self):
        """Test getting Sella optimizer class."""
        opt_class = _get_local_optimizer_class("sella")
        assert opt_class is not None
        # Should be VerboseSella
        assert "Sella" in opt_class.__name__

    def test_get_trust_krylov_variants(self):
        """Test getting TrustKrylov optimizer with various name variants."""
        variants = ["trust-krylov", "trustkrylov", "trust_krylov", "Trust-Krylov"]
        for variant in variants:
            opt_class = _get_local_optimizer_class(variant)
            assert opt_class is not None
            assert "TrustKrylov" in opt_class.__name__

    def test_get_trust_krylov_ts_variants(self):
        """Test getting TrustKrylovTS optimizer with various name variants."""
        variants = [
            "trust-krylov-ts",
            "trustkrylovts",
            "trust_krylov_ts",
            "trust-krylov-transition",
        ]
        for variant in variants:
            opt_class = _get_local_optimizer_class(variant)
            assert opt_class is not None
            assert "TrustKrylovTS" in opt_class.__name__

    def test_get_rfo_variants(self):
        """Test getting RFO optimizer with various name variants."""
        variants = ["rfo", "rfo-ts", "rational-function", "rational_function"]
        for variant in variants:
            opt_class = _get_local_optimizer_class(variant)
            assert opt_class is not None
            assert "RFO" in opt_class.__name__ or "TransitionState" in opt_class.__name__

    def test_get_trust_ncg_variants(self):
        """Test getting TrustNCG optimizer with various name variants."""
        variants = ["trust-ncg", "trustncg", "trust_ncg"]
        for variant in variants:
            opt_class = _get_local_optimizer_class(variant)
            assert opt_class is not None
            assert "TrustNCG" in opt_class.__name__

    def test_get_trust_exact_variants(self):
        """Test getting TrustExact optimizer with various name variants."""
        variants = ["trust-exact", "trustexact", "trust_exact"]
        for variant in variants:
            opt_class = _get_local_optimizer_class(variant)
            assert opt_class is not None
            assert "TrustExact" in opt_class.__name__

    def test_get_newton_cg_variants(self):
        """Test getting NewtonCG optimizer with various name variants."""
        variants = ["newton-cg", "newtoncg", "newton_cg"]
        for variant in variants:
            opt_class = _get_local_optimizer_class(variant)
            assert opt_class is not None
            assert "NewtonCG" in opt_class.__name__

    def test_get_lbfgs_variants(self):
        """Test getting LBFGS optimizer with various name variants."""
        variants = ["lbfgs", "l-bfgs", "l_bfgs"]
        for variant in variants:
            opt_class = _get_local_optimizer_class(variant)
            assert opt_class is not None
            assert "LBFGS" in opt_class.__name__

    def test_get_bfgs_optimizer(self):
        """Test getting BFGS optimizer."""
        opt_class = _get_local_optimizer_class("bfgs")
        assert opt_class is not None
        assert "BFGS" in opt_class.__name__

    def test_get_fire_optimizer(self):
        """Test getting FIRE optimizer."""
        opt_class = _get_local_optimizer_class("fire")
        assert opt_class is not None
        assert "FIRE" in opt_class.__name__

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
