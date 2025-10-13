"""Test Explorer strategy selection logic."""

import pytest
from ase import Atoms

from qme.core.explorer import Explorer
from qme.core.validation import QMEBackendError, QMEValidationError


class TestExplorerStrategySelection:
    """Test strategy selection and explain_run functionality."""

    def test_normalize_strategy_params(self):
        """Test parameter normalization."""
        atoms = Atoms("H2")

        # Test local strategy normalization
        exp = Explorer(atoms, strategy="local", target="minima")
        mode, strategy = exp._normalize_strategy_params("ts")
        assert mode == "ts"
        assert strategy == "local"

        # Test two-ended strategy normalization
        exp = Explorer(atoms, strategy="two-ended", target="path")
        mode, strategy = exp._normalize_strategy_params("neb")
        assert mode == "neb"
        assert strategy == "two-ended"

        # Test default mode
        exp = Explorer(atoms, strategy="local")
        mode, strategy = exp._normalize_strategy_params()
        assert mode == "minima"
        assert strategy == "local"

    def test_select_strategy_local(self):
        """Test local strategy selection."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, strategy="local")

        # Test minima selection
        strategy_key, strategy_type = exp._select_strategy("minima", "local")
        assert strategy_key == "local:minima"  # First in preferred list
        assert strategy_type == "local"

        # Test TS selection
        strategy_key, strategy_type = exp._select_strategy("ts", "local")
        assert strategy_key == "local:ts"  # First in preferred list
        assert strategy_type == "local"

    def test_select_strategy_twoended(self):
        """Test two-ended strategy selection."""
        atoms = Atoms("H2")

        # Test minima selection
        exp = Explorer(atoms, strategy="two-ended", target="minima")
        strategy_key, strategy_type = exp._select_strategy("minima", "two-ended")
        assert strategy_key == "twoended:minima"
        assert strategy_type == "two-ended"

        # Test TS selection
        exp = Explorer(atoms, strategy="two-ended", target="ts")
        strategy_key, strategy_type = exp._select_strategy("ts", "two-ended")
        assert strategy_key == "twoended:ts"
        assert strategy_type == "two-ended"

        # Test path selection
        exp = Explorer(atoms, strategy="two-ended", target="path")
        strategy_key, strategy_type = exp._select_strategy("neb", "two-ended")
        assert strategy_key == "twoended:path"
        assert strategy_type == "two-ended"

    def test_explain_run_local_minima(self):
        """Test explain_run for local minima optimization."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, strategy="local", target="minima", backend="mock")

        explanation = exp.explain_run("minima")

        assert explanation["valid"] is True
        assert explanation["strategy"] == "local:minima"
        assert explanation["strategy_type"] == "local"
        assert "local_minima_runner" in explanation["runner"]
        assert "Will use" in explanation["notes"]

    def test_explain_run_twoended_ts(self):
        """Test explain_run for two-ended TS optimization."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, strategy="two-ended", target="ts", backend="mock")

        explanation = exp.explain_run("ts")

        assert explanation["valid"] is True
        assert explanation["strategy"] == "twoended:ts"
        assert explanation["strategy_type"] == "two-ended"
        assert "twoended_ts_guess_runner" in explanation["runner"]

    def test_explain_run_invalid_strategy(self):
        """Test explain_run with invalid strategy."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, strategy="invalid", auto_register=False)

        # Clear all strategies to test invalid case
        exp._strategies = {}

        explanation = exp.explain_run("minima")

        assert explanation["valid"] is False
        assert "error" in explanation

    def test_validation_minima_run(self):
        """Test validation for minima runs."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, strategy="local", target="minima", backend="mock")

        # Should not raise for valid setup
        explanation = exp.explain_run("minima")
        assert explanation["valid"] is True

    def test_validation_ts_run(self):
        """Test validation for TS runs."""
        atoms = Atoms("H2")

        # Should raise for mock backend with TS
        exp = Explorer(atoms, strategy="local", target="ts", backend="mock")

        with pytest.raises(QMEBackendError):
            exp.run(mode="ts")

    def test_validation_path_run(self):
        """Test validation for path runs."""
        atoms = Atoms("H2")

        # Should not raise for valid two-ended setup
        exp = Explorer(atoms, strategy="two-ended", target="path", backend="mock")
        explanation = exp.explain_run("neb")
        assert explanation["valid"] is True

        # Should raise for single atoms with path
        exp = Explorer(atoms, strategy="local", target="path", backend="mock")
        with pytest.raises(QMEValidationError):
            exp.run(mode="neb")

    def test_run_method_simplified(self):
        """Test that run method is simplified and uses helper methods."""
        atoms = Atoms("H2")
        exp = Explorer(
            atoms, strategy="local", target="minima", backend="mock", local_optimizer="lbfgs"
        )

        # Test that explain_run works
        explanation = exp.explain_run()
        assert explanation["valid"] is True

        # Test that actual run works (should return dict now)
        result = exp.run(mode="minima", steps=1)
        assert isinstance(result, dict)
        assert "optimized_atoms" in result
        assert "strategy" in result
        assert result["strategy"] == "local_minima_runner"
