"""Test Explorer strategy selection logic."""

import pytest
from ase import Atoms

from qme.core.explorer import Explorer
from qme.core.validation import QMEBackendError, QMEValidationError


class TestExplorerStrategySelection:
    """Test strategy selection and explain_run functionality."""

    def test_resolve_target_and_strategy(self):
        """Test target and strategy resolution."""
        atoms = Atoms("H2")

        # Test local strategy resolution
        exp = Explorer(atoms, target="minima", strategy="local")
        target, strategy = exp._resolve_target_and_strategy("ts")
        assert target == "ts"
        assert strategy == "local"

        # Test path strategy resolution
        exp = Explorer(atoms, target="path", strategy="neb")
        target, strategy = exp._resolve_target_and_strategy("cineb")
        assert target == "path"  # Target inferred from instance
        assert strategy == "cineb"  # Strategy from mode parameter

        # Test default resolution
        exp = Explorer(atoms, target="minima")
        target, strategy = exp._resolve_target_and_strategy()
        assert target == "minima"
        assert strategy == "local"

    def test_select_strategy_runner_local(self):
        """Test local strategy runner selection."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, target="minima", strategy="local")

        # Test minima selection
        strategy_key, strategy_type = exp._select_strategy_runner("minima", "local")
        assert strategy_key == "minima:local"
        assert strategy_type == "local"

        # Test TS selection
        exp = Explorer(atoms, target="ts", strategy="local")
        strategy_key, strategy_type = exp._select_strategy_runner("ts", "local")
        assert strategy_key == "ts:local"
        assert strategy_type == "local"

    def test_select_strategy_runner_path(self):
        """Test path strategy runner selection."""
        atoms = Atoms("H2")

        # Test NEB selection
        exp = Explorer(atoms, target="path", strategy="neb")
        strategy_key, strategy_type = exp._select_strategy_runner("path", "neb")
        assert strategy_key == "path:neb"
        assert strategy_type == "two-ended"

        # Test CI-NEB selection
        exp = Explorer(atoms, target="path", strategy="cineb")
        strategy_key, strategy_type = exp._select_strategy_runner("path", "cineb")
        assert strategy_key == "path:cineb"
        assert strategy_type == "two-ended"

        # Test interpolate selection
        exp = Explorer(atoms, target="path", strategy="interpolate")
        strategy_key, strategy_type = exp._select_strategy_runner("path", "interpolate")
        assert strategy_key == "path:interpolate"
        assert strategy_type == "two-ended"

    def test_explain_run_local_minima(self):
        """Test explain_run for local minima optimization."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, target="minima", strategy="local", backend="mock")

        explanation = exp.explain_run("minima")

        assert explanation["valid"] is True
        assert explanation["target"] == "minima"
        assert explanation["strategy"] == "local"
        assert explanation["strategy_key"] == "minima:local"
        assert explanation["strategy_type"] == "local"
        assert "local_minima_runner" in explanation["runner"]
        assert "Will use" in explanation["notes"]

    def test_explain_run_path_neb(self):
        """Test explain_run for path NEB optimization."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, target="path", strategy="neb", backend="mock")

        explanation = exp.explain_run("neb")

        assert explanation["valid"] is True
        assert explanation["target"] == "path"
        assert explanation["strategy"] == "neb"
        assert explanation["strategy_key"] == "path:neb"
        assert explanation["strategy_type"] == "two-ended"
        assert "twoended_neb_runner" in explanation["runner"]

    def test_explain_run_invalid_strategy(self):
        """Test explain_run with invalid strategy."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, target="minima", strategy="invalid", auto_register=False)

        # Clear all strategies to test invalid case
        exp._strategies = {}

        explanation = exp.explain_run("minima")

        assert explanation["valid"] is False
        assert "error" in explanation

    def test_validation_minima_run(self):
        """Test validation for minima runs."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, target="minima", strategy="local", backend="mock")

        # Should not raise for valid setup
        explanation = exp.explain_run("minima")
        assert explanation["valid"] is True

    def test_validation_ts_run(self):
        """Test validation for TS runs."""
        atoms = Atoms("H2")

        # Should raise for mock backend with TS
        exp = Explorer(atoms, target="ts", strategy="local", backend="mock")

        with pytest.raises(QMEBackendError):
            exp.run(mode="ts")

    def test_validation_path_run(self):
        """Test validation for path runs."""
        atoms = Atoms("H2")

        # Should not raise for valid path setup
        exp = Explorer(atoms, target="path", strategy="neb", backend="mock")
        explanation = exp.explain_run("neb")
        assert explanation["valid"] is True

        # Should raise for single atoms with path
        exp = Explorer(atoms, target="path", strategy="neb", backend="mock")
        with pytest.raises(QMEValidationError):
            exp.run(mode="neb")

    def test_run_method_simplified(self):
        """Test that run method is simplified and uses helper methods."""
        atoms = Atoms("H2")
        exp = Explorer(
            atoms, target="minima", strategy="local", backend="mock", local_optimizer="lbfgs"
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
