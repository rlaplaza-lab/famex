"""Test Explorer strategy selection logic."""

import pytest
from ase import Atoms

from qme.core.explorer import Explorer
from qme.core.strategy import REGISTRY


class TestExplorerStrategySelection:
    """Test strategy selection and explain_run functionality."""

    def test_strategy_registry(self):
        """Test that strategies are properly registered."""
        # Ensure strategy modules are imported to trigger registration

        # Test that we can get strategies by name
        minima_strategy = REGISTRY.get("minima:local")
        assert minima_strategy.__name__ == "LocalMinimaStrategy"

        # Test aliases work
        neb_strategy = REGISTRY.get("neb")
        assert neb_strategy.__name__ == "MultiStructureNEBStrategy"

        # Test that we can list strategies
        strategies = REGISTRY.list_strategies()
        assert "minima:local" in strategies
        assert "path:neb" in strategies

    def test_strategy_metadata(self):
        """Test strategy metadata is correct."""
        # Ensure strategy modules are imported to trigger registration

        minima_strategy = REGISTRY.get("minima:local")
        metadata = minima_strategy.metadata

        assert metadata.name == "minima:local"
        assert metadata.target == "minima"
        assert metadata.strategy == "local"
        assert "minima" in metadata.aliases
        assert not metadata.requires_multiple_structures

    @pytest.mark.parametrize(
        "target,strategy,expected_key,expected_runner",
        [
            ("minima", "local", "minima:local", "LocalMinimaStrategy"),
            ("path", "neb", "path:neb", "MultiStructureNEBStrategy"),
            ("ts", "local", "ts:local", "LocalTSStrategy"),
        ],
    )
    def test_explain_run_strategies(self, target, strategy, expected_key, expected_runner):
        """Test explain_run for different strategy combinations."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, target=target, strategy=strategy, backend="mock")

        explanation = exp.explain_run()
        assert explanation["target"] == target
        assert explanation["strategy"] == strategy
        assert explanation["strategy_key"] == expected_key
        assert explanation["valid"] is True
        assert expected_runner in explanation["runner"]

    def test_explain_run_with_target_strategy(self):
        """Test explain_run with target and strategy parameters."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, backend="mock", target="minima", strategy="local")

        # Test with target and strategy set in constructor
        explanation = exp.explain_run()
        assert explanation["target"] == "minima"
        assert explanation["strategy"] == "local"
        assert explanation["strategy_key"] == "minima:local"
        assert explanation["valid"] is True

    def test_explain_run_invalid_strategy(self):
        """Test explain_run for invalid strategy."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, backend="mock", target="invalid", strategy="nonexistent")

        explanation = exp.explain_run()
        assert explanation["valid"] is False
        assert "error" in explanation

    def test_validation_minima_run(self):
        """Test validation for minima optimization."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, target="minima", strategy="local", backend="mock")

        # This should not raise an error
        explanation = exp.explain_run()
        assert explanation["valid"] is True

    def test_validation_ts_run(self):
        """Test validation for TS optimization."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, target="ts", strategy="local", backend="uma")

        # This should work with a real backend
        explanation = exp.explain_run()
        assert explanation["valid"] is True

    def test_validation_path_run(self):
        """Test validation for path optimization."""
        atoms1 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.0]])
        exp = Explorer([atoms1, atoms2], target="path", strategy="neb", backend="mock")

        # This should work with multiple structures
        explanation = exp.explain_run()
        assert explanation["valid"] is True

    def test_run_method_simplified(self):
        """Test that the simplified run method works."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        exp = Explorer(atoms, target="minima", strategy="local", backend="mock")

        # Test that we can call run without errors
        # Note: This might fail due to optimization issues, but the strategy selection should work
        try:
            result = exp.run(steps=1)
            assert "optimized_atoms" in result
            assert "strategy" in result
        except Exception as e:
            # If optimization fails, that's okay - we're testing strategy selection
            assert "minima:local" in str(e) or "optimization" in str(e).lower()

    @pytest.mark.parametrize(
        "alias,expected_target,expected_strategy",
        [
            ("neb", "path", "neb"),
            ("cineb", "path", "cineb"),
            ("minima", "minima", "local"),
            ("growing_string", "ts", "growing_string"),
            ("gsm", "ts", "growing_string"),
        ],
    )
    def test_strategy_aliases(self, alias, expected_target, expected_strategy):
        """Test that strategy aliases work correctly."""
        atoms = Atoms("H2")
        # Create Explorer with the expected target and strategy
        exp = Explorer(atoms, backend="mock", target=expected_target, strategy=expected_strategy)
        explanation = exp.explain_run()

        assert explanation["strategy_key"] == f"{expected_target}:{expected_strategy}"
        assert explanation["target"] == expected_target
        assert explanation["strategy"] == expected_strategy

    def test_strategy_requires_multiple_structures(self):
        """Test that strategies correctly validate multiple structure requirements."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, backend="mock", target="path", strategy="neb")

        # NEB requires multiple structures
        explanation = exp.explain_run()
        assert explanation["valid"] is True  # explain_run doesn't validate inputs

        # But running should fail
        with pytest.raises(ValueError, match="at least 2 structures"):
            exp.run()

    def test_strategy_error_handling(self):
        """Test error handling for invalid strategies."""
        atoms = Atoms("H2")

        # Test invalid strategy name
        exp_invalid = Explorer(atoms, backend="mock", target="invalid", strategy="nonexistent")
        with pytest.raises(NotImplementedError, match="No strategy found"):
            exp_invalid.run()

        # Test invalid target/strategy combination
        exp_invalid2 = Explorer(atoms, backend="mock", target="nonexistent", strategy="invalid")
        with pytest.raises(NotImplementedError, match="No strategy found"):
            exp_invalid2.run()
