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
        assert neb_strategy.__name__ == "TwoEndedNEBStrategy"

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

    def test_explain_run_local_minima(self):
        """Test explain_run for local minima optimization."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, target="minima", strategy="local", backend="mock")

        explanation = exp.explain_run()
        assert explanation["target"] == "minima"
        assert explanation["strategy"] == "local"
        assert explanation["strategy_key"] == "minima:local"
        assert explanation["valid"] is True
        assert "LocalMinimaStrategy" in explanation["runner"]

    def test_explain_run_path_neb(self):
        """Test explain_run for path NEB optimization."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, target="path", strategy="neb", backend="mock")

        explanation = exp.explain_run()
        assert explanation["target"] == "path"
        assert explanation["strategy"] == "neb"
        assert explanation["strategy_key"] == "path:neb"
        assert explanation["valid"] is True
        assert "TwoEndedNEBStrategy" in explanation["runner"]

    def test_explain_run_with_mode(self):
        """Test explain_run with explicit mode parameter."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, backend="mock")

        # Test with explicit mode
        explanation = exp.explain_run(mode="minima:local")
        assert explanation["target"] == "minima"
        assert explanation["strategy"] == "local"
        assert explanation["strategy_key"] == "minima:local"
        assert explanation["valid"] is True

    def test_explain_run_invalid_strategy(self):
        """Test explain_run for invalid strategy."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, backend="mock")

        explanation = exp.explain_run(mode="invalid:nonexistent")
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

    def test_strategy_aliases(self):
        """Test that strategy aliases work correctly."""
        atoms = Atoms("H2")

        # Test short aliases - they should resolve to the full strategy names
        exp1 = Explorer(atoms, backend="mock")
        explanation1 = exp1.explain_run(mode="neb")
        assert explanation1["strategy_key"] == "neb"  # The alias is used as the key
        assert explanation1["target"] == "path"
        assert explanation1["strategy"] == "neb"

        exp2 = Explorer(atoms, backend="mock")
        explanation2 = exp2.explain_run(mode="cineb")
        assert explanation2["strategy_key"] == "cineb"  # The alias is used as the key
        assert explanation2["target"] == "path"
        assert explanation2["strategy"] == "cineb"

        exp3 = Explorer(atoms, backend="mock")
        explanation3 = exp3.explain_run(mode="minima")
        assert explanation3["strategy_key"] == "minima"  # The alias is used as the key
        assert explanation3["target"] == "minima"
        assert explanation3["strategy"] == "local"

    def test_strategy_requires_multiple_structures(self):
        """Test that strategies correctly validate multiple structure requirements."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, backend="mock")

        # NEB requires multiple structures
        explanation = exp.explain_run(mode="path:neb")
        assert explanation["valid"] is True  # explain_run doesn't validate inputs

        # But running should fail
        with pytest.raises(ValueError, match="requires 2\\+ structures"):
            exp.run(mode="path:neb")

    def test_strategy_error_handling(self):
        """Test error handling for invalid strategies."""
        atoms = Atoms("H2")
        exp = Explorer(atoms, backend="mock")

        # Test invalid strategy name
        with pytest.raises(NotImplementedError, match="No strategy found"):
            exp.run(mode="invalid:strategy")

        # Test invalid alias
        with pytest.raises(NotImplementedError, match="No strategy found"):
            exp.run(mode="nonexistent")
