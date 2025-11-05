from __future__ import annotations

import pytest
from ase import Atoms

from qme.core.base_strategy import BaseStrategy, StrategyMetadata


class TestStrategyMetadata:
    def test_metadata_creation(self):
        metadata = StrategyMetadata(
            name="test:strategy",
            target="test",
            strategy="strategy",
            description="Test strategy",
            aliases=["test", "ts"],
        )
        assert metadata.name == "test:strategy"
        assert metadata.target == "test"
        assert metadata.strategy == "strategy"
        assert metadata.description == "Test strategy"
        assert metadata.aliases == ["test", "ts"]
        assert metadata.requires_multiple_structures is False

    def test_metadata_multiple_structures_flag(self):
        metadata = StrategyMetadata(
            name="test:multi",
            target="path",
            strategy="neb",
            description="Multi-structure strategy",
            aliases=["neb"],
            requires_multiple_structures=True,
        )
        assert metadata.requires_multiple_structures is True

    def test_metadata_empty_aliases(self):
        metadata = StrategyMetadata(
            name="test:noalias",
            target="test",
            strategy="strategy",
            description="No aliases",
            aliases=[],
        )
        assert metadata.aliases == []

    def test_metadata_equality(self):
        metadata1 = StrategyMetadata(
            name="test:strategy",
            target="test",
            strategy="strategy",
            description="Test",
            aliases=["test"],
        )
        metadata2 = StrategyMetadata(
            name="test:strategy",
            target="test",
            strategy="strategy",
            description="Test",
            aliases=["test"],
        )
        assert metadata1 == metadata2


class ConcreteTestStrategy(BaseStrategy):
    metadata = StrategyMetadata(
        name="test:concrete",
        target="test",
        strategy="concrete",
        description="Concrete test strategy",
        aliases=["test", "concrete"],
        requires_multiple_structures=False,
    )

    def run(
        self,
        atoms_list,
        **kwargs,
    ):
        self.validate_inputs(atoms_list)
        optimized = atoms_list[0].copy()
        result = self.prepare_result(optimized, converged=True, steps_taken=10)
        return self._merge_profiler_results(result)


class TestBaseStrategy:
    def test_strategy_initialization(self):
        explorer_mock = object()
        strategy = ConcreteTestStrategy(explorer_mock)

        assert strategy.explorer is explorer_mock
        assert strategy.profiler is None

    def test_strategy_initialization_with_profiler(self):
        explorer_mock = object()
        profiler_mock = object()
        strategy = ConcreteTestStrategy(explorer_mock, profiler=profiler_mock)

        assert strategy.explorer is explorer_mock
        assert strategy.profiler is profiler_mock

    def test_validate_inputs_empty_list(self):
        explorer_mock = object()
        strategy = ConcreteTestStrategy(explorer_mock)

        with pytest.raises(ValueError, match="No atoms provided"):
            strategy.validate_inputs([])

    def test_validate_inputs_valid_list(self):
        explorer_mock = object()
        strategy = ConcreteTestStrategy(explorer_mock)
        atoms = Atoms("H2")

        # Should not raise
        strategy.validate_inputs([atoms])

    def test_prepare_result_single_atoms(self):
        explorer_mock = object()
        strategy = ConcreteTestStrategy(explorer_mock)
        atoms = Atoms("H2")

        result = strategy.prepare_result(atoms, converged=True, steps_taken=5)

        assert result["optimized_atoms"] == atoms
        assert result["strategy"] == "test:concrete"
        assert result["converged"] is True
        assert result["steps_taken"] == 5

    def test_prepare_result_list_atoms(self):
        explorer_mock = object()
        strategy = ConcreteTestStrategy(explorer_mock)
        atoms_list = [Atoms("H2"), Atoms("H2")]

        result = strategy.prepare_result(atoms_list, converged=True)

        assert result["optimized_atoms"] == atoms_list
        assert result["strategy"] == "test:concrete"
        assert result["converged"] is True

    def test_prepare_result_with_non_standard_types(self):
        explorer_mock = object()
        strategy = ConcreteTestStrategy(explorer_mock)
        atoms = Atoms("H2")

        # Use a dict or list as metadata (should be converted to string)
        result = strategy.prepare_result(atoms, metadata={"key": "value"})

        assert result["optimized_atoms"] == atoms
        assert result["strategy"] == "test:concrete"
        # Metadata dict should be converted to string
        assert isinstance(result["metadata"], str)

    def test_merge_profiler_results_no_profiler(self):
        explorer_mock = object()
        strategy = ConcreteTestStrategy(explorer_mock)
        result = {"test": "value"}

        merged = strategy._merge_profiler_results(result)

        assert merged == result
        assert "performance" not in merged

    def test_merge_profiler_results_with_profiler(self):
        explorer_mock = object()

        class MockProfiler:
            def get_summary(self):
                return {"time": 1.0, "memory": 100}

        profiler = MockProfiler()
        strategy = ConcreteTestStrategy(explorer_mock, profiler=profiler)
        result = {"test": "value"}

        merged = strategy._merge_profiler_results(result)

        assert merged["test"] == "value"
        assert "performance" in merged
        assert merged["performance"] == {"time": 1.0, "memory": 100}

    def test_run_abstract_method(self):
        explorer_mock = object()
        strategy = ConcreteTestStrategy(explorer_mock)
        atoms = Atoms("H2")

        # Concrete implementation should work
        result = strategy.run([atoms])

        assert "optimized_atoms" in result
        assert result["strategy"] == "test:concrete"
        assert result["converged"] is True
        assert result["steps_taken"] == 10

    def test_run_validates_inputs(self):
        explorer_mock = object()
        strategy = ConcreteTestStrategy(explorer_mock)

        # Should raise ValueError for empty list
        with pytest.raises(ValueError, match="No atoms provided"):
            strategy.run([])

    def test_metadata_attribute_exists(self):
        explorer_mock = object()
        strategy = ConcreteTestStrategy(explorer_mock)

        assert hasattr(strategy, "metadata")
        assert isinstance(strategy.metadata, StrategyMetadata)
        assert strategy.metadata.name == "test:concrete"
