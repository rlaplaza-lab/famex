from __future__ import annotations

import pytest

from famex.core.base_strategy import BaseStrategy, StrategyMetadata
from famex.core.exceptions import StrategyError
from famex.core.registry import REGISTRY, StrategyRegistry


def _make_strategy(
    name: str = "test:strategy",
    *,
    target: str = "test",
    strategy: str = "strategy",
    description: str = "Test",
    aliases: list[str] | None = None,
):
    if aliases is None:
        aliases = ["test"]

    class _Strategy(BaseStrategy):
        metadata = StrategyMetadata(
            name=name,
            target=target,
            strategy=strategy,
            description=description,
            aliases=aliases,
        )

        def run(self, atoms_list, **kwargs):
            return {"optimized_atoms": atoms_list[0]}

    return _Strategy


@pytest.fixture
def strategy_registry():
    return StrategyRegistry()


class TestStrategyRegistry:
    def test_registry_initialization(self, strategy_registry):
        assert len(strategy_registry._strategies) == 0

    def test_register_strategy(self, strategy_registry):
        TestStrategy = _make_strategy()
        strategy_registry.register(TestStrategy)

        assert "test:strategy" in strategy_registry._strategies
        assert strategy_registry._strategies["test:strategy"] == TestStrategy

    def test_register_strategy_auto_aliases(self, strategy_registry):
        class TestStrategy(BaseStrategy):
            metadata = StrategyMetadata(
                name="test:strategy",
                target="test",
                strategy="strategy",
                description="Test",
                aliases=["test", "alias1", "alias2"],
            )

            def run(self, atoms_list, **kwargs):
                return {"optimized_atoms": atoms_list[0]}

        strategy_registry.register(TestStrategy)

        assert "test:strategy" in strategy_registry._strategies
        assert "test" in strategy_registry._strategies
        assert "alias1" in strategy_registry._strategies
        assert "alias2" in strategy_registry._strategies
        assert strategy_registry._strategies["test"] == TestStrategy
        assert strategy_registry._strategies["alias1"] == TestStrategy

    def test_register_strategy_missing_metadata(self, strategy_registry):
        class BadStrategy:
            pass

        with pytest.raises(StrategyError, match="missing metadata"):
            strategy_registry.register(BadStrategy)

    def test_register_strategy_duplicate_alias(self, strategy_registry):
        class Strategy1(BaseStrategy):
            metadata = StrategyMetadata(
                name="test:strategy1",
                target="test",
                strategy="strategy1",
                description="Test",
                aliases=["common"],
            )

            def run(self, atoms_list, **kwargs):
                return {"optimized_atoms": atoms_list[0]}

        class Strategy2(BaseStrategy):
            metadata = StrategyMetadata(
                name="test:strategy2",
                target="test",
                strategy="strategy2",
                description="Test",
                aliases=["common"],  # Duplicate alias
            )

            def run(self, atoms_list, **kwargs):
                return {"optimized_atoms": atoms_list[0]}

        strategy_registry.register(Strategy1)

        with pytest.raises(StrategyError, match="already registered"):
            strategy_registry.register(Strategy2)

    def test_get_strategy_by_name(self, strategy_registry):
        class TestStrategy(BaseStrategy):
            metadata = StrategyMetadata(
                name="test:strategy",
                target="test",
                strategy="strategy",
                description="Test",
                aliases=["test"],
            )

            def run(self, atoms_list, **kwargs):
                return {"optimized_atoms": atoms_list[0]}

        strategy_registry.register(TestStrategy)

        retrieved = strategy_registry.get("test:strategy")
        assert retrieved == TestStrategy

    def test_get_strategy_by_alias(self, strategy_registry):
        class TestStrategy(BaseStrategy):
            metadata = StrategyMetadata(
                name="test:strategy",
                target="test",
                strategy="strategy",
                description="Test",
                aliases=["test"],
            )

            def run(self, atoms_list, **kwargs):
                return {"optimized_atoms": atoms_list[0]}

        strategy_registry.register(TestStrategy)

        retrieved = strategy_registry.get("test")
        assert retrieved == TestStrategy

    def test_get_strategy_not_found(self, strategy_registry):
        from famex.core.exceptions import StrategyNotFoundError

        with pytest.raises(StrategyNotFoundError, match="No strategy found"):
            strategy_registry.get("nonexistent")

    def test_list_strategies(self, strategy_registry):
        class Strategy1(BaseStrategy):
            metadata = StrategyMetadata(
                name="test:strategy1",
                target="test",
                strategy="strategy1",
                description="Test 1",
                aliases=["test1"],
            )

            def run(self, atoms_list, **kwargs):
                return {"optimized_atoms": atoms_list[0]}

        class Strategy2(BaseStrategy):
            metadata = StrategyMetadata(
                name="test:strategy2",
                target="test",
                strategy="strategy2",
                description="Test 2",
                aliases=["test2"],
            )

            def run(self, atoms_list, **kwargs):
                return {"optimized_atoms": atoms_list[0]}

        strategy_registry.register(Strategy1)
        strategy_registry.register(Strategy2)

        strategies = strategy_registry.list_strategies()

        assert isinstance(strategies, dict)
        assert "test:strategy1" in strategies
        assert "test:strategy2" in strategies
        # list_strategies returns StrategyMetadata objects directly
        assert strategies["test:strategy1"].name == "test:strategy1"
        assert strategies["test:strategy2"].name == "test:strategy2"

    def test_list_strategies_no_duplicates(self, strategy_registry):
        class TestStrategy(BaseStrategy):
            metadata = StrategyMetadata(
                name="test:strategy",
                target="test",
                strategy="strategy",
                description="Test",
                aliases=["test", "alias1", "alias2"],
            )

            def run(self, atoms_list, **kwargs):
                return {"optimized_atoms": atoms_list[0]}

        strategy_registry.register(TestStrategy)

        strategies = strategy_registry.list_strategies()

        # Should only have one entry (the main name, not aliases)
        assert len(strategies) == 1
        assert "test:strategy" in strategies
        assert "test" not in strategies  # Aliases shouldn't appear


class TestGlobalRegistry:
    def test_global_registry_exists(self):
        assert REGISTRY is not None
        assert isinstance(REGISTRY, StrategyRegistry)

    def test_global_registry_has_registered_strategies(self):
        # Import strategies to trigger registration
        import famex.strategies  # noqa: F401

        strategies = REGISTRY.list_strategies()

        # Should have at least some strategies registered
        assert len(strategies) > 0
        assert isinstance(strategies, dict)

        # Check that common strategies are registered
        strategy_names = list(strategies.keys())
        # At least one strategy should have "minima" or "path" or "ts" in name
        assert any("minima" in name or "path" in name or "ts" in name for name in strategy_names)
