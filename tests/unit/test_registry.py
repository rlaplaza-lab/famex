"""Unit tests for StrategyRegistry."""

from __future__ import annotations

import pytest
from ase import Atoms

from qme.core.base_strategy import BaseStrategy, StrategyMetadata
from qme.core.registry import REGISTRY, StrategyRegistry


class TestStrategyRegistry:
    """Test StrategyRegistry functionality."""

    def test_registry_initialization(self) -> None:
        """Test registry initializes empty."""
        registry = StrategyRegistry()
        assert len(registry._strategies) == 0

    def test_register_strategy(self) -> None:
        """Test registering a strategy class."""
        registry = StrategyRegistry()

        class TestStrategy(BaseStrategy):
            metadata = StrategyMetadata(
                name="test:strategy",
                target="test",
                strategy="strategy",
                description="Test",
                aliases=["test"],
            )

            def run(self, atoms_list: list[Atoms], **kwargs):
                return {"optimized_atoms": atoms_list[0]}

        registry.register(TestStrategy)

        assert "test:strategy" in registry._strategies
        assert registry._strategies["test:strategy"] == TestStrategy

    def test_register_strategy_auto_aliases(self) -> None:
        """Test that aliases are automatically registered."""
        registry = StrategyRegistry()

        class TestStrategy(BaseStrategy):
            metadata = StrategyMetadata(
                name="test:strategy",
                target="test",
                strategy="strategy",
                description="Test",
                aliases=["test", "alias1", "alias2"],
            )

            def run(self, atoms_list: list[Atoms], **kwargs):
                return {"optimized_atoms": atoms_list[0]}

        registry.register(TestStrategy)

        assert "test:strategy" in registry._strategies
        assert "test" in registry._strategies
        assert "alias1" in registry._strategies
        assert "alias2" in registry._strategies
        assert registry._strategies["test"] == TestStrategy
        assert registry._strategies["alias1"] == TestStrategy

    def test_register_strategy_missing_metadata(self) -> None:
        """Test registering strategy without metadata raises error."""
        registry = StrategyRegistry()

        class BadStrategy:
            pass

        with pytest.raises(ValueError, match="missing metadata"):
            registry.register(BadStrategy)

    def test_register_strategy_duplicate_alias(self) -> None:
        """Test registering strategy with duplicate alias raises error."""
        registry = StrategyRegistry()

        class Strategy1(BaseStrategy):
            metadata = StrategyMetadata(
                name="test:strategy1",
                target="test",
                strategy="strategy1",
                description="Test",
                aliases=["common"],
            )

            def run(self, atoms_list: list[Atoms], **kwargs):
                return {"optimized_atoms": atoms_list[0]}

        class Strategy2(BaseStrategy):
            metadata = StrategyMetadata(
                name="test:strategy2",
                target="test",
                strategy="strategy2",
                description="Test",
                aliases=["common"],  # Duplicate alias
            )

            def run(self, atoms_list: list[Atoms], **kwargs):
                return {"optimized_atoms": atoms_list[0]}

        registry.register(Strategy1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(Strategy2)

    def test_get_strategy_by_name(self) -> None:
        """Test getting strategy by full name."""
        registry = StrategyRegistry()

        class TestStrategy(BaseStrategy):
            metadata = StrategyMetadata(
                name="test:strategy",
                target="test",
                strategy="strategy",
                description="Test",
                aliases=["test"],
            )

            def run(self, atoms_list: list[Atoms], **kwargs):
                return {"optimized_atoms": atoms_list[0]}

        registry.register(TestStrategy)

        retrieved = registry.get("test:strategy")
        assert retrieved == TestStrategy

    def test_get_strategy_by_alias(self) -> None:
        """Test getting strategy by alias."""
        registry = StrategyRegistry()

        class TestStrategy(BaseStrategy):
            metadata = StrategyMetadata(
                name="test:strategy",
                target="test",
                strategy="strategy",
                description="Test",
                aliases=["test"],
            )

            def run(self, atoms_list: list[Atoms], **kwargs):
                return {"optimized_atoms": atoms_list[0]}

        registry.register(TestStrategy)

        retrieved = registry.get("test")
        assert retrieved == TestStrategy

    def test_get_strategy_not_found(self) -> None:
        """Test getting non-existent strategy raises KeyError."""
        registry = StrategyRegistry()

        with pytest.raises(KeyError, match="No strategy found"):
            registry.get("nonexistent")

    def test_get_by_target_strategy(self) -> None:
        """Test getting strategy by target and strategy components."""
        registry = StrategyRegistry()

        class TestStrategy(BaseStrategy):
            metadata = StrategyMetadata(
                name="test:strategy",
                target="test",
                strategy="strategy",
                description="Test",
                aliases=[],
            )

            def run(self, atoms_list: list[Atoms], **kwargs):
                return {"optimized_atoms": atoms_list[0]}

        registry.register(TestStrategy)

        retrieved = registry.get_by_target_strategy("test", "strategy")
        assert retrieved == TestStrategy

    def test_get_by_target_strategy_not_found(self) -> None:
        """Test getting strategy by target/strategy that doesn't exist."""
        registry = StrategyRegistry()

        with pytest.raises(KeyError):
            registry.get_by_target_strategy("nonexistent", "strategy")

    def test_has_strategy(self) -> None:
        """Test checking if strategy exists."""
        registry = StrategyRegistry()

        class TestStrategy(BaseStrategy):
            metadata = StrategyMetadata(
                name="test:strategy",
                target="test",
                strategy="strategy",
                description="Test",
                aliases=["test"],
            )

            def run(self, atoms_list: list[Atoms], **kwargs):
                return {"optimized_atoms": atoms_list[0]}

        registry.register(TestStrategy)

        assert registry.has_strategy("test:strategy") is True
        assert registry.has_strategy("test") is True
        assert registry.has_strategy("nonexistent") is False

    def test_list_strategies(self) -> None:
        """Test listing all registered strategies."""
        registry = StrategyRegistry()

        class Strategy1(BaseStrategy):
            metadata = StrategyMetadata(
                name="test:strategy1",
                target="test",
                strategy="strategy1",
                description="Test 1",
                aliases=["test1"],
            )

            def run(self, atoms_list: list[Atoms], **kwargs):
                return {"optimized_atoms": atoms_list[0]}

        class Strategy2(BaseStrategy):
            metadata = StrategyMetadata(
                name="test:strategy2",
                target="test",
                strategy="strategy2",
                description="Test 2",
                aliases=["test2"],
            )

            def run(self, atoms_list: list[Atoms], **kwargs):
                return {"optimized_atoms": atoms_list[0]}

        registry.register(Strategy1)
        registry.register(Strategy2)

        strategies = registry.list_strategies()

        assert isinstance(strategies, dict)
        assert "test:strategy1" in strategies
        assert "test:strategy2" in strategies
        # list_strategies returns StrategyMetadata objects directly
        assert strategies["test:strategy1"].name == "test:strategy1"
        assert strategies["test:strategy2"].name == "test:strategy2"

    def test_list_strategies_no_duplicates(self) -> None:
        """Test that list_strategies doesn't include duplicate entries from aliases."""
        registry = StrategyRegistry()

        class TestStrategy(BaseStrategy):
            metadata = StrategyMetadata(
                name="test:strategy",
                target="test",
                strategy="strategy",
                description="Test",
                aliases=["test", "alias1", "alias2"],
            )

            def run(self, atoms_list: list[Atoms], **kwargs):
                return {"optimized_atoms": atoms_list[0]}

        registry.register(TestStrategy)

        strategies = registry.list_strategies()

        # Should only have one entry (the main name, not aliases)
        assert len(strategies) == 1
        assert "test:strategy" in strategies
        assert "test" not in strategies  # Aliases shouldn't appear


class TestGlobalRegistry:
    """Test the global REGISTRY instance."""

    def test_global_registry_exists(self) -> None:
        """Test that global REGISTRY instance exists."""
        assert REGISTRY is not None
        assert isinstance(REGISTRY, StrategyRegistry)

    def test_global_registry_has_registered_strategies(self) -> None:
        """Test that global registry has strategies after imports."""
        # Import strategies to trigger registration
        import qme.strategies  # noqa: F401

        strategies = REGISTRY.list_strategies()

        # Should have at least some strategies registered
        assert len(strategies) > 0
        assert isinstance(strategies, dict)

        # Check that common strategies are registered
        strategy_names = list(strategies.keys())
        # At least one strategy should have "minima" or "path" or "ts" in name
        assert any("minima" in name or "path" in name or "ts" in name for name in strategy_names)
