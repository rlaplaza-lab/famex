"""Strategy registry for QME Explorer.

This module provides the StrategyRegistry class for managing strategy classes
and their registration.
"""

from __future__ import annotations

from qme.core.base_strategy import BaseStrategy, StrategyMetadata


class StrategyRegistry:
    """Registry for strategy classes.

    Manages registration and lookup of strategy classes by name or alias.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._strategies: dict[str, type[BaseStrategy]] = {}

    def register(self, strategy_class: type[BaseStrategy]) -> None:
        """Register strategy by class.

        Parameters
        ----------
        strategy_class : type[BaseStrategy]
            Strategy class to register

        Raises:
        ------
        ValueError
            If strategy class is missing metadata

        """
        if not hasattr(strategy_class, "metadata"):
            msg = f"Strategy class {strategy_class.__name__} missing metadata"
            raise ValueError(msg)

        meta = strategy_class.metadata
        self._strategies[meta.name] = strategy_class

        # Auto-register aliases
        for alias in meta.aliases:
            if alias in self._strategies:
                # Fail hard on duplicate aliases
                msg = f"Alias '{alias}' already registered"
                raise ValueError(msg)
            self._strategies[alias] = strategy_class

    def get(self, strategy_name: str) -> type[BaseStrategy]:
        """Get strategy by name or alias.

        Parameters
        ----------
        strategy_name : str
            Strategy name (e.g., "minima:local") or alias (e.g., "neb")

        Returns:
        -------
        type[BaseStrategy]
            Strategy class

        Raises:
        ------
        KeyError
            If strategy not found

        """
        if strategy_name not in self._strategies:
            available = sorted(self._strategies.keys())
            msg = f"No strategy found for '{strategy_name}'. Available strategies: {available}"
            raise KeyError(
                msg,
            )
        return self._strategies[strategy_name]

    def list_strategies(self) -> dict[str, StrategyMetadata]:
        """List all registered strategies.

        Returns:
        -------
        dict[str, StrategyMetadata]
            Dictionary mapping strategy names to metadata

        """
        seen = set()
        result = {}
        for cls in self._strategies.values():
            if cls.metadata.name not in seen:
                result[cls.metadata.name] = cls.metadata
                seen.add(cls.metadata.name)
        return result

    def get_by_target_strategy(self, target: str, strategy: str) -> type[BaseStrategy]:
        """Get strategy by target and strategy components.

        Parameters
        ----------
        target : str
            Target type ("minima", "ts", "path")
        strategy : str
            Strategy type ("local", "neb", "cineb", etc.)

        Returns:
        -------
        type[BaseStrategy]
            Strategy class

        Raises:
        ------
        KeyError
            If strategy not found

        """
        full_name = f"{target}:{strategy}"
        return self.get(full_name)

    def has_strategy(self, strategy_name: str) -> bool:
        """Check if strategy is registered.

        Parameters
        ----------
        strategy_name : str
            Strategy name or alias

        Returns:
        -------
        bool
            True if strategy is registered

        """
        return strategy_name in self._strategies


# Global registry instance
REGISTRY = StrategyRegistry()
