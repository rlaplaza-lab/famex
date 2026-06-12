"""Strategy registry for FAMEX Explorer.

This module provides the StrategyRegistry class for managing strategy classes
and their registration.
"""

from __future__ import annotations

from famex.core.base_strategy import BaseStrategy, StrategyMetadata
from famex.core.exceptions import StrategyNotFoundError
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)


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

        Raises
        ------
        ValueError
            If strategy class is missing metadata

        """
        if not hasattr(strategy_class, "metadata"):
            msg = f"Strategy class {strategy_class.__name__} missing metadata"
            logger.error(msg)
            raise ValueError(msg)

        meta = strategy_class.metadata
        logger.debug("Registering strategy: %s (aliases: %s)", meta.name, meta.aliases)
        self._strategies[meta.name] = strategy_class

        # Auto-register aliases
        for alias in meta.aliases:
            if alias in self._strategies:
                # Fail hard on duplicate aliases
                msg = f"Alias '{alias}' already registered"
                logger.error("%s - Existing strategy: %s", msg, self._strategies[alias].__name__)
                raise ValueError(msg)
            self._strategies[alias] = strategy_class

    def get(self, strategy_name: str) -> type[BaseStrategy]:
        """Get strategy by name or alias.

        Parameters
        ----------
        strategy_name : str
            Strategy name (e.g., "minima:local") or alias (e.g., "neb")

        Returns
        -------
        type[BaseStrategy]
            Strategy class

        Raises
        ------
        StrategyNotFoundError
            If strategy not found

        """
        if strategy_name not in self._strategies:
            available = sorted(
                {name for name, cls in self._strategies.items() if name == cls.metadata.name}
            )
            logger.error("Strategy '%s' not found. Available: %s", strategy_name, available)
            raise StrategyNotFoundError(strategy_name, available)
        logger.debug(
            "Retrieved strategy '%s': %s",
            strategy_name,
            self._strategies[strategy_name].__name__,
        )
        return self._strategies[strategy_name]

    def list_strategies(
        self,
        target: str | None = None,
        strategy: str | None = None,
    ) -> dict[str, StrategyMetadata]:
        """List all registered strategies, optionally filtered.

        Parameters
        ----------
        target : str, optional
            Filter by target type ("minima", "ts", "path")
        strategy : str, optional
            Filter by strategy type ("local", "neb", "cineb", etc.)

        Returns
        -------
        dict[str, StrategyMetadata]
            Dictionary mapping strategy names to metadata

        """
        seen = set()
        result = {}
        for cls in self._strategies.values():
            meta = cls.metadata
            if meta.name not in seen:
                # Apply filters if specified
                if target is not None and meta.target != target:
                    continue
                if strategy is not None and meta.strategy != strategy:
                    continue
                result[meta.name] = meta
                seen.add(meta.name)
        return result

    def get_by_target_strategy(self, target: str, strategy: str) -> type[BaseStrategy]:
        """Get strategy by target and strategy components.

        Parameters
        ----------
        target : str
            Target type ("minima", "ts", "path")
        strategy : str
            Strategy type ("local", "neb", "cineb", etc.)

        Returns
        -------
        type[BaseStrategy]
            Strategy class

        Raises
        ------
        StrategyNotFoundError
            If strategy not found

        """
        full_name = f"{target}:{strategy}"
        return self.get(full_name)

    def get_by_metadata(
        self,
        target: str | None = None,
        strategy: str | None = None,
        requires_multiple: bool | None = None,
    ) -> list[type[BaseStrategy]]:
        """Get strategies matching metadata criteria.

        Parameters
        ----------
        target : str, optional
            Filter by target type
        strategy : str, optional
            Filter by strategy type
        requires_multiple : bool, optional
            Filter by whether multiple structures are required

        Returns
        -------
        list[type[BaseStrategy]]
            List of matching strategy classes

        """
        matches = []
        seen = set()
        for cls in self._strategies.values():
            meta = cls.metadata
            if meta.name in seen:
                continue
            seen.add(meta.name)

            # Apply filters
            if target is not None and meta.target != target:
                continue
            if strategy is not None and meta.strategy != strategy:
                continue
            if (
                requires_multiple is not None
                and meta.requires_multiple_structures != requires_multiple
            ):
                continue

            matches.append(cls)
        return matches

    def has_strategy(self, strategy_name: str) -> bool:
        """Check if strategy is registered.

        Parameters
        ----------
        strategy_name : str
            Strategy name or alias

        Returns
        -------
        bool
            True if strategy is registered

        """
        return strategy_name in self._strategies

    def unregister(self, strategy_name: str) -> None:
        """Unregister a strategy by name or alias.

        This is useful for testing or plugin systems that need to replace
        or remove strategies dynamically.

        Parameters
        ----------
        strategy_name : str
            Strategy name or alias to unregister

        Raises
        ------
        StrategyNotFoundError
            If strategy not found

        Notes
        -----
        This removes both the main strategy name and all its aliases from
        the registry. The strategy class itself is not modified.

        """
        if strategy_name not in self._strategies:
            available = sorted(
                {name for name, cls in self._strategies.items() if name == cls.metadata.name}
            )
            raise StrategyNotFoundError(strategy_name, available)

        # Get the strategy class to find all its aliases
        strategy_class = self._strategies[strategy_name]
        meta = strategy_class.metadata

        # Remove main name and all aliases
        names_to_remove = [meta.name] + meta.aliases
        for name in names_to_remove:
            if name in self._strategies:
                del self._strategies[name]
                logger.debug("Unregistered strategy: %s", name)


# Global registry instance
REGISTRY = StrategyRegistry()
