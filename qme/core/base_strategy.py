"""Base strategy classes for QME Explorer.

This module provides the foundation for the strategy system, including:
- BaseStrategy: Abstract base class for all optimization strategies
- StrategyMetadata: Dataclass for strategy metadata
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ase import Atoms


@dataclass
class StrategyMetadata:
    """Metadata for strategy registration.

    Attributes:
    ----------
    name : str
        Full strategy name, e.g., "minima:local"
    target : str
        Target type: "minima", "ts", "path"
    strategy : str
        Strategy type: "local", "neb", "cineb", etc.
    description : str
        Human-readable description
    aliases : list[str]
        Alternative names for the strategy
    requires_multiple_structures : bool
        Whether strategy needs 2+ structures

    """

    name: str
    target: str
    strategy: str
    description: str
    aliases: list[str]
    requires_multiple_structures: bool = False


class BaseStrategy(ABC):
    """Base class for all optimization strategies.

    All strategies must inherit from this class and implement the run() method.
    The metadata attribute defines the strategy's registration information.
    """

    metadata: StrategyMetadata

    def __init__(self, explorer: Any, profiler: Any | None = None) -> None:
        """Initialize strategy with explorer instance.

        Parameters
        ----------
        explorer : Any
            Explorer instance for calculator and constraint management
        profiler : Any, optional
            Performance profiler instance for tracking execution metrics

        """
        self.explorer = explorer
        self.profiler = profiler

    @abstractmethod
    def run(
        self,
        atoms_list: list[Atoms],
        **kwargs: Any,
    ) -> dict[str, Atoms | list[Atoms] | bool | int | float | str]:
        """Execute strategy. Returns standardized result dict.

        Parameters
        ----------
        atoms_list : list[Atoms]
            List of structures to optimize
        **kwargs : Any
            Strategy-specific parameters

        Returns:
        -------
        dict[str, Union[Atoms, list[Atoms], bool, int, float, str]]
            Standardized result dictionary with at least:
            - optimized_atoms: Optimized structure(s) (Atoms or list[Atoms])
            - strategy: Strategy name (str)
            - converged: Whether optimization converged (bool)
            - Additional strategy-specific metadata

        """

    def validate_inputs(self, atoms_list: list[Atoms]) -> None:
        """Validate inputs before running.

        Parameters
        ----------
        atoms_list : list[Atoms]
            List of structures to validate

        Raises:
        ------
        ValueError
            If inputs are invalid for this strategy

        """
        if not atoms_list:
            msg = "No atoms provided"
            raise ValueError(msg)

    def _merge_profiler_results(self, result: dict[str, Any]) -> dict[str, Any]:
        """Merge profiler results into strategy result if profiler is available.

        Parameters
        ----------
        result : dict[str, Any]
            Strategy result dictionary

        Returns:
        -------
        dict[str, Any]
            Result dictionary with profiler data merged in

        """
        if self.profiler is not None:
            result["performance"] = self.profiler.get_summary()
        return result

    def prepare_result(
        self,
        optimized_atoms: Atoms | list[Atoms],
        **metadata: Any,
    ) -> dict[str, Atoms | list[Atoms] | bool | int | float | str]:
        """Standardize result format.

        Parameters
        ----------
        optimized_atoms : Atoms or list[Atoms]
            Optimized structure(s) from strategy
        **metadata : Any
            Additional metadata to include in result

        Returns:
        -------
        dict[str, Union[Atoms, list[Atoms], bool, int, float, str]]
            Standardized result dictionary

        """
        result: dict[str, Atoms | list[Atoms] | bool | int | float | str] = {
            "optimized_atoms": optimized_atoms,
            "strategy": self.metadata.name,
        }
        # Convert metadata values to the expected types
        for key, value in metadata.items():
            if isinstance(value, (Atoms, list, bool, int, float, str)):
                result[key] = value
            else:
                # Convert other types to string for compatibility
                result[key] = str(value)
        return result
