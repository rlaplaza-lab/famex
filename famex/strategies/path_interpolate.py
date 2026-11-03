"""Generate interpolated path only (no optimization)."""

from __future__ import annotations

from typing import Any

from ase import Atoms

from famex.core.base_strategy import BaseStrategy, StrategyMetadata
from famex.core.registry import REGISTRY
from famex.io.path_manager import PathManager


class PathInterpolateStrategy(BaseStrategy):
    """Generate interpolated path only (no optimization)."""

    metadata = StrategyMetadata(
        name="path:interpolate",
        target="path",
        strategy="interpolate",
        description="Generate interpolated path only (no optimization)",
        aliases=["interpolate"],
        requires_multiple_structures=True,
    )

    def run(
        self,
        atoms_list: list[Atoms],
        npoints: int = 11,
        method: str = "geodesic",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate interpolated path using PathManager without optimization."""
        self.validate_inputs(atoms_list)

        # Use PathManager to interpolate; do not optimize path
        path_mgr = PathManager(atoms_list)
        path = path_mgr.interpolate(npoints=npoints, method=method)

        # No calculator needed for pure interpolation; do not attach by default.

        return self.prepare_result(
            optimized_atoms=path,
            converged=True,
            trajectory=path,
        )


# Register the strategy
REGISTRY.register(PathInterpolateStrategy)
