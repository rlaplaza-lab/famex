"""Generate interpolated path only (no optimization)."""

from typing import Any

from ase import Atoms

from qme.core.base_strategy import BaseStrategy, StrategyMetadata
from qme.core.registry import REGISTRY
from qme.io.path_manager import PathManager
from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)


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
        # Only forward kwargs accepted by PathManager.interpolate
        allowed_keys = {"rmsd_threshold", "energy_threshold", "calculator"}
        interpolate_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
        path = path_mgr.interpolate(
            npoints=npoints,
            method=method,
            optimize_path=False,
            explorer=self.explorer,
            **interpolate_kwargs,
        )

        # No calculator needed for pure interpolation; do not attach by default.

        return self.prepare_result(
            optimized_atoms=path,
            converged=True,
            trajectory=path,
        )


# Register the strategy
REGISTRY.register(PathInterpolateStrategy)
