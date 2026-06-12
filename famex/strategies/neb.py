"""NEB (Nudged Elastic Band) and CI-NEB (Climbing Image NEB) path optimization strategies."""

from __future__ import annotations

import warnings
from typing import Any

from ase import Atoms

from famex.core.base_strategy import BaseStrategy, StrategyMetadata
from famex.core.registry import REGISTRY
from famex.io.path_manager import PathManager
from famex.strategies.helpers import filter_interpolation_kwargs
from famex.strategies.neb_optimizer import NEBOptimizer
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)


class MultiStructureNEBStrategy(BaseStrategy):
    """Unified NEB strategy supporting both regular NEB and CI-NEB.

    This class handles both NEB and CI-NEB optimizations. The behavior is determined
    by the `climb` parameter and the strategy metadata used for registration.
    """

    metadata = StrategyMetadata(
        name="path:neb",
        target="path",
        strategy="neb",
        description="NEB path optimization with geodesic interpolation",
        aliases=["neb"],
        requires_multiple_structures=True,
    )

    def run(
        self,
        atoms_list: list[Atoms],
        npoints: int = 11,
        method: str = "geodesic",
        fmax: float = 0.05,
        steps: int = 1000,
        spring_constant: float = 5.0,
        climb: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Atoms | list[Atoms] | bool | int | float | str]:
        """Run NEB or CI-NEB path optimization.

        Parameters
        ----------
        atoms_list : list[Atoms]
            List of structures defining the path endpoints
        npoints : int, default=11
            Number of images in the NEB path
        method : str, default="geodesic"
            Interpolation method for initial path generation
        fmax : float, default=0.05
            Force convergence threshold
        steps : int, default=1000
            Maximum optimization steps
        spring_constant : float, default=5.0
            Spring constant for NEB spring forces
        climb : bool, optional
            Whether to enable climbing image behavior. If None, uses default based on strategy:
            False for regular NEB, True for CI-NEB
        **kwargs : Any
            Additional keyword arguments

        Returns
        -------
        dict[str, Atoms | list[Atoms] | bool | int | float | str]
            Standardized result dictionary containing:
            - optimized_atoms: NEB path structures (list[Atoms])
            - strategy: Strategy name (str)
            - converged: Whether NEB optimization converged (bool)
            - steps_taken: Number of optimization steps (int)
            - npoints: Number of images in the path (int)
            - method: Interpolation method used (str)
            - climb: Whether climbing image was used (bool)

        """
        self.validate_inputs(atoms_list)

        if climb is None:
            climb = self.metadata.name == "path:cineb"

        method_name = "CI-NEB" if climb else "NEB"

        path_mgr = PathManager(atoms_list)
        interpolate_kwargs = filter_interpolation_kwargs(kwargs, allowed_keys={"calculator"})
        path = path_mgr.interpolate(
            npoints=npoints,
            method=method,
            optimize_path=False,
            explorer=self.explorer,
            **interpolate_kwargs,
        )

        if path and hasattr(path[0], "__iter__") and not isinstance(path[0], Atoms):
            try:  # type: ignore[unreachable]
                flat = []
                for seg in path:
                    if isinstance(seg, (list, tuple)):  # noqa: UP038
                        flat.extend(seg)
                    else:
                        flat.append(seg)
                path = flat
            except (TypeError, AttributeError):
                pass

        if len(path) < 3:
            logger.error(
                "%s requires at least 3 images (npoints >= 3), got %d",
                method_name,
                len(path),
            )
            msg = f"{method_name} requires at least 3 images (npoints >= 3)"
            raise ValueError(msg)

        if self.explorer is not None:
            PathManager.attach_calculators(self.explorer, path)
            if any(getattr(img, "calc", None) is None for img in path):
                logger.error(
                    f"Failed to attach calculators to {method_name} images. "
                    "Check backend/model availability."
                )
                raise RuntimeError(
                    f"Failed to attach calculators to {method_name} images. "
                    "Check backend/model availability.",
                )

        neb_opt = NEBOptimizer(
            images=path,
            spring_constant=spring_constant,
            climb=climb,
            fmax=fmax,
            steps=steps,
            **kwargs,
        )

        optimized_path = neb_opt.optimize()

        if optimized_path:
            input_atoms = list(atoms_list)

            filtered_path, _removed_indices, warnings_list = (
                PathManager.filter_redundant_structures(
                    optimized_path,
                    input_structures=input_atoms,
                    rmsd_threshold=kwargs.get("rmsd_threshold", 0.1),
                    energy_threshold=kwargs.get("energy_threshold", 0.001),
                    strategy_name=self.metadata.name,
                )
            )

            for warning_msg in warnings_list:
                warnings.warn(warning_msg, stacklevel=2)

            optimized_path = filtered_path

        return self.prepare_result(
            optimized_path,
            converged=True,
            trajectory=optimized_path,
            climb=climb,
        )


class MultiStructureCINEBStrategy(MultiStructureNEBStrategy):
    """CI-NEB (Climbing Image Nudged Elastic Band) path optimization strategy.

    This is a thin wrapper around MultiStructureNEBStrategy that registers CI-NEB
    with appropriate metadata. The actual implementation is shared with NEB.
    """

    metadata = StrategyMetadata(
        name="path:cineb",
        target="path",
        strategy="cineb",
        description="Climbing Image NEB (CI-NEB) optimization with geodesic interpolation",
        aliases=["cineb"],
        requires_multiple_structures=True,
    )


REGISTRY.register(MultiStructureNEBStrategy)
REGISTRY.register(MultiStructureCINEBStrategy)
