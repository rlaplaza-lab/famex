"""Symmetry handling for thermochemistry and force/Hessian projection.

When neither a symmetry number nor a point group is supplied, libmolsymm
(``molsymm``) detects σ from the geometry. With libmolsymm 2.0+, the same
detection drives optional force and Hessian symmetrization.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import molsymm
import numpy as np
from ase import Atoms
from numpy.typing import NDArray

from famex.utils.logging import get_famex_logger

__all__ = [
    "SymmetryHandler",
    "MolecularSymmetryPolicy",
    "MolecularSymmetry",
    "get_point_group_symmetry_number",
    "detect_symmetry_from_atoms",
    "endpoints_share_point_group_axis",
    "is_path_workflow",
    "molecular_symmetry_from_calculator",
    "project_assembled_hessian",
]

logger = get_famex_logger(__name__)

# Point group to symmetry number mapping from GoodVibes
POINT_GROUP_SYMMETRY_NUMBERS = {
    "C1": 1,
    "Cs": 1,
    "Ci": 1,
    "C2": 2,
    "C3": 3,
    "C4": 4,
    "C5": 5,
    "C6": 6,
    "C7": 7,
    "C8": 8,
    "D2": 4,
    "D3": 6,
    "D4": 8,
    "D5": 10,
    "D6": 12,
    "D7": 14,
    "D8": 16,
    "C2v": 2,
    "C3v": 3,
    "C4v": 4,
    "C5v": 5,
    "C6v": 6,
    "C7v": 7,
    "C8v": 8,
    "C2h": 2,
    "C3h": 3,
    "C4h": 4,
    "C5h": 5,
    "C6h": 6,
    "C7h": 7,
    "C8h": 8,
    "D2h": 4,
    "D3h": 6,
    "D4h": 8,
    "D5h": 10,
    "D6h": 12,
    "D7h": 14,
    "D8h": 16,
    "D2d": 4,
    "D3d": 6,
    "D4d": 8,
    "D5d": 10,
    "D6d": 12,
    "D7d": 14,
    "D8d": 16,
    "S4": 4,
    "S6": 6,
    "S8": 8,
    "T": 6,
    "Th": 12,
    "Td": 12,
    "O": 12,
    "Oh": 24,
    "Cinfv": 1,
    "Dinfh": 2,
    "I": 30,
    "Ih": 60,
    "Kh": 1,
}

_AXIS_DOT_TOL = 1e-2
_NO_PRINCIPAL_AXIS = frozenset({"C1", "Cs", "Ci", "Cinfv", "Dinfh", "Kh"})
_PATH_STRATEGIES = frozenset({"neb", "cineb", "growing_string", "gsm", "dhs", "irc", "interpolate"})


def _warn_assumption(message: str, exc: BaseException | None = None) -> None:
    if exc is not None:
        message = f"{message} ({exc})"
    warnings.warn(message, UserWarning, stacklevel=3)


def get_point_group_symmetry_number(point_group: str) -> int:
    """Get symmetry number for a given point group."""
    point_group = point_group.strip()
    if point_group not in POINT_GROUP_SYMMETRY_NUMBERS:
        msg = (
            f"Unknown point group: {point_group}. "
            f"Known point groups: {list(POINT_GROUP_SYMMETRY_NUMBERS.keys())}"
        )
        raise ValueError(msg)
    return POINT_GROUP_SYMMETRY_NUMBERS[point_group]


def _atomic_numbers(atoms: Atoms) -> list[int]:
    return [int(z) for z in atoms.get_atomic_numbers()]


def _positions(atoms: Atoms) -> NDArray[np.float64]:
    return np.asarray(atoms.get_positions(), dtype=np.float64)


def detect_symmetry_from_atoms(
    atoms: Atoms,
    *,
    primary: float = 1e-3,
) -> tuple[int, str]:
    """Return ``(symmetry_number, point_group)`` via molsymm.

    A failed detection returns ``(1, "C1")``.
    """
    try:
        analysis = molsymm.analyze(
            _atomic_numbers(atoms),
            _positions(atoms),
            primary=primary,
        )
        point_group = str(analysis.point_group)
        symmetry_number = int(analysis.symmetry_number)
        if symmetry_number <= 0:
            return 1, "C1"
        return symmetry_number, point_group
    except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
        _warn_assumption("molsymm symmetry detection failed; assuming C1 symmetry", exc)
        return 1, "C1"


def _kabsch_rotation(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return rotation R such that ``target ≈ source @ R`` (row vectors)."""
    p = np.asarray(source, dtype=np.float64)
    q = np.asarray(target, dtype=np.float64)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.linalg.det(vt.T @ u.T)
    sign = np.array([1.0, 1.0, 1.0 if d >= 0.0 else -1.0], dtype=np.float64)
    return np.asarray(vt.T @ np.diag(sign) @ u.T, dtype=np.float64)


def _principal_axis_lab(atoms: Atoms, analysis: Any) -> NDArray[np.float64] | None:
    """Lab-frame Cotton principal axis (+z in analysis frame)."""
    if str(analysis.point_group) in _NO_PRINCIPAL_AXIS:
        return None

    centered = _positions(atoms)
    centered = centered - centered.mean(axis=0)
    frame = np.asarray(analysis.frame_xyz, dtype=np.float64)
    if frame.shape != centered.shape:
        return None

    rotation = _kabsch_rotation(frame, centered)
    axis = rotation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return None
    return np.asarray(axis / norm, dtype=np.float64)


def endpoints_share_point_group_axis(
    atoms_a: Atoms,
    atoms_b: Atoms,
    *,
    primary: float = 1e-3,
    axis_tol: float = _AXIS_DOT_TOL,
) -> bool:
    """Return True when endpoints share the same non-C1 PG and principal axis."""
    try:
        analysis_a = molsymm.analyze(_atomic_numbers(atoms_a), _positions(atoms_a), primary=primary)
        analysis_b = molsymm.analyze(_atomic_numbers(atoms_b), _positions(atoms_b), primary=primary)
    except (ValueError, TypeError, RuntimeError, AttributeError):
        return False

    pg = str(analysis_a.point_group)
    if pg == "C1" or pg != str(analysis_b.point_group):
        return False

    axis_a = _principal_axis_lab(atoms_a, analysis_a)
    axis_b = _principal_axis_lab(atoms_b, analysis_b)
    if axis_a is None or axis_b is None:
        return False
    return abs(float(np.dot(axis_a, axis_b))) >= (1.0 - axis_tol)


@dataclass
class MolecularSymmetryPolicy:
    """User-facing knobs for molecular point-group enforcement."""

    enabled: bool = True
    point_group: str | None = None
    primary: float = 1e-3


class MolecularSymmetry:
    """Resolve a point group and project forces / Hessians with libmolsymm 2.0."""

    def __init__(
        self,
        policy: MolecularSymmetryPolicy | None = None,
        *,
        atoms_list: list[Atoms] | None = None,
        is_path: bool = False,
    ) -> None:
        self.policy = policy or MolecularSymmetryPolicy()
        self.active = False
        self.resolved_point_group = "C1"
        self.symmetry_number = 1
        self._projection_warned = False

        if not self.policy.enabled:
            return

        structures = atoms_list or []
        if not structures:
            return

        user_pg = self.policy.point_group
        if user_pg is not None:
            user_pg = user_pg.strip()
            for atoms in structures:
                self.apply_user_geometry(atoms, user_pg)
            self.resolved_point_group = user_pg
            if user_pg in POINT_GROUP_SYMMETRY_NUMBERS:
                self.symmetry_number = POINT_GROUP_SYMMETRY_NUMBERS[user_pg]
            else:
                self.symmetry_number, _ = detect_symmetry_from_atoms(
                    structures[0], primary=self.policy.primary
                )
        else:
            self.symmetry_number, self.resolved_point_group = detect_symmetry_from_atoms(
                structures[0], primary=self.policy.primary
            )

        if is_path:
            if len(structures) < 2 or not endpoints_share_point_group_axis(
                structures[0], structures[-1], primary=self.policy.primary
            ):
                logger.info(
                    "Path symmetry inactive: endpoints do not share the same "
                    "point group and principal axis (resolved=%s)",
                    self.resolved_point_group,
                )
                return

        if self.resolved_point_group == "C1":
            return

        self.active = True
        logger.info(
            "Point-group symmetry active: %s (σ=%d)",
            self.resolved_point_group,
            self.symmetry_number,
        )

    @property
    def should_project(self) -> bool:
        return self.policy.enabled and self.active and self.resolved_point_group != "C1"

    def apply_user_geometry(self, atoms: Atoms, point_group: str | None = None) -> None:
        """Project ``atoms`` onto ``point_group`` in place, or raise ValueError."""
        pg = (point_group or self.policy.point_group or "").strip()
        if not pg:
            raise ValueError("A point group must be supplied to project the geometry.")

        primary = self.policy.primary
        try:
            out = molsymm.symmetrize(
                _atomic_numbers(atoms),
                _positions(atoms),
                pg,
                primary=primary,
            )
        except RuntimeError as exc:
            raise ValueError(
                f"Geometry is not compatible with point group '{pg}' "
                f"within tolerance {primary:g} Å."
            ) from exc

        atoms.set_positions(np.asarray(out.positions, dtype=np.float64))
        self.resolved_point_group = pg

    def project_forces(self, atoms: Atoms, forces: NDArray[np.floating]) -> NDArray[np.float64]:
        """Project Cartesian forces onto the resolved point group."""
        forces_arr = np.asarray(forces, dtype=np.float64)
        if not self.should_project:
            return forces_arr
        out = self._symmetrize(atoms, forces=forces_arr, hessian=None)
        if out is None or out.forces is None:
            return forces_arr
        projected = np.asarray(out.forces, dtype=np.float64)
        return projected.reshape(-1, 3) if projected.ndim == 1 else projected

    def project_hessian(self, atoms: Atoms, hessian: NDArray[np.floating]) -> NDArray[np.float64]:
        """Project a Cartesian Hessian onto the resolved point group."""
        hessian_arr = np.asarray(hessian, dtype=np.float64)
        if not self.should_project:
            return hessian_arr
        out = self._symmetrize(atoms, forces=None, hessian=hessian_arr)
        if out is None or out.hessian is None:
            return hessian_arr
        return np.asarray(out.hessian, dtype=np.float64)

    def _symmetrize(
        self,
        atoms: Atoms,
        *,
        forces: NDArray[np.float64] | None,
        hessian: NDArray[np.float64] | None,
    ) -> Any | None:
        try:
            return molsymm.symmetrize(
                _atomic_numbers(atoms),
                _positions(atoms),
                self.resolved_point_group,
                forces=forces,
                hessian=hessian,
                primary=self.policy.primary,
            )
        except RuntimeError as exc:
            if not self._projection_warned:
                logger.warning(
                    "Point-group projection with '%s' failed mid-run; "
                    "using unsymmetrized derivatives (%s)",
                    self.resolved_point_group,
                    exc,
                )
                self._projection_warned = True
            return None


def is_path_workflow(*, target: str | None, strategy: str | None) -> bool:
    """Return True for multi-image path strategies / targets."""
    if (target or "").strip().lower() == "path":
        return True
    return (strategy or "").strip().lower() in _PATH_STRATEGIES


def molecular_symmetry_from_calculator(calculator: Any | None) -> MolecularSymmetry | None:
    """Return a ``MolecularSymmetry`` attached to ``calculator`` or nested wrappers."""
    seen: set[int] = set()
    current = calculator
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        symmetry = getattr(current, "molecular_symmetry", None)
        if isinstance(symmetry, MolecularSymmetry):
            return symmetry
        current = getattr(current, "calculator", None)
    return None


def project_assembled_hessian(
    atoms: Atoms,
    hessian: NDArray[np.floating],
    calculator: Any | None = None,
    molecular_symmetry: MolecularSymmetry | None = None,
) -> NDArray[np.float64]:
    """Project an assembled Hessian when an active MolecularSymmetry is available."""
    hessian_arr = np.asarray(hessian, dtype=np.float64)
    symmetry = molecular_symmetry or molecular_symmetry_from_calculator(calculator)
    if symmetry is None or not symmetry.should_project:
        return hessian_arr
    return symmetry.project_hessian(atoms, hessian_arr)


class SymmetryHandler:
    """Handles symmetry corrections for thermodynamic calculations."""

    def __init__(
        self,
        symmetry_number: int | None = None,
        point_group: str | None = None,
        atoms: Atoms | None = None,
        warn_on_assumptions: bool = True,
    ):
        """Resolve σ: explicit number, then point group, then molsymm, then C1."""
        self.warn_on_assumptions = warn_on_assumptions
        self._symmetry_number = 1
        self._point_group = "C1"

        if symmetry_number is not None:
            self._symmetry_number = symmetry_number
            self._point_group = point_group if point_group is not None else "C1"
        elif point_group is not None:
            try:
                self._symmetry_number = get_point_group_symmetry_number(point_group)
                self._point_group = point_group
            except ValueError as exc:
                if warn_on_assumptions:
                    _warn_assumption(
                        f"Could not determine symmetry number from point group "
                        f"'{point_group}'. Using C1 symmetry (symmetry_number=1).",
                        exc,
                    )
        elif atoms is not None:
            self._symmetry_number, self._point_group = detect_symmetry_from_atoms(atoms)
        elif warn_on_assumptions:
            _warn_assumption(
                "No symmetry number, point group, or atoms provided. "
                "Assuming C1 symmetry (symmetry_number=1)."
            )

    @property
    def symmetry_number(self) -> int:
        return self._symmetry_number

    @property
    def point_group(self) -> str:
        return self._point_group

    def __repr__(self) -> str:
        return (
            f"SymmetryHandler(symmetry_number={self.symmetry_number}, "
            f"point_group='{self.point_group}')"
        )
