"""Symmetry handling for thermochemistry calculations.

This module provides symmetry number handling for entropy corrections in
statistical thermodynamics calculations.
"""

from __future__ import annotations

import warnings

__all__ = ["SymmetryHandler", "get_point_group_symmetry_number"]


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


def get_point_group_symmetry_number(point_group: str) -> int:
    """Get symmetry number for a given point group.

    Parameters
    ----------
    point_group : str
        Point group symbol (e.g., 'C1', 'C2v', 'Td', etc.)

    Returns
    -------
    int
        Symmetry number for the point group

    Raises
    ------
    ValueError
        If point group is not recognized
    """
    point_group = point_group.strip()
    if point_group not in POINT_GROUP_SYMMETRY_NUMBERS:
        msg = (
            f"Unknown point group: {point_group}. "
            f"Known point groups: {list(POINT_GROUP_SYMMETRY_NUMBERS.keys())}"
        )
        raise ValueError(msg)
    return POINT_GROUP_SYMMETRY_NUMBERS[point_group]


class SymmetryHandler:
    """Handles symmetry corrections for thermodynamic calculations."""

    def __init__(
        self,
        symmetry_number: int | None = None,
        point_group: str | None = None,
        warn_on_assumptions: bool = True,
    ):
        """Initialize symmetry handler.

        Parameters
        ----------
        symmetry_number : int, optional
            Direct symmetry number to use. If None, will be determined
            from point_group or defaulted to 1 (C1).
        point_group : str, optional
            Point group symbol. Used to look up symmetry number if
            symmetry_number not provided.
        warn_on_assumptions : bool
            If True, warn when assuming C1 symmetry (default behavior).
        """
        self.warn_on_assumptions = warn_on_assumptions
        self._symmetry_number: int | None = None
        self._point_group: str | None = None

        if symmetry_number is not None:
            self._symmetry_number = symmetry_number
        elif point_group is not None:
            try:
                self._symmetry_number = get_point_group_symmetry_number(point_group)
                self._point_group = point_group
            except ValueError as e:
                if warn_on_assumptions:
                    warnings.warn(
                        f"Could not determine symmetry number from point group '{point_group}': {e}. "
                        "Using C1 symmetry (symmetry_number=1).",
                        UserWarning,
                        stacklevel=2,
                    )
                self._symmetry_number = 1
                self._point_group = "C1"
        else:
            # Default to C1
            if warn_on_assumptions:
                warnings.warn(
                    "No symmetry number or point group provided. "
                    "Assuming C1 symmetry (symmetry_number=1). "
                    "For more accurate entropies, specify the correct symmetry number.",
                    UserWarning,
                    stacklevel=2,
                )
            self._symmetry_number = 1
            self._point_group = "C1"

    @property
    def symmetry_number(self) -> int:
        """Get the symmetry number."""
        if self._symmetry_number is None:
            self._symmetry_number = 1  # Should not happen, but safety fallback
        return self._symmetry_number

    @property
    def point_group(self) -> str:
        """Get the point group."""
        if self._point_group is None:
            self._point_group = "C1"
        return self._point_group

    def get_rotational_symmetry_number(
        self,
        linear: bool = False,
    ) -> int:
        """Get rotational symmetry number for entropy corrections.

        Parameters
        ----------
        linear : bool
            Whether the molecule is linear

        Returns
        -------
        int
            Rotational symmetry number
        """
        # For linear molecules, external symmetry number is halved
        # compared to non-linear molecules
        if linear:
            return self.symmetry_number // 2 if self.symmetry_number > 1 else 1
        return self.symmetry_number

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"SymmetryHandler(symmetry_number={self.symmetry_number}, "
            f"point_group='{self.point_group}')"
        )
