"""Finite difference schemes for Hessian calculation.

This module provides the finite difference scheme protocol and implementations
for numerical Hessian calculation. Schemes differ in accuracy and computational cost.

Available schemes:
- ForwardDifferenceScheme (2-point): 1st order accuracy, fast but less accurate
- CentralDifferenceScheme (3-point): 2nd order accuracy, standard choice
- FivePointCentralDifferenceScheme (5-point): 4th order accuracy, high accuracy option
- SevenPointCentralDifferenceScheme (7-point): 6th order accuracy, highest accuracy
"""

from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray


class FiniteDifferenceScheme(Protocol):
    """Protocol for finite difference schemes.

    This protocol allows for extensible finite difference implementations.
    Implemented schemes include 2-point (forward), 3-point (central), and 5-point.

    Examples:
    --------
    Custom schemes can be implemented by creating a class that matches this protocol:

    >>> class CustomScheme:
    ...     def compute_derivative(
    ...         self,
    ...         forces_plus: np.ndarray,
    ...         forces_minus: np.ndarray | None,
    ...         forces_ref: np.ndarray | None,
    ...         delta: float,
    ...         **kwargs: Any,
    ...     ) -> np.ndarray:
    ...         # Custom implementation
    ...         return -(forces_plus - forces_minus) / (2 * delta)

    See Also:
    --------
    CentralDifferenceScheme : 2nd order central difference implementation
    ForwardDifferenceScheme : 1st order forward difference implementation
    FivePointCentralDifferenceScheme : 4th order 5-point stencil implementation
    """

    def compute_derivative(
        self,
        forces_plus: np.ndarray,
        forces_minus: np.ndarray | None,
        forces_ref: np.ndarray | None,
        delta: float,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute Hessian column using finite differences.

        Parameters
        ----------
        forces_plus : np.ndarray
            Forces at +delta displacement
        forces_minus : np.ndarray, optional
            Forces at -delta displacement (for central differences)
        forces_ref : np.ndarray, optional
            Forces at reference geometry (for forward differences)
        delta : float
            Displacement step size
        **kwargs : Any
            Additional force arrays for higher-order schemes

        Returns:
        -------
        np.ndarray
            Hessian column: -∂F/∂x

        """
        ...


class CentralDifferenceScheme:
    """Central difference finite difference scheme (2nd order accuracy).

    This scheme computes the derivative using forces at +δ and -δ displacements
    from the reference geometry, providing second-order accuracy.

    Formula: H_ij = -(F_i(+δj) - F_i(-δj)) / (2δ)

    Where F_i are the forces on atom i and δj is the displacement of coordinate j.

    Examples:
    --------
    >>> scheme = CentralDifferenceScheme()
    >>> # forces_plus: forces at +delta displacement
    >>> # forces_minus: forces at -delta displacement
    >>> hessian_col = scheme.compute_derivative(forces_plus, forces_minus, None, delta=0.01)

    Notes:
    -----
    - Requires both positive and negative displacements (2 force calculations per column)
    - More accurate than forward differences but slower
    - Standard choice for most applications
    - Compatible with Richardson extrapolation

    See Also:
    --------
    ForwardDifferenceScheme : Faster but less accurate 1st order scheme
    FivePointCentralDifferenceScheme : Higher accuracy 4th order scheme

    """

    def compute_derivative(
        self,
        forces_plus: np.ndarray,
        forces_minus: np.ndarray | None,
        forces_ref: np.ndarray | None,
        delta: float,
    ) -> np.ndarray:
        """Compute Hessian column using central differences.

        H_ij = (F_i(+δj) - F_i(-δj)) / (2δ)

        """
        if forces_minus is None:
            msg = "Central difference requires forces_minus"
            raise ValueError(msg)
        return cast(NDArray[np.float64], -(forces_plus - forces_minus) / (2 * delta))


class ForwardDifferenceScheme:
    """Forward difference finite difference scheme (1st order accuracy).

    This scheme computes the derivative using forces at the reference geometry
    and at +δ displacement, providing first-order accuracy. This is the fastest
    method but least accurate.

    Formula: H_ij = -(F_i(+δj) - F_i(0)) / δ

    Where F_i(0) are the forces at the reference geometry and F_i(+δj) are
    forces after displacing coordinate j by +δ.

    Examples:
    --------
    >>> scheme = ForwardDifferenceScheme()
    >>> # forces_ref: forces at reference geometry (computed once)
    >>> # forces_plus: forces at +delta displacement
    >>> hessian_col = scheme.compute_derivative(forces_plus, None, forces_ref, delta=0.01)

    Notes:
    -----
    - Requires only positive displacements (1 force calculation per column + 1 reference)
    - Fastest method but least accurate (1st order)
    - Not compatible with Richardson extrapolation
    - Use only when speed is critical and accuracy requirements are low

    See Also:
    --------
    CentralDifferenceScheme : More accurate 2nd order scheme
    FivePointCentralDifferenceScheme : Highest accuracy 4th order scheme

    """

    def compute_derivative(
        self,
        forces_plus: np.ndarray,
        forces_minus: np.ndarray | None,
        forces_ref: np.ndarray | None,
        delta: float,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute Hessian column using forward differences.

        H_ij = (F_i(+δj) - F_i(0)) / δ

        """
        if forces_ref is None:
            msg = "Forward difference requires forces_ref"
            raise ValueError(msg)
        return cast(NDArray[np.float64], -(forces_plus - forces_ref) / delta)


class FivePointCentralDifferenceScheme:
    """5-point central difference finite difference scheme (4th order accuracy).

    This scheme uses a 5-point stencil to compute derivatives with 4th order
    accuracy, providing the highest accuracy among the implemented schemes.
    It requires forces at ±δ and ±2δ displacements.

    Uses the stencil for first derivative:
    ∂f/∂x ≈ [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / (12h)

    For Hessian calculation from forces F = -∇E:
    H = -∂F/∂x = [-F(x+2δ) + 8F(x+δ) - 8F(x-δ) + F(x-2δ)] / (12δ)

    Examples:
    --------
    >>> scheme = FivePointCentralDifferenceScheme()
    >>> # Requires forces at 4 displacement points
    >>> hessian_col = scheme.compute_derivative(
    ...     forces_plus,      # F(+δ)
    ...     forces_minus,     # F(-δ)
    ...     None,             # forces_ref not used
    ...     delta=0.01,
    ...     forces_plus2=forces_plus2,   # F(+2δ)
    ...     forces_minus2=forces_minus2  # F(-2δ)
    ... )

    Notes:
    -----
    - Requires forces at 4 displacement points: ±δ and ±2δ (4 force calculations per column)
    - Highest accuracy (4th order) but slowest
    - Best for high-accuracy applications where computation time is acceptable
    - Compatible with Richardson extrapolation (can achieve 6th+ order effectively)
    - May be sensitive to very large displacements (>2δ approaching bond lengths)

    See Also:
    --------
    CentralDifferenceScheme : Faster 2nd order scheme
    ForwardDifferenceScheme : Fastest 1st order scheme

    """

    def compute_derivative(
        self,
        forces_plus: np.ndarray,
        forces_minus: np.ndarray | None,
        forces_ref: np.ndarray | None,
        delta: float,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute Hessian column using 5-point central differences.

        Parameters
        ----------
        forces_plus : np.ndarray
            Forces at +delta displacement
        forces_minus : np.ndarray
            Forces at -delta displacement
        forces_ref : np.ndarray, optional
            Forces at reference geometry (not used, for protocol compatibility)
        delta : float
            Displacement step size
        **kwargs : Any
            Additional forces at +2delta and -2delta:
            - forces_plus2: Forces at +2*delta displacement
            - forces_minus2: Forces at -2*delta displacement

        Returns:
        -------
        np.ndarray
            Hessian column: -∂F/∂x

        """
        if forces_minus is None:
            msg = "5-point central difference requires forces_minus"
            raise ValueError(msg)
        if "forces_plus2" not in kwargs or "forces_minus2" not in kwargs:
            msg = "5-point central difference requires forces_plus2 and forces_minus2"
            raise ValueError(msg)

        forces_plus2 = kwargs["forces_plus2"]
        forces_minus2 = kwargs["forces_minus2"]

        # 5-point stencil for first derivative of a function f:
        # ∂f/∂x ≈ [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / (12h)
        #
        # For Hessian calculation from forces F = -∇E:
        # - H = ∂²E/∂x² (Hessian of energy)
        # - dF/dx = ∂(-∇E)/∂x = -∂²E/∂x² = -H
        # - Therefore: H = -dF/dx
        #
        # The 3-point scheme computes: -(F_plus - F_minus)/(2*delta) = (F_minus - F_plus)/(2*delta)
        # This is dF/dx (first derivative of forces), and we return it directly as H = -dF/dx.
        #
        # For 5-point, we use the first derivative stencil instead of second derivative
        hessian_col = (-forces_plus2 + 8 * forces_plus - 8 * forces_minus + forces_minus2) / (
            12 * delta
        )

        # This gives dF/dx, so we need to negate to get H = -dF/dx
        return cast(NDArray[np.float64], -hessian_col)


class SevenPointCentralDifferenceScheme:
    """7-point central difference finite difference scheme (6th order accuracy).

    This scheme uses a 7-point stencil to compute derivatives with 6th order
    accuracy, providing even higher accuracy than the 5-point scheme. It requires
    forces at ±δ, ±2δ, and ±3δ displacements.

    Uses the stencil for first derivative:
    ∂f/∂x ≈ [f(x+3h) - 9f(x+2h) + 45f(x+h) - 45f(x-h) + 9f(x-2h) - f(x-3h)] / (60h)

    For Hessian calculation from forces F = -∇E:
    H = -∂F/∂x = -[F(x+3δ) - 9F(x+2δ) + 45F(x+δ) - 45F(x-δ) + 9F(x-2δ) - F(x-3δ)] / (60δ)

    Examples:
    --------
    >>> scheme = SevenPointCentralDifferenceScheme()
    >>> # Requires forces at 6 displacement points
    >>> hessian_col = scheme.compute_derivative(
    ...     forces_plus,      # F(+δ)
    ...     forces_minus,     # F(-δ)
    ...     None,             # forces_ref not used
    ...     delta=0.01,
    ...     forces_plus2=forces_plus2,   # F(+2δ)
    ...     forces_minus2=forces_minus2, # F(-2δ)
    ...     forces_plus3=forces_plus3,   # F(+3δ)
    ...     forces_minus3=forces_minus3  # F(-3δ)
    ... )

    Notes:
    -----
    - Requires forces at 6 displacement points: ±δ, ±2δ, ±3δ
    - Highest accuracy (6th order) but slowest
    - Best for very high-accuracy applications
    - Compatible with Richardson extrapolation (can achieve 8th+ order effectively)
    - May be sensitive to very large displacements

    See Also:
    --------
    FivePointCentralDifferenceScheme : Faster 4th order scheme
    CentralDifferenceScheme : Faster 2nd order scheme
    ForwardDifferenceScheme : Fastest 1st order scheme

    """

    def compute_derivative(
        self,
        forces_plus: np.ndarray,
        forces_minus: np.ndarray | None,
        forces_ref: np.ndarray | None,
        delta: float,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute Hessian column using 7-point central differences.

        Parameters
        ----------
        forces_plus : np.ndarray
            Forces at +delta displacement
        forces_minus : np.ndarray
            Forces at -delta displacement
        forces_ref : np.ndarray, optional
            Forces at reference geometry (not used, for protocol compatibility)
        delta : float
            Displacement step size
        **kwargs : Any
            Additional forces at ±2delta and ±3delta:
            - forces_plus2: Forces at +2*delta displacement
            - forces_minus2: Forces at -2*delta displacement
            - forces_plus3: Forces at +3*delta displacement
            - forces_minus3: Forces at -3*delta displacement

        Returns:
        -------
        np.ndarray
            Hessian column: -∂F/∂x

        """
        if forces_minus is None:
            msg = "7-point central difference requires forces_minus"
            raise ValueError(msg)
        required_kwargs = ["forces_plus2", "forces_minus2", "forces_plus3", "forces_minus3"]
        missing = [kw for kw in required_kwargs if kw not in kwargs]
        if missing:
            msg = f"7-point central difference requires: {', '.join(missing)}"
            raise ValueError(msg)

        forces_plus2 = kwargs["forces_plus2"]
        forces_minus2 = kwargs["forces_minus2"]
        forces_plus3 = kwargs["forces_plus3"]
        forces_minus3 = kwargs["forces_minus3"]

        # 7-point stencil for first derivative:
        # ∂f/∂x ≈ [f(x+3h) - 9f(x+2h) + 45f(x+h) - 45f(x-h) + 9f(x-2h) - f(x-3h)] / (60h)
        #
        # For Hessian calculation from forces F = -∇E:
        # H = -∂F/∂x
        hessian_col = (
            forces_plus3
            - 9 * forces_plus2
            + 45 * forces_plus
            - 45 * forces_minus
            + 9 * forces_minus2
            - forces_minus3
        ) / (60 * delta)

        # This gives dF/dx, so we need to negate to get H = -dF/dx
        return cast(NDArray[np.float64], -hessian_col)


__all__ = [
    "FiniteDifferenceScheme",
    "CentralDifferenceScheme",
    "ForwardDifferenceScheme",
    "FivePointCentralDifferenceScheme",
    "SevenPointCentralDifferenceScheme",
]
