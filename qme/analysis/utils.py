"""Shared utilities for analysis module.

This module provides common validation and utility functions used across
the analysis module to reduce code duplication and simplify logic.
"""

from __future__ import annotations

from typing import Any

from ase import Atoms


def validate_indices(atoms: Atoms, indices: list[int] | None) -> list[int]:
    """Validate and normalize atom indices.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object
    indices : list[int] | None
        Atom indices to validate. If None, returns all indices.

    Returns
    -------
    list[int]
        Validated list of unique atom indices

    Raises
    ------
    ValueError
        If indices are invalid (empty, duplicate, or out of bounds)
    """
    if indices is None:
        return list(range(len(atoms)))

    if not isinstance(indices, list) or len(indices) == 0:
        msg = "indices must be a non-empty list"
        raise ValueError(msg)

    if len(set(indices)) != len(indices):
        msg = "indices must be unique"
        raise ValueError(msg)

    if not all(0 <= idx < len(atoms) for idx in indices):
        invalid = [idx for idx in indices if not (0 <= idx < len(atoms))]
        msg = f"indices out of bounds: {invalid} (system has {len(atoms)} atoms)"
        raise ValueError(msg)

    return indices


def get_calculator_property(
    calculator: Any,
    property_name: str,
    atoms: Atoms | None = None,
    default: Any = None,
) -> Any:
    """Get a property from calculator using standard interfaces.

    Tries multiple common patterns for accessing calculator properties:
    1. implemented_properties + get_property()
    2. get_{property_name}()
    3. calculate_{property_name}()

    Parameters
    ----------
    calculator : Any
        Calculator object
    property_name : str
        Name of property to retrieve (e.g., 'hessian', 'frequencies')
    atoms : Atoms, optional
        Atoms object to pass to getter methods
    default : Any, optional
        Default value if property is not available

    Returns
    -------
    Any
        Property value, or default if not available

    Raises
    ------
    AttributeError
        If property is not available and no default provided
    """
    # Try implemented_properties interface
    if hasattr(calculator, "implemented_properties"):
        if property_name in calculator.implemented_properties:
            if hasattr(calculator, "get_property"):
                if atoms is not None:
                    return calculator.get_property(property_name, atoms)
                return calculator.get_property(property_name)

    # Try get_{property_name} pattern
    method_name = f"get_{property_name}"
    if hasattr(calculator, method_name):
        method = getattr(calculator, method_name)
        if atoms is not None:
            return method(atoms)
        return method()

    # Try calculate_{property_name} pattern
    alt_method_name = f"calculate_{property_name}"
    if hasattr(calculator, alt_method_name):
        method = getattr(calculator, alt_method_name)
        if atoms is not None:
            return method(atoms)
        return method()

    # Not found
    if default is not None:
        return default

    msg = f"Calculator does not support property '{property_name}'"
    raise AttributeError(msg)


def has_calculator_property(calculator: Any, property_name: str) -> bool:
    """Check if calculator supports a property.

    This function checks for property support without actually calling the getter,
    which is important for properties (like Hessians) that require atoms to be set.

    Parameters
    ----------
    calculator : Any
        Calculator object
    property_name : str
        Property name to check

    Returns
    -------
    bool
        True if calculator supports the property
    """
    # Check implemented_properties interface first (most reliable, doesn't require calling)
    if hasattr(calculator, "implemented_properties"):
        if property_name in calculator.implemented_properties:
            return True

    # Check for get_{property_name} method
    method_name = f"get_{property_name}"
    if hasattr(calculator, method_name):
        return True

    # Check for calculate_{property_name} method
    alt_method_name = f"calculate_{property_name}"
    return bool(hasattr(calculator, alt_method_name))
