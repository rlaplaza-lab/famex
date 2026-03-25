"""Shared utilities for analysis module."""

from __future__ import annotations

from typing import Any

from ase import Atoms


def validate_indices(atoms: Atoms, indices: list[int] | None) -> list[int]:
    if indices is None:
        return list(range(len(atoms)))
    if not isinstance(indices, list) or len(indices) == 0:
        raise ValueError("indices must be a non-empty list")
    if len(set(indices)) != len(indices):
        raise ValueError("indices must be unique")
    if not all(0 <= idx < len(atoms) for idx in indices):
        invalid = [idx for idx in indices if not (0 <= idx < len(atoms))]
        raise ValueError(f"indices out of bounds: {invalid} (system has {len(atoms)} atoms)")
    return indices


def get_calculator_property(
    calculator: Any,
    property_name: str,
    atoms: Atoms | None = None,
    default: Any = None,
) -> Any:
    if hasattr(calculator, "implemented_properties"):
        if property_name in calculator.implemented_properties:
            if hasattr(calculator, "get_property"):
                if atoms is not None:
                    return calculator.get_property(property_name, atoms)
                return calculator.get_property(property_name)

    method_name = f"get_{property_name}"
    if hasattr(calculator, method_name):
        method = getattr(calculator, method_name)
        if atoms is not None:
            return method(atoms)
        return method()

    alt_method_name = f"calculate_{property_name}"
    if hasattr(calculator, alt_method_name):
        method = getattr(calculator, alt_method_name)
        if atoms is not None:
            return method(atoms)
        return method()

    if default is not None:
        return default

    raise AttributeError(f"Calculator does not support property '{property_name}'")


def has_calculator_property(calculator: Any, property_name: str) -> bool:
    if hasattr(calculator, "implemented_properties"):
        if property_name in calculator.implemented_properties:
            return True
    method_name = f"get_{property_name}"
    if hasattr(calculator, method_name):
        return True
    alt_method_name = f"calculate_{property_name}"
    return bool(hasattr(calculator, alt_method_name))
