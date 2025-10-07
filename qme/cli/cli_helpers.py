import os
from typing import Dict, List, Optional

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write


def parse_kv_pairs(pairs: List[str]) -> Dict[str, object]:
    """Parse key=value pairs from CLI into a dict with best-effort typing."""
    result: Dict[str, object] = {}
    for item in pairs or []:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        # Try to coerce value types
        if value.lower() in ("true", "false"):
            coerced: object = value.lower() == "true"
        else:
            try:
                if "." in value:
                    coerced = float(value)
                else:
                    coerced = int(value)
            except ValueError:
                coerced = value
        result[key] = coerced
    return result


def load_atoms_from_xyz(path: str) -> Atoms:
    atoms = ase_read(path)
    if isinstance(atoms, list):
        # If multiple frames in XYZ, take the last one by default
        atoms = atoms[-1]
    return atoms


def _coerce_to_atoms(obj) -> Atoms:
    """Best-effort conversion of various result shapes into an ASE Atoms."""
    if isinstance(obj, Atoms):
        return obj
    # Strategy dict result
    if isinstance(obj, dict) and "optimized_atoms" in obj:
        return obj["optimized_atoms"]
    # List/tuple of Atoms (take first)
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], Atoms):
        return obj[0]
    # Path string to an XYZ; try to read
    if isinstance(obj, str) and os.path.exists(obj):
        return ase_read(obj)
    raise TypeError(f"Cannot coerce object of type {type(obj)} to ASE Atoms")


def write_atoms(atoms: Atoms, out_path: Optional[str]) -> Optional[str]:
    if not out_path:
        return None
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    ase_write(out_path, _coerce_to_atoms(atoms))
    return out_path


__all__ = [
    "parse_kv_pairs",
    "load_atoms_from_xyz",
    "write_atoms",
]
